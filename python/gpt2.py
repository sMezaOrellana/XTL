"""GPT-2 Small forward pass implemented with XTL.

Architecture (gpt2-small):
  n_layer=12, n_head=12, n_embd=768, n_ctx=1024, n_vocab=50257

Weight layout note: GPT-2 uses Conv1D which stores weights as [in, out],
so all linear projections are x @ w (not x @ w.T).
"""

import numpy as np  # used only in _load_weights for weight preprocessing
import tiktoken
import xtl


CONFIGS = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
    "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
}

class GPT2:
    n_ctx   = 1024
    n_vocab = 50257

    def __init__(self, weights_path: str, model: str = "gpt2"):
        cfg = CONFIGS[model]
        self.n_layer = cfg["n_layer"]
        self.n_head  = cfg["n_head"]
        self.n_embd  = cfg["n_embd"]
        self.head_dim = self.n_embd // self.n_head
        self._load_weights(weights_path)
        self._enc = tiktoken.get_encoding("gpt2")
        self.kv_cache = None   # set by prefill()
        self._seq_len = 0

    # ── weight loading ────────────────────────────────────────────────────────

    def _load_weights(self, path: str):
        from safetensors.numpy import load_file
        raw = load_file(path)

        def np32(key):
            return raw[key].astype(np.float32)

        def xt(arr):
            return xtl.from_numpy(arr)

        # Token / position embeddings (keys have no "transformer." prefix in this file)
        wte_np = np32("wte.weight")                       # [50257, 768]
        self.wte   = xt(wte_np)
        self.wte_T = xt(np.ascontiguousarray(wte_np.T))  # [768, 50257] for LM head
        self.wpe   = xt(np32("wpe.weight"))               # [1024, 768]

        # Final layer norm
        self.ln_f_w = xt(np32("ln_f.weight"))
        self.ln_f_b = xt(np32("ln_f.bias"))

        C, H, D = self.n_embd, self.n_head, self.head_dim
        self.blocks = []

        for i in range(self.n_layer):
            p = f"h.{i}"

            # Fused QKV: [C, 3C] → split into Q, K, V per head
            attn_w = np32(f"{p}.attn.c_attn.weight")   # [768, 2304]
            attn_b = np32(f"{p}.attn.c_attn.bias")     # [2304]
            qw, kw, vw = np.split(attn_w, 3, axis=1)  # each [C, C]
            qb, kb, vb = np.split(attn_b, 3)           # each [C]

            self.blocks.append({
                "ln_1_w": xt(np32(f"{p}.ln_1.weight")),
                "ln_1_b": xt(np32(f"{p}.ln_1.bias")),
                # Per-head weight slices: [C, D] and bias [1, D]
                "qw": [xt(qw[:, h*D:(h+1)*D]) for h in range(H)],
                "kw": [xt(kw[:, h*D:(h+1)*D]) for h in range(H)],
                "vw": [xt(vw[:, h*D:(h+1)*D]) for h in range(H)],
                "qb": [xt(qb[h*D:(h+1)*D].reshape(1, D)) for h in range(H)],
                "kb": [xt(kb[h*D:(h+1)*D].reshape(1, D)) for h in range(H)],
                "vb": [xt(vb[h*D:(h+1)*D].reshape(1, D)) for h in range(H)],
                # Output projection
                "c_proj_w": xt(np32(f"{p}.attn.c_proj.weight")),  # [C, C]
                "c_proj_b": xt(np32(f"{p}.attn.c_proj.bias")),    # [C]
                # FFN
                "ln_2_w":      xt(np32(f"{p}.ln_2.weight")),
                "ln_2_b":      xt(np32(f"{p}.ln_2.bias")),
                "c_fc_w":      xt(np32(f"{p}.mlp.c_fc.weight")),       # [C, 4C]
                "c_fc_b":      xt(np32(f"{p}.mlp.c_fc.bias")),         # [4C]
                "c_proj_w_m":  xt(np32(f"{p}.mlp.c_proj.weight")),     # [4C, C]
                "c_proj_b_m":  xt(np32(f"{p}.mlp.c_proj.bias")),       # [C]
            })

    # ── primitives ────────────────────────────────────────────────────────────

    def _ln(self, x, w, b):
        """Layer-norm over axis=1 (hidden dim) + affine: γ*(x-μ)/σ + β."""
        return x.layer_norm(1) * w + b

    def _attn_prefill(self, x, blk, mask):
        """Full causal attention over the prompt. Returns (output, kv_per_head).
        kv_per_head is a list of (k, v) tensors, each [T, D]."""
        D     = self.head_dim
        scale = D ** -0.5

        heads  = []
        kv_out = []
        for h in range(self.n_head):
            q = x @ blk["qw"][h] + blk["qb"][h]  # [T, D]
            k = x @ blk["kw"][h] + blk["kb"][h]  # [T, D]
            v = x @ blk["vw"][h] + blk["vb"][h]  # [T, D]

            kv_out.append((k, v))

            kt = k.contiguous()
            kt.transpose()                        # [D, T]

            scores = (q @ kt).mul_scalar(scale) + mask
            scores = scores.softmax(1)

            heads.append(scores @ v)              # [T, D]

        merged = xtl.cat(heads, axis=1)           # [T, C]
        return merged @ blk["c_proj_w"] + blk["c_proj_b"], kv_out

    def _attn_decode(self, x, blk, kv_cache_layer):
        """Single-token attention using KV cache.
        x: [1, C]. Returns (output [1, C], updated kv_per_head)."""
        D     = self.head_dim
        scale = D ** -0.5

        heads  = []
        kv_out = []
        for h in range(self.n_head):
            q     = x @ blk["qw"][h] + blk["qb"][h]   # [1, D]
            k_new = x @ blk["kw"][h] + blk["kb"][h]   # [1, D]
            v_new = x @ blk["vw"][h] + blk["vb"][h]   # [1, D]

            k_all = xtl.cat([kv_cache_layer[h][0], k_new], axis=0)  # [T+1, D]
            v_all = xtl.cat([kv_cache_layer[h][1], v_new], axis=0)  # [T+1, D]
            kv_out.append((k_all, v_all))

            kt = k_all.contiguous()
            kt.transpose()                             # [D, T+1]

            scores = (q @ kt).mul_scalar(scale)        # [1, T+1]  (causal by construction)
            scores = scores.softmax(1)

            heads.append(scores @ v_all)               # [1, D]

        merged = xtl.cat(heads, axis=1)                # [1, C]
        return merged @ blk["c_proj_w"] + blk["c_proj_b"], kv_out

    def _ffn(self, x, blk):
        """Feed-forward network: linear -> GELU -> linear."""
        return (x @ blk["c_fc_w"] + blk["c_fc_b"]).gelu_apprx() \
               @ blk["c_proj_w_m"] + blk["c_proj_b_m"]

    # ── forward pass (kept for tests / external callers) ──────────────────────

    def forward(self, token_ids: list[int]):
        T    = len(token_ids)
        mask = xtl.causal_mask(T)
        pos  = list(range(T))
        x    = self.wte.gather(token_ids) + self.wpe.gather(pos)

        for blk in self.blocks:
            attn_out, _ = self._attn_prefill(
                self._ln(x, blk["ln_1_w"], blk["ln_1_b"]), blk, mask)
            x = x + attn_out
            x = x + self._ffn(self._ln(x, blk["ln_2_w"], blk["ln_2_b"]), blk)

        x = self._ln(x, self.ln_f_w, self.ln_f_b)
        return x @ self.wte_T  # [T, 50257]

    # ── KV-cache forward pass ─────────────────────────────────────────────────

    def prefill(self, token_ids: list[int]):
        """Run full forward on the prompt, building the KV cache.
        Returns the last-token hidden state [1, C] (after final layer-norm)."""
        T    = len(token_ids)
        mask = xtl.causal_mask(T)
        pos  = list(range(T))
        x    = self.wte.gather(token_ids) + self.wpe.gather(pos)

        self.kv_cache = []
        for blk in self.blocks:
            attn_out, kv = self._attn_prefill(
                self._ln(x, blk["ln_1_w"], blk["ln_1_b"]), blk, mask)
            x = x + attn_out
            x = x + self._ffn(self._ln(x, blk["ln_2_w"], blk["ln_2_b"]), blk)
            self.kv_cache.append(kv)

        self._seq_len = T
        last = self._ln(x, self.ln_f_w, self.ln_f_b).gather([T - 1])  # [1, C]
        return last

    def decode_one(self, token_id: int):
        """Run one decode step for a single token using the KV cache.
        Returns hidden state [1, C] (after final layer-norm)."""
        x = self.wte.gather([token_id]) + self.wpe.gather([self._seq_len])  # [1, C]

        new_cache = []
        for i, blk in enumerate(self.blocks):
            attn_out, kv = self._attn_decode(
                self._ln(x, blk["ln_1_w"], blk["ln_1_b"]), blk, self.kv_cache[i])
            x = x + attn_out
            x = x + self._ffn(self._ln(x, blk["ln_2_w"], blk["ln_2_b"]), blk)
            new_cache.append(kv)

        self.kv_cache = new_cache
        self._seq_len += 1
        return self._ln(x, self.ln_f_w, self.ln_f_b)  # [1, C]

    # ── generation ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = 20,
                 temperature: float = 1.0, print_prefix: str | None = None,
                 stop: list[str] | None = None) -> str:
        tokens = self._enc.encode(prompt)
        if len(tokens) > self.n_ctx:
            tokens = tokens[-self.n_ctx:]
        prompt_len = len(tokens)

        lookback = max((len(s) for s in stop), default=0) if stop else 0

        if print_prefix is not None:
            print(print_prefix, end="", flush=True)

        printed = 0

        # Prefill: one full forward pass over the whole prompt
        x = self.prefill(tokens)

        for _ in range(max_new_tokens):
            last = (x @ self.wte_T).reshape([self.n_vocab])  # [n_vocab]
            if temperature == 1.0:
                next_tok = xtl.argmax(last)
            else:
                probs = last.div_scalar(temperature).softmax(0)
                next_tok = xtl.sample(probs)

            tokens.append(next_tok)
            generated = self._enc.decode(tokens[prompt_len:])

            if stop:
                hit = next((s for s in stop if s in generated), None)
                if hit:
                    clean = generated[:generated.index(hit)]
                    if print_prefix is not None:
                        print(clean[printed:], end="", flush=True)
                    tokens = tokens[:prompt_len] + self._enc.encode(clean)
                    break

            # Stream everything except the last `lookback` chars
            if print_prefix is not None:
                safe = max(0, len(generated) - lookback)
                print(generated[printed:safe], end="", flush=True)
                printed = safe

            # Decode next token using KV cache (only processes 1 token)
            x = self.decode_one(next_tok)

        else:
            # Max tokens reached — flush the held-back buffer
            if print_prefix is not None:
                print(self._enc.decode(tokens[prompt_len:])[printed:], end="", flush=True)

        if print_prefix is not None:
            print()

        return self._enc.decode(tokens)


if __name__ == "__main__":
    import sys
    import time

    weights = "model.safetensors"
    model_name = "gpt2"
    if len(sys.argv) > 1:
        weights = sys.argv[1]
    if len(sys.argv) > 2:
        model_name = sys.argv[2]

    print(f"Loading {model_name} weights from {weights} ...", flush=True)
    t0 = time.time()
    model = GPT2(weights, model=model_name)
    print(f"Loaded in {time.time()-t0:.1f}s")
    print("Type your prompt and press Enter. Ctrl+C or Ctrl+D to quit.\n")

    history = ""

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user:
            continue

        # Q:/A: format signals GPT-2 to answer rather than complete the question
        history += f"Q: {user}\nA:"
        t0 = time.time()
        history = model.generate(history, max_new_tokens=60, print_prefix="GPT-2:",
                                 stop=["\nQ:", "\n\n"])
        print(f"  ({time.time()-t0:.1f}s)\n")
