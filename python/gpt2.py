"""GPT-2 Small forward pass implemented with XTL.

Architecture (gpt2-small):
  n_layer=12, n_head=12, n_embd=768, n_ctx=1024, n_vocab=50257

Weight layout note: GPT-2 uses Conv1D which stores weights as [in, out],
so all linear projections are x @ w (not x @ w.T).
"""

import numpy as np
import tiktoken
import xtl


class GPT2:
    n_layer  = 12
    n_head   = 12
    n_embd   = 768
    n_ctx    = 1024
    n_vocab  = 50257

    def __init__(self, weights_path: str):
        self.head_dim = self.n_embd // self.n_head
        self._load_weights(weights_path)
        self._enc = tiktoken.get_encoding("gpt2")

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

    def _attn(self, x, blk, mask):
        """Causal multi-head self-attention."""
        T     = x.shape[0]
        D     = self.head_dim
        scale = D ** -0.5

        head_nps = []
        for h in range(self.n_head):
            q = x @ blk["qw"][h] + blk["qb"][h]  # [T, D]
            k = x @ blk["kw"][h] + blk["kb"][h]  # [T, D]
            v = x @ blk["vw"][h] + blk["vb"][h]  # [T, D]

            # k^T: [T, D] -> contiguous copy then transpose -> [D, T]
            kt = k.contiguous()
            kt.transpose()

            scores = (q @ kt).mul_scalar(scale)  # [T, T]
            scores = scores + mask               # add -inf to future positions
            scores = scores.softmax(1)           # row-wise softmax

            head_nps.append(xtl.to_numpy(scores @ v))  # [T, D]

        # Merge heads: numpy concat then back to XTL
        merged = xtl.from_numpy(np.concatenate(head_nps, axis=1))  # [T, C]
        return merged @ blk["c_proj_w"] + blk["c_proj_b"]

    def _ffn(self, x, blk):
        """Feed-forward network: linear -> GELU -> linear."""
        return (x @ blk["c_fc_w"] + blk["c_fc_b"]).gelu_apprx() \
               @ blk["c_proj_w_m"] + blk["c_proj_b_m"]

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(self, token_ids: list[int]):
        T = len(token_ids)

        # Causal mask: 0 where attending, -1e10 where blocked
        mask_np = np.where(
            np.tril(np.ones((T, T), dtype=np.float32)), 0.0, -1e10
        ).astype(np.float32)
        mask = xtl.from_numpy(mask_np)

        # Embeddings: token + position
        pos = list(range(T))
        x = self.wte.gather(token_ids) + self.wpe.gather(pos)

        # Transformer blocks
        for blk in self.blocks:
            x = x + self._attn(self._ln(x, blk["ln_1_w"], blk["ln_1_b"]), blk, mask)
            x = x + self._ffn(self._ln(x, blk["ln_2_w"], blk["ln_2_b"]), blk)

        # Final norm + LM head (weight-tied with wte)
        x = self._ln(x, self.ln_f_w, self.ln_f_b)
        return x @ self.wte_T  # [T, 50257]

    # ── generation ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, max_new_tokens: int = 20,
                 temperature: float = 1.0, print_prefix: str | None = None) -> str:
        tokens = self._enc.encode(prompt)

        if print_prefix is not None:
            print(print_prefix, end="", flush=True)

        for _ in range(max_new_tokens):
            ctx = tokens[-self.n_ctx:]
            logits = self.forward(ctx)

            last = xtl.to_numpy(logits)[-1]  # [50257]
            if temperature == 1.0:
                next_tok = int(np.argmax(last))
            else:
                last = last / temperature
                last -= last.max()
                probs = np.exp(last)
                probs /= probs.sum()
                next_tok = int(np.random.choice(len(probs), p=probs))

            tokens.append(next_tok)
            if print_prefix is not None:
                print(self._enc.decode([next_tok]), end="", flush=True)

        if print_prefix is not None:
            print()

        return self._enc.decode(tokens)


if __name__ == "__main__":
    import sys
    import time

    weights = "model.safetensors"
    if len(sys.argv) > 1:
        weights = sys.argv[1]

    print(f"Loading GPT-2 weights from {weights} ...", flush=True)
    t0 = time.time()
    model = GPT2(weights)
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
        history = model.generate(history, max_new_tokens=40, print_prefix="GPT-2:")
        print(f"  ({time.time()-t0:.1f}s)\n")
