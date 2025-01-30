import torch
import torch.nn as nn
import torch.nn.functional as F


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int = 64,  # Dimension per attention head
        max_seq_len: int = 2048,  # Maximum sequence length
        base: int = 10000,  # Base for the angle calculations
        device: str = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Create cache for position frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Create position sequence
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x: torch.Tensor, seq_len: int):
        # Return early if cache is valid
        if seq_len <= self._seq_len_cached:
            return

        # Update cache size
        self._seq_len_cached = seq_len

        # Create position sequence
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        # Calculate position frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Calculate embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos()  # [None, None, :, :]
        self._sin_cached = emb.sin()  # [None, None, :, :]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_heads, seq_len, head_dim = q.shape

        # Update cos/sin tables if needed
        self._update_cos_sin_tables(q, seq_len)

        # Get cos and sin for current sequence
        cos = (
            self._cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        )  # Shape: [1, 1, seq_len, dim]
        sin = (
            self._sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(0)
        )  # Shape: [1, 1, seq_len, dim]

        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Apply rotary embeddings to q and k
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
