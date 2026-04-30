"""eSASRec — Transformer-based sequential CG.

Vanilla SASRec (Kang & McAuley, 2018) with sampled-softmax loss. We chose
this rather than the rectools-mts implementation to keep the dep tree small
and the code transparent for hyperparam sweeps. The "e" prefix in the
roadmap refers to the *enhanced* variant from the antklen
recsys_challenge_2025 setup; the architecture below covers the same
mechanics (causal masked self-attention + sampled softmax).

Pipeline:
    - tokenise items as 1..n_items (0 reserved for padding)
    - turn ``positive_listens(train)`` into per-user chronological sequences
    - auto-regressive next-item prediction with random negatives
    - inference: feed the user's last K tokens, take the last position's
      hidden state, score the full item embedding table, top-N

For 50m we mostly use this for *smoke-testing* the implementation; serious
training on the full 500m sequence corpus runs on the server (Phase C),
ideally on 2×A100 with DDP and ~10× more epochs.

Output of ``recommend`` matches the standard CG contract:
``uid, item_id, score, esasrec_rank``.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import positive_listens, to_sequential
from src.models.base import BaseModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inner module
# ---------------------------------------------------------------------------


class _SASRec(nn.Module):
    """Standard pre-norm Transformer encoder over an item-token sequence."""

    def __init__(
        self,
        n_items: int,
        emb_dim: int,
        max_len: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
        # 0 = padding; real items occupy indices 1..n_items
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_in = nn.LayerNorm(emb_dim)
        self.norm_out = nn.LayerNorm(emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm = more stable for small data
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) int64 → hidden: (B, L, D) float32."""
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.norm_in(self.dropout(x))
        pad_mask = seq == 0  # (B, L) — True means pad / ignore
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1,
        )
        out = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        return self.norm_out(out)

    def all_item_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Score every real item (skips index 0 = pad). Output (..., n_items)."""
        return hidden @ self.item_emb.weight[1:].T


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class _SeqDataset(Dataset):
    """Sliding-window auto-regressive sequences.

    For each user-history of token ids (1..n_items) we keep the last
    ``max_len + 1`` tokens, left-pad if shorter, then split:

        x = padded[:-1]   (input)
        y = padded[1:]    (target — predict next token)

    Padding is index 0; loss masks positions where ``y == 0``.
    """

    def __init__(self, sequences: list[list[int]], max_len: int):
        self.sequences = sequences
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx][-(self.max_len + 1):]
        pad = (self.max_len + 1) - len(seq)
        padded = [0] * pad + seq
        x = torch.tensor(padded[:-1], dtype=torch.int64)
        y = torch.tensor(padded[1:], dtype=torch.int64)
        return x, y


# ---------------------------------------------------------------------------
# BaseModel impl
# ---------------------------------------------------------------------------


class ESASRecModel(BaseModel):
    """Sequential SASRec CG with sampled-softmax training.

    Use small overrides (``max_epochs=1, emb_dim=64, batch_size=64``) for
    local smoke-tests; production hyperparams target an 8-12h server run.

    Pickling is supported — the inner ``_SASRec`` lives on the configured
    device but ``__getstate__`` moves it to CPU before serialisation, and
    ``__setstate__`` puts it back on the auto-resolved device on load. So
    the cache is portable across CPU/MPS/CUDA hosts.
    """

    def __init__(
        self,
        name: str = "esasrec",
        n_cand: int = 200,
        emb_dim: int = 256,
        n_blocks: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        sequence_max_len: int = 200,
        n_negatives: int = 256,
        max_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 128,
        device: str = "auto",
        random_state: int = 42,
    ):
        self.name = name
        self.n_cand = n_cand
        self.emb_dim = emb_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.sequence_max_len = sequence_max_len
        self.n_negatives = n_negatives
        self.max_epochs = max_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state

        self._model: Optional[_SASRec] = None
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: Optional[np.ndarray] = None  # (n_items + 1,)
        self._user_history: dict[int, list[int]] = {}    # uid -> token list

    # ── pickle helpers ──────────────────────────────────────────────────
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self._model is not None:
            state["_model"] = self._model.cpu()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self._model is not None:
            self._model.to(self._resolve_device())

    # ── helpers ─────────────────────────────────────────────────────────
    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ── fit ─────────────────────────────────────────────────────────────
    def fit(self, train: pl.DataFrame, **kwargs) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        pos = positive_listens(train).select(["uid", "item_id", "timestamp"])
        log.info(
            "ESASRec.fit: %d positive listens, %d users, %d items",
            len(pos), pos["uid"].n_unique(), pos["item_id"].n_unique(),
        )

        # 1. Build item index. Tokens 1..n_items; 0 reserved for padding.
        items_sorted = sorted(pos["item_id"].unique().to_list())
        self._item_to_idx = {int(it): i + 1 for i, it in enumerate(items_sorted)}
        idx_to_item = np.zeros(len(items_sorted) + 1, dtype=np.int64)
        idx_to_item[0] = -1  # sentinel, never returned by recommend
        for it, i in self._item_to_idx.items():
            idx_to_item[i] = it
        self._idx_to_item = idx_to_item
        n_items = len(items_sorted)
        log.info("ESASRec: vocab=%d (1..%d, 0=pad)", n_items, n_items)

        # 2. Per-user chronological sequences
        seq_df = to_sequential(pos)
        sequences: list[list[int]] = []
        user_ids: list[int] = []
        for row in seq_df.iter_rows(named=True):
            tokens = [self._item_to_idx[int(it)] for it in row["item_ids"]]
            if len(tokens) < 2:
                # Need at least one input + one target token.
                continue
            sequences.append(tokens)
            user_ids.append(int(row["uid"]))

        # Cache last K tokens per user for inference.
        for uid, tokens in zip(user_ids, sequences):
            self._user_history[uid] = tokens[-self.sequence_max_len:]
        log.info("ESASRec: %d trainable sequences (>=2 tokens)", len(sequences))

        # 3. Build model + optimizer
        device = self._resolve_device()
        log.info("ESASRec: training on device=%s", device)
        model = _SASRec(
            n_items=n_items,
            emb_dim=self.emb_dim,
            max_len=self.sequence_max_len,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            dropout=self.dropout,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        loader = DataLoader(
            _SeqDataset(sequences, max_len=self.sequence_max_len),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        # 4. Training loop — sampled softmax (BCE on positive vs random negatives)
        for epoch in range(self.max_epochs):
            model.train()
            running_loss = 0.0
            n_steps = 0
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                hidden = model(x)                                # (B, L, D)
                non_pad = y != 0                                 # (B, L)
                if not non_pad.any():
                    continue

                pos_emb = model.item_emb(y)                      # (B, L, D)
                neg_idx = torch.randint(
                    1, n_items + 1,
                    (x.size(0), x.size(1), self.n_negatives),
                    device=device,
                )
                neg_emb = model.item_emb(neg_idx)                # (B, L, n_neg, D)

                pos_score = (hidden * pos_emb).sum(-1)           # (B, L)
                neg_score = (hidden.unsqueeze(2) * neg_emb).sum(-1)  # (B, L, n_neg)

                pos_loss = -torch.nn.functional.logsigmoid(pos_score)
                neg_loss = -torch.nn.functional.logsigmoid(-neg_score).mean(-1)
                loss_per = (pos_loss + neg_loss) * non_pad.float()
                loss = loss_per.sum() / non_pad.float().sum().clamp(min=1.0)

                opt.zero_grad()
                loss.backward()
                opt.step()

                running_loss += float(loss.item())
                n_steps += 1

            avg = running_loss / max(n_steps, 1)
            log.info("ESASRec epoch %d/%d: loss=%.4f", epoch + 1, self.max_epochs, avg)

        model.eval()
        self._model = model
        log.info("ESASRec fitted (n_items=%d, n_users=%d)",
                 n_items, len(self._user_history))

    # ── recommend ───────────────────────────────────────────────────────
    def recommend(self, users: list[int], n: int = 100, **kwargs) -> pl.DataFrame:
        if self._model is None:
            raise RuntimeError("ESASRec not fitted")
        rank_col = f"{self.name}_rank"

        u_with_history = [u for u in users if u in self._user_history]
        log.info(
            "ESASRec: recommending for %d / %d users (others have no history)",
            len(u_with_history), len(users),
        )
        if not u_with_history:
            return pl.DataFrame(schema={
                "uid": pl.Int64, "item_id": pl.Int64,
                "score": pl.Float32, rank_col: pl.Int32,
            })

        device = next(self._model.parameters()).device
        max_len = self.sequence_max_len
        n_items = len(self._item_to_idx)
        bs = self.batch_size
        top_n = min(n, n_items)

        chunks: list[pl.DataFrame] = []
        with torch.no_grad():
            for start in range(0, len(u_with_history), bs):
                chunk = u_with_history[start:start + bs]
                X = np.zeros((len(chunk), max_len), dtype=np.int64)
                for i, uid in enumerate(chunk):
                    tokens = self._user_history[uid][-max_len:]
                    X[i, -len(tokens):] = tokens
                x = torch.from_numpy(X).to(device)
                hidden = self._model(x)                  # (B, L, D)
                user_h = hidden[:, -1, :]                # take last position (B, D)
                scores = self._model.all_item_scores(user_h)  # (B, n_items)
                top_scores, top_idx = torch.topk(scores, top_n, dim=1)

                top_idx_np = top_idx.cpu().numpy() + 1  # back to 1-based token idx
                top_scores_np = top_scores.cpu().numpy()

                B = len(chunk)
                uid_flat = np.repeat(np.asarray(chunk, dtype=np.int64), top_n)
                item_flat = self._idx_to_item[top_idx_np.ravel()]
                score_flat = top_scores_np.ravel().astype(np.float32)
                rank_flat = np.tile(np.arange(1, top_n + 1, dtype=np.int32), B)
                chunks.append(pl.DataFrame({
                    "uid": uid_flat,
                    "item_id": item_flat,
                    "score": score_flat,
                    rank_col: rank_flat,
                }))

        return pl.concat(chunks)
