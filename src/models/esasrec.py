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
ideally on 2×A100 with DataParallel and ~20 epochs.

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

from src.data.dataset import positive_listens
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
    """Auto-regressive sequences with optional sliding window.

    sequences: list of numpy int32 arrays (per-user token histories).

    Without sliding window (window_stride=None): one sample per user —
    the last max_len+1 tokens. With window_stride set, each user
    contributes multiple overlapping windows of the same size, enabling
    training on the full history rather than only the tail.

    Padding is index 0; loss masks positions where ``y == 0``.
    """

    def __init__(
        self,
        sequences: list[np.ndarray],
        max_len: int,
        window_stride: int | None = None,
    ):
        self.max_len = max_len
        self.sequences = sequences
        # Build (seq_idx, end_pos) index lazily — avoids materialising all windows
        self.index: list[tuple[int, int]] = []
        for i, seq in enumerate(sequences):
            if len(seq) < 2:
                continue
            if window_stride is None:
                self.index.append((i, len(seq)))
            else:
                last_added = -1
                for end in range(max_len + 1, len(seq) + 1, window_stride):
                    self.index.append((i, end))
                    last_added = end
                # Always include the very last window (covers short sequences too)
                if last_added != len(seq):
                    self.index.append((i, len(seq)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        seq_idx, end = self.index[idx]
        window = self.sequences[seq_idx][max(0, end - (self.max_len + 1)):end]
        pad = (self.max_len + 1) - len(window)
        padded = np.concatenate(
            [np.zeros(pad, dtype=np.int64), window.astype(np.int64)]
        )
        x = torch.from_numpy(padded[:-1].copy())
        y = torch.from_numpy(padded[1:].copy())
        return x, y


# ---------------------------------------------------------------------------
# BaseModel impl
# ---------------------------------------------------------------------------


class ESASRecModel(BaseModel):
    """Sequential SASRec CG with sampled-softmax training.

    Key improvements vs naive stub:
    - sequences stored as numpy int32 (7× less RAM than Python list[list[int]])
    - item→token mapping via Polars join (vectorised, not O(N) Python dict loop)
    - n_negatives=16 (was 256; at batch=512 that was 26.8 GB per step → OOM)
    - batch_size=512 (was 128; A100 80 GB easily holds this)
    - AdamW with weight_decay + gradient clipping
    - early stopping on validation loss with best-state checkpoint
    - optional sliding window (window_stride param)
    - separate infer_batch_size to avoid OOM on all-item scoring (9.4M items)
    - DataParallel when multiple CUDA GPUs are available
    - AMP / BF16 autocast on CUDA (halves VRAM, ~2× speedup on A100)

    Pickling is supported — ``__getstate__`` unwraps DataParallel and moves
    weights to CPU; ``__setstate__`` restores DataParallel if applicable.
    """

    name: str = "esasrec"

    def __init__(
        self,
        name: str = "esasrec",
        n_cand: int = 200,
        # Architecture
        emb_dim: int = 256,
        n_blocks: int = 4,
        n_heads: int = 8,
        dropout: float = 0.2,
        sequence_max_len: int = 200,
        # Training
        n_negatives: int = 16,
        max_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 512,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        # Early stopping
        patience: int = 5,
        val_pct: float = 0.1,
        # DataLoader
        num_workers: int = 4,
        # AMP
        use_amp: bool = True,
        # Sliding window
        window_stride: int | None = None,
        # Inference
        infer_batch_size: int = 64,
        # Device / seed
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
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.val_pct = val_pct
        self.num_workers = num_workers
        self.use_amp = use_amp
        self.window_stride = window_stride
        self.infer_batch_size = infer_batch_size
        self.device = device
        self.random_state = random_state

        self._model: Optional[nn.Module] = None  # _SASRec or DataParallel(_SASRec)
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: Optional[np.ndarray] = None  # (n_items + 1,)
        self._user_history: dict[int, np.ndarray] = {}  # uid -> int32 tokens

    # ── pickle helpers ──────────────────────────────────────────────────
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self._model is not None:
            inner = self._unwrap(self._model)
            state["_model"] = inner.cpu()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self._model is not None:
            dev = self._resolve_device()
            self._model = self._model.to(dev)
            if torch.cuda.device_count() > 1 and dev.startswith("cuda"):
                self._model = nn.DataParallel(self._model)

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, nn.DataParallel) else model

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

        # 1. Build item vocabulary. Tokens 1..n_items; 0 reserved for padding.
        items_sorted = sorted(pos["item_id"].unique().to_list())
        self._item_to_idx = {int(it): i + 1 for i, it in enumerate(items_sorted)}
        idx_to_item = np.zeros(len(items_sorted) + 1, dtype=np.int64)
        idx_to_item[0] = -1  # sentinel, never returned by recommend
        for it, i in self._item_to_idx.items():
            idx_to_item[i] = it
        self._idx_to_item = idx_to_item
        n_items = len(items_sorted)
        log.info("ESASRec: vocab=%d (1..%d, 0=pad)", n_items, n_items)

        # 2. Vectorised item→token mapping via Polars join (avoids O(N) Python dict loop).
        #    sort_by inside agg guarantees chronological order regardless of group_by hash.
        item_vocab_df = pl.DataFrame({
            "item_id": items_sorted,
            "token": np.arange(1, n_items + 1, dtype=np.int32),
        }).with_columns(pl.col("item_id").cast(pos["item_id"].dtype))

        seq_df = (
            pos
            .join(item_vocab_df, on="item_id")
            .group_by("uid")
            .agg(
                pl.col("token").sort_by("timestamp").alias("tokens"),
            )
            .sort("uid")
        )

        # Flat int32 numpy array split by lengths — 7× less RAM than Python list[list[int]]
        flat: np.ndarray = seq_df["tokens"].explode().cast(pl.Int32).to_numpy()
        lengths: np.ndarray = seq_df["tokens"].list.len().to_numpy()
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        all_uid: np.ndarray = seq_df["uid"].to_numpy()

        sequences: list[np.ndarray] = []
        for i in range(len(seq_df)):
            seq = flat[offsets[i]:offsets[i + 1]]
            if len(seq) < 2:
                continue
            sequences.append(seq)
            # Store last max_len tokens for inference
            self._user_history[int(all_uid[i])] = seq[-self.sequence_max_len:]

        log.info("ESASRec: %d trainable sequences (>=2 tokens)", len(sequences))

        # 3. Train/val split for early stopping
        rng = np.random.default_rng(self.random_state)
        n_val = max(1, int(len(sequences) * self.val_pct))
        val_idx_set = set(rng.choice(len(sequences), n_val, replace=False).tolist())
        train_seqs = [s for i, s in enumerate(sequences) if i not in val_idx_set]
        val_seqs = [s for i, s in enumerate(sequences) if i in val_idx_set]
        log.info(
            "ESASRec: train=%d sequences, val=%d sequences",
            len(train_seqs), len(val_seqs),
        )

        # 4. Build model
        device = self._resolve_device()
        log.info("ESASRec: training on device=%s", device)
        sasrec = _SASRec(
            n_items=n_items,
            emb_dim=self.emb_dim,
            max_len=self.sequence_max_len,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            dropout=self.dropout,
        ).to(device)

        use_dp = torch.cuda.device_count() > 1 and device.startswith("cuda")
        model: nn.Module = nn.DataParallel(sasrec) if use_dp else sasrec
        if use_dp:
            log.info("ESASRec: using DataParallel on %d GPUs", torch.cuda.device_count())

        use_amp = self.use_amp and device.startswith("cuda")
        if use_amp:
            log.info("ESASRec: BF16 AMP enabled")

        opt = torch.optim.AdamW(
            sasrec.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # DataLoader kwargs: num_workers>0 needs fork-safe sequences; pin_memory speeds H2D
        _ldr_kw: dict = dict(
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=device.startswith("cuda"),
        )
        # num_workers > 0 requires picklable dataset — numpy arrays are fine
        if self.num_workers > 0:
            _ldr_kw.update(num_workers=self.num_workers, persistent_workers=True)

        train_loader = DataLoader(
            _SeqDataset(train_seqs, max_len=self.sequence_max_len, window_stride=self.window_stride),
            **_ldr_kw,
        )
        val_loader = DataLoader(
            _SeqDataset(val_seqs, max_len=self.sequence_max_len),  # no sliding window for val
            **{**_ldr_kw, "shuffle": False},
        )

        n_train_windows = len(train_loader.dataset)  # type: ignore[arg-type]
        log.info(
            "ESASRec: %d training windows (window_stride=%s)",
            n_train_windows, self.window_stride,
        )

        # 5. Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict | None = None

        for epoch in range(self.max_epochs):
            # ── train ──
            model.train()
            running_loss = 0.0
            n_steps = 0
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                opt.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_amp):
                    hidden = model(x)                            # (B, L, D)
                    non_pad = y != 0                             # (B, L)
                    if not non_pad.any():
                        continue

                    pos_emb = sasrec.item_emb(y)                 # (B, L, D)
                    neg_idx = torch.randint(
                        1, n_items + 1,
                        (x.size(0), x.size(1), self.n_negatives),
                        device=device,
                    )
                    neg_emb = sasrec.item_emb(neg_idx)           # (B, L, n_neg, D)

                    pos_score = (hidden * pos_emb).sum(-1)       # (B, L)
                    neg_score = (hidden.unsqueeze(2) * neg_emb).sum(-1)  # (B, L, n_neg)

                    pos_loss = -torch.nn.functional.logsigmoid(pos_score)
                    neg_loss = -torch.nn.functional.logsigmoid(-neg_score).mean(-1)
                    loss_per = (pos_loss + neg_loss) * non_pad.float()
                    loss = loss_per.sum() / non_pad.float().sum().clamp(min=1.0)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(sasrec.parameters(), self.max_grad_norm)
                opt.step()

                running_loss += float(loss.item())
                n_steps += 1

            avg_train = running_loss / max(n_steps, 1)

            # ── val ──
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_amp):
                        hidden = model(x)
                        non_pad = y != 0
                        if not non_pad.any():
                            continue
                        pos_emb = sasrec.item_emb(y)
                        neg_idx = torch.randint(
                            1, n_items + 1,
                            (x.size(0), x.size(1), self.n_negatives),
                            device=device,
                        )
                        neg_emb = sasrec.item_emb(neg_idx)
                        pos_score = (hidden * pos_emb).sum(-1)
                        neg_score = (hidden.unsqueeze(2) * neg_emb).sum(-1)
                        pos_loss = -torch.nn.functional.logsigmoid(pos_score)
                        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).mean(-1)
                        loss_per = (pos_loss + neg_loss) * non_pad.float()
                        vloss = loss_per.sum() / non_pad.float().sum().clamp(min=1.0)
                    val_loss_sum += float(vloss.item())
                    val_steps += 1

            avg_val = val_loss_sum / max(val_steps, 1)
            log.info(
                "ESASRec epoch %d/%d: train_loss=%.4f val_loss=%.4f",
                epoch + 1, self.max_epochs, avg_train, avg_val,
            )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in sasrec.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    log.info("ESASRec early stopping at epoch %d", epoch + 1)
                    break

        # Restore best checkpoint
        if best_state is not None:
            sasrec.load_state_dict(best_state)
            sasrec.to(device)
            log.info("ESASRec: restored best checkpoint (val_loss=%.4f)", best_val_loss)

        sasrec.eval()
        self._model = model
        log.info(
            "ESASRec fitted (n_items=%d, n_users=%d)",
            n_items, len(self._user_history),
        )

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

        sasrec = self._unwrap(self._model)
        device = next(sasrec.parameters()).device
        max_len = self.sequence_max_len
        n_items = len(self._item_to_idx)
        bs = self.infer_batch_size  # separate from train batch_size to avoid OOM
        top_n = min(n, n_items)

        self._model.eval()
        chunks: list[pl.DataFrame] = []
        with torch.no_grad():
            for start in range(0, len(u_with_history), bs):
                chunk = u_with_history[start:start + bs]
                X = np.zeros((len(chunk), max_len), dtype=np.int64)
                for i, uid in enumerate(chunk):
                    tokens = self._user_history[uid][-max_len:]
                    X[i, -len(tokens):] = tokens
                x = torch.from_numpy(X).to(device)
                with torch.cuda.amp.autocast(
                    dtype=torch.bfloat16,
                    enabled=self.use_amp and str(device).startswith("cuda"),
                ):
                    hidden = sasrec(x)                       # (B, L, D)
                    user_h = hidden[:, -1, :].float()        # cast to fp32 for topk stability
                    scores = sasrec.all_item_scores(user_h)  # (B, n_items)
                top_scores, top_idx = torch.topk(scores, top_n, dim=1)

                top_idx_np = top_idx.cpu().numpy() + 1      # back to 1-based token idx
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
