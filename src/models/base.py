from abc import ABC, abstractmethod

import polars as pl


class BaseModel(ABC):
    """Abstract base class for all candidate generators.

    Convention: every concrete model has a `name` attribute (e.g. ``"als"``).
    The merge_candidates utility uses this name to derive feature columns
    ``{name}_score`` and ``{name}_rank`` in the merged candidates table.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, train: pl.DataFrame, **kwargs) -> None:
        """Train the model on interaction data.

        Args:
            train: DataFrame with at minimum uid, item_id, timestamp columns.
        """

    @abstractmethod
    def recommend(
        self,
        users: list[int],
        n: int = 100,
        **kwargs,
    ) -> pl.DataFrame:
        """Generate top-n candidate items for each user.

        Args:
            users: List of user IDs to generate recommendations for.
            n: Maximum number of candidates per user.

        Returns:
            DataFrame with columns
                uid: Int64
                item_id: Int64
                score: Float32 (or Float64)
                {name}_rank: Int32 — 1-based rank within this CG for the user.
            Sorted by (uid, score desc). At most n rows per uid.
            The {name}_rank column is mandatory: merge_candidates relies on
            its presence to track per-CG ranks across the union of candidates.
        """
