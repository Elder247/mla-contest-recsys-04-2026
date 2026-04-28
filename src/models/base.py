from abc import ABC, abstractmethod

import polars as pl


class BaseModel(ABC):
    """Abstract base class for all candidate generators."""

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
            DataFrame with columns: uid (Int64), item_id (Int64), score (Float64).
            Sorted by (uid, score desc). At most n rows per uid.
        """
