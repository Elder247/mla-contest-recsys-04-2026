"""CatBoost-based reranker: takes labeled candidates DataFrame, returns top-n per user."""
import logging

import polars as pl
from catboost import CatBoostRanker, Pool

log = logging.getLogger(__name__)

_SKIP_COLS = {"uid", "item_id", "label"}


class RankerModel:
    """Thin wrapper around CatBoostRanker.

    Expects a DataFrame with columns: uid, item_id, label (0/1), and feature columns.
    """

    def __init__(
        self,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.1,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self._model: CatBoostRanker | None = None
        self._feature_cols: list[str] = []

    def fit(self, df_train: pl.DataFrame, df_val: pl.DataFrame | None = None) -> None:
        self._feature_cols = [c for c in df_train.columns if c not in _SKIP_COLS]
        log.info("fitting RankerModel: %d features, train=%d rows", len(self._feature_cols), len(df_train))

        # CatBoostRanker requires query_ids to be contiguous per group.
        df_train = df_train.sort("uid")
        train_pool = Pool(
            data=df_train[self._feature_cols].to_pandas(),
            label=df_train["label"].to_pandas(),
            group_id=df_train["uid"].to_pandas(),
        )
        params = dict(
            loss_function="YetiRank",
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_state,
            verbose=100,
            nan_mode="Min",
            thread_count=-1,
        )
        if df_val is not None:
            df_val = df_val.sort("uid")
            val_pool = Pool(
                data=df_val[self._feature_cols].to_pandas(),
                label=df_val["label"].to_pandas(),
                group_id=df_val["uid"].to_pandas(),
            )
            params["early_stopping_rounds"] = self.early_stopping_rounds
            self._model = CatBoostRanker(**params)
            self._model.fit(train_pool, eval_set=val_pool)
        else:
            self._model = CatBoostRanker(**params)
            self._model.fit(train_pool)

        log.info("RankerModel fitted, best_iteration=%s", self._model.get_best_iteration())

    def predict(self, df: pl.DataFrame, n: int = 100) -> pl.DataFrame:
        scores = self._model.predict(df[self._feature_cols].to_pandas())
        return (
            df.select(["uid", "item_id"])
            .with_columns(pl.Series("ranker_score", scores))
            .sort(["uid", "ranker_score"], descending=[False, True])
            .group_by("uid")
            .head(n)
            .select(["uid", "item_id", "ranker_score"])
        )

    def feature_importance(self, prettified: bool = True):
        """CatBoost native PredictionValuesChange feature importance.

        For ``YetiRank`` (and other ranking losses) CatBoost defaults to
        ``LossFunctionChange``, which needs a Pool to compute. We force
        ``PredictionValuesChange`` so the call works without refeeding data.

        Returns a pandas DataFrame sorted descending by importance when
        ``prettified=True``; otherwise an array aligned with feature order.
        """
        if self._model is None:
            raise RuntimeError("RankerModel.feature_importance: model is not fitted yet")
        return self._model.get_feature_importance(
            type="PredictionValuesChange",
            prettified=prettified,
        )
