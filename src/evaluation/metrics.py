import polars as pl


def recall_at_k(
    df_true: pl.DataFrame,
    df_pred: pl.DataFrame,
    k: int = 100,
) -> float:
    """
    Contest metric: Recall@k * 1000, averaged over users.
    Denominator: min(k, |G_u|) per official formula.

    Args:
        df_true: ground truth — uid (Int64), item_id (Int64)
        df_pred: predictions  — uid (Int64), item_id (Int64), score (Float64, optional)
    """
    assert {"uid", "item_id"}.issubset(df_true.columns), "df_true: нужны uid, item_id"
    assert {"uid", "item_id"}.issubset(df_pred.columns), "df_pred: нужны uid, item_id"
    assert df_pred["uid"].dtype == pl.Int64, "uid должен быть Int64"
    assert df_pred["item_id"].dtype == pl.Int64, "item_id должен быть Int64"
    assert (
        df_pred.group_by(["uid", "item_id"]).len()["len"].max() == 1
    ), "дублирующиеся (uid, item_id) пары в df_pred"

    if "score" in df_pred.columns:
        top_k = df_pred.sort("score", descending=True).group_by("uid").head(k)
    else:
        top_k = df_pred.group_by("uid").head(k)

    top_k = top_k[["uid", "item_id"]].with_columns(pl.lit(1).cast(pl.Int8).alias("hit"))

    denom = (
        df_true
        .group_by("uid")
        .len()
        .with_columns(pl.min_horizontal(pl.col("len"), pl.lit(k)).alias("denom"))
        .select(["uid", "denom"])
    )

    result = (
        df_true[["uid", "item_id"]]
        .join(top_k, on=["uid", "item_id"], how="left")
        .with_columns(pl.col("hit").fill_null(0))
        .group_by("uid")
        .agg(pl.col("hit").sum().alias("n_hits"))
        .join(denom, on="uid")
        .with_columns((pl.col("n_hits") / pl.col("denom")).alias("recall"))
    )

    return result["recall"].mean() * 1000
