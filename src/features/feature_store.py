"""S3 helpers for reading and writing feature parquet files."""

import io
import logging
from datetime import datetime

import boto3
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _s3_client():
    return boto3.client("s3")


def save_features(
    df: pd.DataFrame,
    ticker: str,
    bucket: str,
    prefix: str = "processed",
    date_tag: str | None = None,
) -> str:
    """Write feature DataFrame to S3 as parquet.

    Returns:
        The S3 key the file was written to.
    """
    tag = date_tag or datetime.utcnow().strftime("%Y%m%d")
    key = f"{prefix}/{ticker}/features_{tag}.parquet"

    buf = io.BytesIO()
    df.to_parquet(buf, index=True)
    buf.seek(0)

    _s3_client().put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info("Saved features to s3://%s/%s", bucket, key)
    return key


def load_features(
    ticker: str,
    bucket: str,
    prefix: str = "processed",
    date_tag: str | None = None,
) -> pd.DataFrame:
    """Load feature DataFrame from S3 parquet."""
    tag = date_tag or "latest"
    if tag == "latest":
        key = _get_latest_key(bucket, f"{prefix}/{ticker}/features_")
    else:
        key = f"{prefix}/{ticker}/features_{tag}.parquet"

    logger.info("Loading features from s3://%s/%s", bucket, key)
    obj = _s3_client().get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _get_latest_key(bucket: str, prefix: str) -> str:
    s3 = _s3_client()
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    objects = sorted(
        resp.get("Contents", []), key=lambda o: o["LastModified"], reverse=True
    )
    if not objects:
        raise FileNotFoundError(f"No objects found under s3://{bucket}/{prefix}")
    return objects[0]["Key"]
