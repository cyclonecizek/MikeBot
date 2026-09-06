"""Anonymous S3 listing for the NOAA open-data buckets.

Both `unidata-nexrad-level2` and `noaa-goes19` allow unauthenticated reads over
plain HTTPS, so no boto3 and no credentials. The XML parsing is a pure
function and is tested; the network call is not.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def parse_listing(xml_text: str) -> tuple[list[str], str | None]:
    """Return (keys, continuation_token) from an S3 ListObjectsV2 response."""
    root = ET.fromstring(xml_text)
    keys = [e.text for e in root.iter(f"{NS}Key") if e.text]
    token = None
    for e in root.iter(f"{NS}NextContinuationToken"):
        token = e.text
    truncated = any(e.text == "true" for e in root.iter(f"{NS}IsTruncated"))
    return keys, (token if truncated else None)


def listing_url(bucket: str, prefix: str, token: str | None = None) -> str:
    params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
    if token:
        params["continuation-token"] = token
    return f"https://{bucket}.s3.amazonaws.com/?{urllib.parse.urlencode(params)}"


def object_url(bucket: str, key: str) -> str:
    return f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"


def list_keys(bucket: str, prefix: str, timeout: float = 30.0) -> list[str]:
    """UNVERIFIED: needs network. Pages through a prefix and returns all keys."""
    keys: list[str] = []
    token = None
    while True:
        with urllib.request.urlopen(listing_url(bucket, prefix, token),
                                    timeout=timeout) as fh:
            batch, token = parse_listing(fh.read().decode("utf-8"))
        keys.extend(batch)
        if not token:
            break
    return sorted(keys)


def download(bucket: str, key: str, dest: str, timeout: float = 120.0) -> str:
    """UNVERIFIED: needs network."""
    urllib.request.urlretrieve(object_url(bucket, key), dest)
    return dest


def nexrad_prefix(site: str, when: datetime) -> str:
    return f"{when:%Y/%m/%d}/{site.upper()}/"


def glm_prefix(when: datetime) -> str:
    """GLM keys are organised by day of year and hour."""
    return f"GLM-L2-LCFA/{when:%Y}/{when.timetuple().tm_yday:03d}/{when:%H}/"
