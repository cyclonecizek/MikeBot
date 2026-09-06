"""Diagnose S3 access to the NOAA open-data buckets.

A 403 on an anonymous listing can mean several different things, and the
response body names which. This tries each plausible access path and reports
what came back, so the fix is chosen from evidence rather than guessed.

    python3 diagnose_s3.py
"""

from __future__ import annotations

import urllib.error
import urllib.request

BUCKET = "noaa-nexrad-level2"
PREFIX = "2026/08/26/KMLB/"

VARIANTS = [
    ("virtual-hosted, ListObjectsV2",
     f"https://{BUCKET}.s3.amazonaws.com/?list-type=2&prefix={PREFIX}&max-keys=5"),
    ("virtual-hosted, legacy list",
     f"https://{BUCKET}.s3.amazonaws.com/?prefix={PREFIX}&max-keys=5"),
    ("regional endpoint, ListObjectsV2",
     f"https://{BUCKET}.s3.us-east-1.amazonaws.com/?list-type=2&prefix={PREFIX}&max-keys=5"),
    ("path-style, ListObjectsV2",
     f"https://s3.amazonaws.com/{BUCKET}/?list-type=2&prefix={PREFIX}&max-keys=5"),
    ("no prefix at all",
     f"https://{BUCKET}.s3.amazonaws.com/?list-type=2&max-keys=5"),
    ("delimiter listing (directory style)",
     f"https://{BUCKET}.s3.amazonaws.com/?list-type=2&delimiter=/&prefix={PREFIX}&max-keys=5"),
]

HEADERS = {"User-Agent": "llcc-evaluator/1.0 (research)"}


def attempt(label: str, url: str) -> None:
    print(f"\n--- {label}")
    print(f"    {url}")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as fh:
            body = fh.read().decode("utf-8", "replace")
        keys = body.count("<Key>")
        print(f"    HTTP {fh.status}   {keys} key(s)")
        if keys:
            start = body.find("<Key>") + 5
            print(f"    first key: {body[start:body.find('</Key>', start)]}")
        else:
            print(f"    body: {body[:300]}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        print(f"    HTTP {exc.code} {exc.reason}")
        for tag in ("Code", "Message"):
            if f"<{tag}>" in body:
                start = body.find(f"<{tag}>") + len(tag) + 2
                print(f"    {tag}: {body[start:body.find(f'</{tag}>', start)]}")
        if "<Code>" not in body:
            print(f"    body: {body[:300]}")
    except Exception as exc:
        print(f"    {type(exc).__name__}: {exc}")


def try_boto3() -> None:
    print("\n--- boto3 with unsigned credentials")
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        print("    boto3 not installed  (pip install boto3)")
        return
    try:
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, MaxKeys=5)
        keys = [o["Key"] for o in resp.get("Contents", [])]
        print(f"    OK   {len(keys)} key(s)")
        if keys:
            print(f"    first key: {keys[0]}")
    except Exception as exc:
        print(f"    {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    print(f"bucket: {BUCKET}\nprefix: {PREFIX}")
    for label, url in VARIANTS:
        attempt(label, url)
    try_boto3()
    print("\nReport which variants returned keys.")
