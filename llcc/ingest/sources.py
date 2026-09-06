"""Where NEXRAD Level II volumes actually come from.

AWS is the primary source, but the bucket was **renamed** in July 2025 from
`noaa-nexrad-level2` to `unidata-nexrad-level2` at the AWS Open Datasets
team's request. The old name now returns AccessDenied for both ListBucket
and GetObject, which reads like a permissions failure but is really a moved
resource. Filename pattern and format are unchanged.

Two mirrors are kept as fallbacks:

  GCP `gcp-public-data-nexrad-l2` stores **hourly tar archives**, not
  individual volumes, so a window has to be resolved to tars, downloaded and
  unpacked. Durable: the full archive is there.

  Unidata THREDDS serves individual volume files, which is simpler and much
  lighter when the date is recent. It is a rolling archive, so old dates age
  out -- fine for a scan you are looking at today, not for building a
  validation corpus.

GLM is unaffected: `noaa-goes19` still lists anonymously.

Listing and URL construction are pure enough to test. The downloads are not,
and are marked UNVERIFIED.
"""

from __future__ import annotations

import json
import re
import tarfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

AWS_BUCKET = "unidata-nexrad-level2"
GCP_BUCKET = "gcp-public-data-nexrad-l2"
THREDDS = "https://thredds.ucar.edu/thredds"
HEADERS = {"User-Agent": "llcc-evaluator/1.0 (research)"}

# Two different stamp shapes, and conflating them silently mis-parses:
#   volume files  KMLB20260826_000345_V06        8 digits, underscore, 6
#   hourly tars   ..._20260826000000_20260826005959.tar   two 14-digit stamps
STAMP = re.compile(r"(\d{8})_(\d{6})")
STAMP14 = re.compile(r"(\d{14})")


@dataclass
class Volume:
    time: datetime
    path: str
    source: str
    url: str | None = None
    url: str | None = None


def parse_stamp(name: str) -> datetime | None:
    m = STAMP.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _get(url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read().decode("utf-8", "replace")


def _download(url: str, dest: str, timeout: float = 600.0) -> str:
    dest_p = Path(dest)
    if dest_p.exists() and dest_p.stat().st_size > 0:
        return dest
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as fh, \
            open(dest, "wb") as out:
        while chunk := fh.read(1 << 20):
            out.write(chunk)
    return dest


# --------------------------------------------------------------------------
# GCP mirror: hourly tars
# --------------------------------------------------------------------------

def gcp_list_day(site: str, day: datetime) -> list[str]:
    """UNVERIFIED: needs network. Object names for one site-day."""
    prefix = f"{day:%Y/%m/%d}/{site.upper()}/"
    names, token = [], None
    while True:
        url = (f"https://storage.googleapis.com/storage/v1/b/{GCP_BUCKET}/o"
               f"?prefix={urllib.parse.quote(prefix)}&maxResults=1000")
        if token:
            url += f"&pageToken={token}"
        doc = json.loads(_get(url))
        names.extend(item["name"] for item in doc.get("items", []))
        token = doc.get("nextPageToken")
        if not token:
            break
    return sorted(names)


def gcp_tars_for_window(site: str, start: datetime, end: datetime) -> list[str]:
    """Tar objects whose hour overlaps the requested window.

    Names look like
    NWS_NEXRAD_NXL2DPBL_KMLB_20260826000000_20260826005959.tar
    with a start and an end stamp, so overlap is decidable from the name.
    """
    out = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        for name in gcp_list_day(site, day):
            if not name.endswith(".tar"):
                continue
            stamps = STAMP14.findall(Path(name).name)
            if len(stamps) < 2:
                continue
            t0 = datetime.strptime(stamps[0], "%Y%m%d%H%M%S")
            t1 = datetime.strptime(stamps[1], "%Y%m%d%H%M%S")
            if t1 >= start and t0 <= end:
                out.append(name)
        day += timedelta(days=1)
    return sorted(set(out))


def gcp_extract(tar_path: str, site: str, start: datetime, end: datetime,
                cache_dir: str) -> list[Volume]:
    """Unpack the volumes inside an hourly tar that fall in the window."""
    out: list[Volume] = []
    dest = Path(cache_dir) / "volumes"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            when = parse_stamp(name)
            if when is None or not (start <= when <= end):
                continue
            if site.upper() not in name.upper():
                continue
            target = dest / name
            if not target.exists():
                src = tf.extractfile(member)
                if src is None:
                    continue
                target.write_bytes(src.read())
            out.append(Volume(when, str(target), "gcp"))
    return sorted(out, key=lambda v: v.time)


def gcp_volumes(site: str, start: datetime, end: datetime,
                cache_dir: str = "/tmp/nexrad") -> list[Volume]:
    """UNVERIFIED: needs network. Resolve a window to unpacked volume files."""
    volumes: list[Volume] = []
    for name in gcp_tars_for_window(site, start, end):
        url = f"https://storage.googleapis.com/{GCP_BUCKET}/{urllib.parse.quote(name)}"
        local = f"{cache_dir}/{Path(name).name}"
        print(f"    fetching {Path(name).name}")
        _download(url, local)
        volumes.extend(gcp_extract(local, site, start, end, cache_dir))
    return sorted(volumes, key=lambda v: v.time)


def tar_member_names(tar_path: str, limit: int = 20) -> list[str]:
    """Diagnostic: what is actually inside one of these archives."""
    with tarfile.open(tar_path) as tf:
        return [m.name for m in tf.getmembers()[:limit]]


# --------------------------------------------------------------------------
# Unidata THREDDS: individual volumes, rolling archive
# --------------------------------------------------------------------------

def thredds_paths(site: str, day: datetime) -> list[str]:
    """UNVERIFIED: needs network. urlPath entries from the day's catalog."""
    url = f"{THREDDS}/catalog/nexrad/level2/{site.upper()}/{day:%Y%m%d}/catalog.xml"
    body = _get(url)
    return re.findall(r'urlPath="([^"]+)"', body)


def thredds_volumes(site: str, start: datetime, end: datetime,
                    cache_dir: str = "/tmp/nexrad") -> list[Volume]:
    """UNVERIFIED: needs network."""
    out: list[Volume] = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        for path in thredds_paths(site, day):
            when = parse_stamp(path)
            if when is None or not (start <= when <= end):
                continue
            url = f"{THREDDS}/fileServer/{path}"
            local = f"{cache_dir}/{Path(path).name}"
            _download(url, local)
            out.append(Volume(when, local, "thredds"))
        day += timedelta(days=1)
    return sorted(out, key=lambda v: v.time)


def aws_list_day(site: str, day: datetime) -> list[str]:
    """UNVERIFIED: needs network. Keys for one site-day from the AWS bucket."""
    prefix = f"{day:%Y/%m/%d}/{site.upper()}/"
    keys, token = [], None
    while True:
        url = (f"https://{AWS_BUCKET}.s3.amazonaws.com/?list-type=2"
               f"&prefix={urllib.parse.quote(prefix)}&max-keys=1000")
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        body = _get(url)
        keys.extend(re.findall(r"<Key>([^<]+)</Key>", body))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        token = m.group(1) if m and "<IsTruncated>true</IsTruncated>" in body else None
        if not token:
            break
    return sorted(keys)


def aws_index(site: str, start: datetime, end: datetime,
              cache_dir: str = "/tmp/nexrad") -> list[Volume]:
    """Volumes in the window, WITHOUT downloading. Listing is cheap; the
    files are not, so the two stay separate."""
    out: list[Volume] = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        for key in aws_list_day(site, day):
            if key.endswith("_MDM"):
                continue
            when = parse_stamp(Path(key).name)
            if when is None or not (start <= when <= end):
                continue
            url = f"https://{AWS_BUCKET}.s3.amazonaws.com/{urllib.parse.quote(key)}"
            out.append(Volume(when, f"{cache_dir}/{Path(key).name}", "aws", url))
        day += timedelta(days=1)
    return sorted(out, key=lambda v: v.time)


def fetch(volume: Volume) -> Volume:
    """Download one volume if it is not already cached."""
    if volume.url:
        _download(volume.url, volume.path)
    return volume


def aws_volumes(site: str, start: datetime, end: datetime,
                cache_dir: str = "/tmp/nexrad") -> list[Volume]:
    """UNVERIFIED: needs network."""
    index = aws_index(site, start, end, cache_dir)
    for n, vol in enumerate(index, start=1):
        print(f"    [{n}/{len(index)}] {Path(vol.path).name}", flush=True)
        fetch(vol)
    return index


def resolve_volumes(site: str, start: datetime, end: datetime,
                    source: str = "aws",
                    cache_dir: str = "/tmp/nexrad") -> list[Volume]:
    if source == "aws":
        return aws_volumes(site, start, end, cache_dir)
    if source == "gcp":
        return gcp_volumes(site, start, end, cache_dir)
    if source == "thredds":
        return thredds_volumes(site, start, end, cache_dir)
    raise ValueError(f"unknown source {source!r}; use 'aws', 'gcp' or 'thredds'")
