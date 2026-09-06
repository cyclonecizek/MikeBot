"""Find a working source for NEXRAD Level II and GLM.

Anonymous listing on noaa-nexrad-level2 is returning AccessDenied for every
access path, so this checks independent mirrors. Note that a bucket can deny
ListBucket while still allowing anonymous GetObject -- so if a mirror gives
us the file names, downloading from AWS may still work. This tests that too.

    python3 diagnose_sources.py
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

SITE = "KMLB"
Y, M, D = "2026", "08", "26"
HEADERS = {"User-Agent": "llcc-evaluator/1.0 (research)"}
found: dict[str, str] = {}


def get(url: str, timeout: float = 30.0) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return fh.status, fh.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:400]
    except Exception as exc:
        return -1, f"{type(exc).__name__}: {exc}"


def report(label: str, url: str, status: int, body: str, keys: list[str]) -> None:
    print(f"\n--- {label}")
    print(f"    {url}")
    if keys:
        print(f"    HTTP {status}   {len(keys)} key(s)")
        print(f"    first: {keys[0]}")
        found.setdefault(label, keys[0])
    else:
        print(f"    HTTP {status}")
        print(f"    {body[:250].strip()}")


def xml_keys(body: str) -> list[str]:
    out, i = [], 0
    while True:
        i = body.find("<Key>", i)
        if i < 0:
            return out
        j = body.find("</Key>", i)
        out.append(body[i + 5:j])
        i = j


print(f"looking for {SITE} on {Y}-{M}-{D}")

# 1. Google Cloud public NEXRAD mirror, XML listing.
url = (f"https://storage.googleapis.com/gcp-public-data-nexrad-l2"
       f"?prefix={Y}/{M}/{D}/{SITE}/&max-keys=5")
st, body = get(url)
report("GCP mirror, XML listing", url, st, body, xml_keys(body))

# 2. Google Cloud JSON API.
url = (f"https://storage.googleapis.com/storage/v1/b/gcp-public-data-nexrad-l2/o"
       f"?prefix={Y}/{M}/{D}/{SITE}/&maxResults=5")
st, body = get(url)
keys = []
if st == 200:
    try:
        keys = [it["name"] for it in json.loads(body).get("items", [])]
    except Exception:
        pass
report("GCP mirror, JSON API", url, st, body, keys)

# 3. Unidata THREDDS, rolling archive of recent data.
url = f"https://thredds.ucar.edu/thredds/catalog/nexrad/level2/{SITE}/{Y}{M}{D}/catalog.xml"
st, body = get(url)
names = []
i = 0
while st == 200:
    i = body.find('urlPath="', i)
    if i < 0:
        break
    j = body.find('"', i + 9)
    names.append(body[i + 9:j])
    i = j
report("Unidata THREDDS catalog", url, st, body, names)

# 4. NCEI THREDDS.
url = (f"https://www.ncei.noaa.gov/thredds/catalog/nexrad-level2/"
       f"{Y}/{M}/{D}/{SITE}/catalog.xml")
st, body = get(url)
names = []
i = 0
while st == 200:
    i = body.find('urlPath="', i)
    if i < 0:
        break
    j = body.find('"', i + 9)
    names.append(body[i + 9:j])
    i = j
report("NCEI THREDDS catalog", url, st, body, names)

# 5. Does AWS allow GetObject even though it denies ListBucket?
print("\n--- AWS GetObject with a key from a mirror")
key = next(iter(found.values()), None)
if key is None:
    print("    skipped: no mirror returned a key to test with")
else:
    aws = f"https://noaa-nexrad-level2.s3.amazonaws.com/{urllib.parse.quote(key)}"
    print(f"    {aws}")
    req = urllib.request.Request(aws, headers={**HEADERS, "Range": "bytes=0-63"})
    try:
        with urllib.request.urlopen(req, timeout=30) as fh:
            head = fh.read()
        print(f"    HTTP {fh.status}   {len(head)} bytes")
        print(f"    magic: {head[:8]!r}  (expect b'AR2V00' for Level II)")
    except urllib.error.HTTPError as exc:
        print(f"    HTTP {exc.code} {exc.reason} -- listing denied AND download denied")
    except Exception as exc:
        print(f"    {type(exc).__name__}: {exc}")

# 6. GLM, in case the GOES buckets behave differently.
print("\n--- GLM on noaa-goes19")
doy = 238  # 26 August 2026
url = (f"https://noaa-goes19.s3.amazonaws.com/"
       f"?list-type=2&prefix=GLM-L2-LCFA/{Y}/{doy:03d}/16/&max-keys=3")
st, body = get(url)
report("AWS noaa-goes19 listing", url, st, body, xml_keys(body))

print("\n" + "=" * 60)
if found:
    print("Working sources:")
    for label in found:
        print(f"  {label}")
else:
    print("No source returned keys. Paste the whole output.")
