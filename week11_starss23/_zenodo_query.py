"""Tiny helper to inspect Zenodo records for download URLs."""
import json
import sys
import urllib.request


def inspect(record_id: str) -> None:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url) as resp:
        d = json.load(resp)
    print(f"Title: {d.get('metadata', {}).get('title', '?')}")
    print(f"Version: {d.get('metadata', {}).get('version', '?')}")
    print(f"DOI: {d.get('doi', '?')}")
    print()
    print("Files:")
    for f in d.get("files", []):
        size_mb = f["size"] / 1e6
        print(f"  {f['key']}  {size_mb:.1f} MB")


if __name__ == "__main__":
    for rid in sys.argv[1:]:
        print(f"=== record {rid} ===")
        inspect(rid)
        print()
