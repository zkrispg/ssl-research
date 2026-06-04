"""Compare STARSS22 vs STARSS23 metadata class IDs + format."""
import csv
import glob
import os


def head(f, n=3):
    rows = []
    with open(f, "r") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if i < n:
                rows.append(row)
            else:
                break
    return rows


def classes(f):
    cs = set()
    with open(f, "r") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                try:
                    cs.add(int(row[1]))
                except (ValueError, IndexError):
                    pass
    return sorted(cs)


files22 = glob.glob(r"D:\ssl-research\data\STARSS22\metadata_dev\dev-test-sony\*.csv")[:1]
files23 = glob.glob(r"D:\ssl-research\DCASE2024_SELD_dataset\metadata_dev\metadata_dev\dev-test-sony\*.csv")[:1]

for f in files22 + files23:
    parent = os.path.basename(os.path.dirname(os.path.dirname(f)))
    print(f"--- {parent}/{os.path.basename(f)} ---")
    print("  first 3 rows:", head(f))
    print("  class IDs:", classes(f))
    print()

all22 = set()
for f in glob.glob(r"D:\ssl-research\data\STARSS22\metadata_dev\dev-test-*\*.csv"):
    all22 |= set(classes(f))
print(f"STARSS22 dev-test all class IDs: {sorted(all22)}  count={len(all22)}")

all23 = set()
for f in glob.glob(r"D:\ssl-research\DCASE2024_SELD_dataset\metadata_dev\metadata_dev\dev-test-*\*.csv"):
    all23 |= set(classes(f))
print(f"STARSS23 dev-test all class IDs: {sorted(all23)}  count={len(all23)}")

# Field count check
def n_fields(f):
    with open(f, "r") as fh:
        for row in csv.reader(fh):
            return len(row)
    return None

print(f"\nSTARSS22 row width: {n_fields(files22[0])}  (DCASE 2022 = 5 cols: frame,cls,src,az,el)")
print(f"STARSS23 row width: {n_fields(files23[0])}  (DCASE 2024 = 6 cols: frame,cls,src,az,el,dist)")
