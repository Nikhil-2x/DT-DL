"""Write a small 3-class, 4-feature CSV for examples/csv_classifier.py.

    python examples/make_blobs_csv.py            # writes blobs.csv in the current folder
"""
import random
import sys

CENTERS = [(0, 0, 0, 0), (3, 3, 0, 1), (-3, 2, 3, -2)]


def main(path="blobs.csv", per_class=300, seed=7):
    rng = random.Random(seed)
    rows = []
    for label, center in enumerate(CENTERS):
        for _ in range(per_class):
            rows.append([round(rng.gauss(c, 1.0), 4) for c in center] + [label])
    rng.shuffle(rows)
    with open(path, "w", newline="") as f:
        f.write("f1,f2,f3,f4,label\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
