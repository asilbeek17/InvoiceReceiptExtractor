"""
Measure extraction accuracy against samples/ground_truth.json.

  python evaluate.py                 # free local engine
  python evaluate.py --engine ai     # Claude vision (needs ANTHROPIC_API_KEY, costs a little)
  python evaluate.py --limit 10      # only the first 10 documents

NOTE: these are synthetic documents. Real-world accuracy is usually lower, so also test
on real (anonymized) documents before quoting numbers to clients.
"""
import argparse
import json
import os
import re
from collections import defaultdict

from extractor import extract_document

FIELDS = ["vendor_name", "invoice_number", "date", "currency", "subtotal", "tax", "total", "n_items"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def check(field, got, truth):
    if field == "vendor_name":
        return norm(got) == norm(truth)
    if field in ("subtotal", "tax", "total"):
        if truth is None:
            return got in (None, 0, 0.0)
        return got is not None and abs(got - truth) < 0.011
    if field == "n_items":
        return got == truth
    return (got or None) == truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="local", choices=["local", "ai"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dir", default="samples")
    args = ap.parse_args()

    truth = json.load(open(os.path.join(args.dir, "ground_truth.json")))
    names = sorted(truth)[: args.limit]
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # group -> field -> [correct, total]
    all_correct, failures = defaultdict(lambda: [0, 0]), []
    wrong_docs, wrong_flagged = 0, 0

    for n, name in enumerate(names, 1):
        t = truth[name]
        flagged = False
        try:
            res = extract_document(open(os.path.join(args.dir, name), "rb").read(), name, engine=args.engine)
            data = res["data"]
            got = {**data, "n_items": len(data.get("line_items") or [])}
            flagged = bool(res["warnings"])
        except Exception as e:
            got = {}
            failures.append(f"{name}: crashed ({e})")
        ok_fields = []
        for f in FIELDS:
            ok = check(f, got.get(f), t[f])
            for grp in (t["difficulty"], "ALL"):
                stats[grp][f][0] += ok
                stats[grp][f][1] += 1
            if not ok:
                ok_fields.append(f"{f}: got {got.get(f)!r}, expected {t[f]!r}")
        key_ok = all(check(f, got.get(f), t[f]) for f in ("vendor_name", "date", "subtotal", "tax", "total"))
        for grp in (t["difficulty"], "ALL"):
            all_correct[grp][0] += key_ok
            all_correct[grp][1] += 1
        if not key_ok:
            wrong_docs += 1
            wrong_flagged += flagged
        if ok_fields:
            failures.append(f"{name} [{t['difficulty']}]: " + " | ".join(ok_fields))
        print(f"\r{n}/{len(names)} processed", end="")
    print("\n")

    print(f"Engine: {args.engine}   Documents: {len(names)}\n")
    header = f"{'':8s}" + "".join(f"{f:>16s}" for f in FIELDS) + f"{'all key fields':>16s}"
    print(header)
    for grp in ("ALL", "clean", "hard"):
        if grp not in stats:
            continue
        row = "".join(f"{stats[grp][f][0] / stats[grp][f][1]:>16.0%}" for f in FIELDS)
        c, tot = all_correct[grp]
        print(f"{grp:8s}{row}{c / tot:>16.0%}   ({c}/{tot} fully correct)")
    if wrong_docs:
        print(f"\nDocuments with a wrong key field: {wrong_docs}. "
              f"Flagged 'needs review' by the validator: {wrong_flagged} of {wrong_docs}.")
    print("\nFailures:")
    print("\n".join(failures) or "none")


if __name__ == "__main__":
    main()
