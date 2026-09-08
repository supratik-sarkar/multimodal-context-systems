"""Command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import DatasetManifest, diff_manifests
from .parsers import ingest, supported_suffixes
from .pipeline import FoundryConfig, build_dataset
from .stages import scan_pii


def _load(path: str) -> DatasetManifest:
    d = json.loads(Path(path).read_text())
    from .manifest import SplitManifest
    return DatasetManifest(
        dataset_version=d["dataset_version"], created_at=d["created_at"],
        config=d["config"], stage_counts=d["stage_counts"],
        rejections=d.get("rejections", []),
        splits={n: SplitManifest(**s) for n, s in d["splits"].items()},
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gdf", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="build a versioned dataset")
    p.add_argument("source")
    p.add_argument("--version", required=True)
    p.add_argument("--out", default="build")
    p.add_argument("--near-dedup-threshold", type=float, default=0.7)
    p.add_argument("--eval-fraction", type=float, default=0.2)

    p = sub.add_parser("inspect", help="summarise a manifest")
    p.add_argument("manifest")

    p = sub.add_parser("diff", help="compare two manifests")
    p.add_argument("a")
    p.add_argument("b")

    p = sub.add_parser("scan-pii", help="report PII matches without modifying")
    p.add_argument("source")

    sub.add_parser("formats", help="list supported file suffixes")

    args = ap.parse_args(argv)

    match args.cmd:
        case "formats":
            for s in supported_suffixes():
                print(s)
        case "build":
            cfg = FoundryConfig(near_dedup_threshold=args.near_dedup_threshold,
                                eval_fraction=args.eval_fraction)
            result = build_dataset(args.source, args.version, config=cfg)
            out = Path(args.out)
            path = result.manifest.write(out / f"manifest-{args.version}.json")
            for name, recs in result.splits.items():
                f = out / f"{name}-{args.version}.jsonl"
                f.write_text("\n".join(r.to_json() for r in recs))
            print(f"wrote {path}")
            print(json.dumps(result.manifest.stage_counts, indent=2))
        case "inspect":
            m = _load(args.manifest)
            print(json.dumps({
                "version": m.dataset_version,
                "content_digest": m.content_digest,
                "total_records": m.total_records,
                "stage_counts": m.stage_counts,
                "splits": {n: s.record_count for n, s in m.splits.items()},
                "rejections": len(m.rejections),
            }, indent=2))
        case "diff":
            print(json.dumps(diff_manifests(_load(args.a), _load(args.b)),
                             indent=2))
        case "scan-pii":
            for rec in ingest(Path(args.source)):
                found = scan_pii(rec.text)
                if found:
                    print(f"{rec.record_id:<24} {found}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
