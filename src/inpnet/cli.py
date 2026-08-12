"""Command line interface for the corpus stages of spec 0001.

    inpnet seed     --contact you@example.org
    inpnet estimate --contact you@example.org
    inpnet fetch    --contact you@example.org --limit 200
    inpnet clean

Each stage is a separate command on purpose: a failed fetch must never force a
re-query, and a cleaning bug must never force a re-fetch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .corpus import clean, estimate, fetch, seed

DEFAULT_QUERY = Path("docs/queries/physicists.rq")
DEFAULT_SEED = Path("data/raw/seed.jsonl")
DEFAULT_RAW = Path("data/raw")
DEFAULT_DOCS = Path("data/interim/documents.jsonl")


def _contact(args: argparse.Namespace) -> str:
    contact = args.contact or os.environ.get("INPNET_CONTACT")
    if not contact:
        sys.exit(
            "error: a contact address is required.\n"
            "Wikimedia's User-Agent policy requires requests to identify an "
            "operator. Pass --contact you@example.org or set INPNET_CONTACT."
        )
    return contact


def main(argv: list[str] | None = None) -> int:
    # Shared options, attached to every subcommand so they may appear on either
    # side of it -- `inpnet --contact X seed` and `inpnet seed --contact X`
    # both work.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--contact", help="email or URL identifying the operator")
    common.add_argument("--delay", type=float, default=1.0, help="seconds between requests")

    parser = argparse.ArgumentParser(prog="inpnet", description=__doc__, parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("seed", parents=[common], help="run the Wikidata query -> seed.jsonl")
    p_seed.add_argument("--query", type=Path, default=DEFAULT_QUERY)
    p_seed.add_argument("--out", type=Path, default=DEFAULT_SEED)

    p_est = sub.add_parser("estimate", parents=[common], help="project download size")
    p_est.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    p_est.add_argument("--sample", type=int, default=30, help="articles sampled for the ratio")
    p_est.add_argument("--limit", type=int, help="only consider the first N of the seed list")

    p_fetch = sub.add_parser("fetch", parents=[common], help="download article HTML")
    p_fetch.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    p_fetch.add_argument("--out", type=Path, default=DEFAULT_RAW)
    p_fetch.add_argument("--limit", type=int, help="fetch only the first N articles")
    p_fetch.add_argument("--no-resume", action="store_true", help="re-fetch existing files")

    p_clean = sub.add_parser("clean", parents=[common], help="HTML -> documents.jsonl")
    p_clean.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p_clean.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    p_clean.add_argument("--out", type=Path, default=DEFAULT_DOCS)
    p_clean.add_argument("--limit", type=int)
    p_clean.add_argument(
        "--lenient", action="store_true", help="report offset errors instead of failing"
    )

    args = parser.parse_args(argv)

    if args.command == "seed":
        counts = seed.run(args.query, args.out, contact=_contact(args), delay=args.delay)
        print(f"wrote {args.out}")

    elif args.command == "estimate":
        counts = estimate.run(
            args.seed,
            contact=_contact(args),
            sample_size=args.sample,
            limit=args.limit,
            delay=args.delay,
        )
        print(
            f"\n{counts['articles']:,} articles\n"
            f"  wikitext total : {estimate.human_bytes(counts['total_wikitext_bytes'])} (measured)\n"
            f"  wikitext mean  : {estimate.human_bytes(counts['mean_wikitext_bytes'])}\n"
            f"  HTML ratio     : {counts['html_ratio_weighted']}x "
            f"(sampled on {counts['sample_size']} articles)\n"
            f"  projected HTML : {counts['projected_html_human']}\n"
        )

    elif args.command == "fetch":
        counts = fetch.run(
            args.seed,
            args.out,
            contact=_contact(args),
            limit=args.limit,
            delay=args.delay,
            resume=not args.no_resume,
        )

    elif args.command == "clean":
        counts = clean.run(
            args.raw,
            args.out,
            seed_path=args.seed,
            limit=args.limit,
            strict=not args.lenient,
        )

    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
