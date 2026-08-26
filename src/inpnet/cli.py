"""Command line interface for the `inpnet` pipeline (specs 0001-0003).

    inpnet seed     --contact you@example.org
    inpnet estimate --contact you@example.org
    inpnet fetch    --contact you@example.org --limit 200
    inpnet clean
    inpnet resolve  --contact you@example.org
    inpnet segment
    inpnet mentions
    inpnet cluster                       # needs `uv sync --extra relations`

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
from .nlp import mentions as mentions_stage
from .nlp import resolve as resolve_stage
from .nlp import segment as segment_stage
from .relations import cluster as cluster_stage

DEFAULT_QUERY = Path("docs/queries/physicists.rq")
DEFAULT_SEED = Path("data/raw/seed.jsonl")
DEFAULT_RAW = Path("data/raw")
DEFAULT_INTERIM = Path("data/interim")
DEFAULT_DOCS = DEFAULT_INTERIM / "documents.jsonl"
DEFAULT_SENTENCES = DEFAULT_INTERIM / "sentences.jsonl"
DEFAULT_CACHE = DEFAULT_INTERIM / "wikidata_cache.jsonl"
DEFAULT_CANDIDATES = DEFAULT_INTERIM / "candidates.jsonl"


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

    # --- spec 0002: entity & mention layer ---------------------------
    p_res = sub.add_parser("resolve", parents=[common], help="link targets -> Wikidata humans")
    p_res.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    p_res.add_argument("--out", type=Path, default=DEFAULT_INTERIM)
    p_res.add_argument("--limit", type=int, help="only the N most frequent link targets")

    p_seg = sub.add_parser("segment", parents=[common], help="split sections into sentences")
    p_seg.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    p_seg.add_argument("--out", type=Path, default=DEFAULT_SENTENCES)
    p_seg.add_argument("--limit", type=int, help="only the first N documents")
    p_seg.add_argument("--n-process", type=int, default=1)

    p_men = sub.add_parser("mentions", parents=[common], help="build mentions + candidate pairs")
    p_men.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    p_men.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    p_men.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p_men.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    p_men.add_argument("--out", type=Path, default=DEFAULT_INTERIM)
    p_men.add_argument("--limit", type=int)
    p_men.add_argument(
        "--lenient",
        action="store_true",
        help="proceed on an incomplete Wikidata cache (people will be omitted)",
    )

    # --- spec 0003: relation clustering interface --------------------
    p_clu = sub.add_parser(
        "cluster", parents=[common], help="candidate sentences -> field-agnostic clusters"
    )
    p_clu.add_argument("--in", dest="candidates", type=Path, default=DEFAULT_CANDIDATES)
    p_clu.add_argument("--out", type=Path, default=DEFAULT_INTERIM)
    p_clu.add_argument("--model", default=cluster_stage.DEFAULT_MODEL)
    p_clu.add_argument(
        "--min-cluster-size", type=int, default=cluster_stage.DEFAULT_MIN_CLUSTER_SIZE
    )
    p_clu.add_argument("--n-neighbors", type=int, default=cluster_stage.DEFAULT_N_NEIGHBORS)
    p_clu.add_argument("--n-components", type=int, default=cluster_stage.DEFAULT_N_COMPONENTS)
    p_clu.add_argument(
        "--random-state", type=int, default=cluster_stage.DEFAULT_RANDOM_STATE
    )
    p_clu.add_argument("--limit", type=int, help="only the first N unique sentences")

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

    elif args.command == "resolve":
        counts = resolve_stage.run(
            args.docs, args.out, contact=_contact(args), delay=args.delay, limit=args.limit
        )
        rate = counts["human_instance_rate"]
        print(
            f"\n{counts['distinct_targets']:,} distinct link targets, "
            f"{counts['human_targets']:,} are people\n"
            f"  person-link instances: {counts['human_link_instances']:,} "
            f"of {counts['total_link_instances']:,} ({rate:.1%})\n"
        )

    elif args.command == "segment":
        counts = segment_stage.run(
            args.docs, args.out, limit=args.limit, n_process=args.n_process
        )

    elif args.command == "mentions":
        counts = mentions_stage.run(
            args.docs,
            args.sentences,
            args.cache,
            args.seed,
            args.out,
            limit=args.limit,
            strict=not args.lenient,
        )

    elif args.command == "cluster":
        try:
            counts = cluster_stage.run(
                args.candidates,
                args.out,
                model_name=args.model,
                min_cluster_size=args.min_cluster_size,
                n_neighbors=args.n_neighbors,
                n_components=args.n_components,
                random_state=args.random_state,
                limit=args.limit,
            )
        except ImportError as exc:
            sys.exit(f"error: {exc}")
        print(
            f"\n{counts['n_sentences']:,} unique sentences -> {counts['n_clusters']:,} "
            f"clusters, {counts['n_noise']:,} noise ({counts['noise_fraction']:.1%})\n"
            f"  clustering_run_id: {counts['clustering_run_id']}\n"
        )

    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
