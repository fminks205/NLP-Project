"""Command line interface for the `inpnet` pipeline (specs 0001-0002, 0004-0005; spec
0003 superseded but its commands kept for reruns/ablation).

    inpnet seed     --contact you@example.org
    inpnet estimate --contact you@example.org
    inpnet fetch    --contact you@example.org --limit 200
    inpnet clean
    inpnet resolve  --contact you@example.org
    inpnet segment
    inpnet mentions
    inpnet cluster                       # needs `uv sync --extra relations` (spec 0003, superseded)
    inpnet cluster-sweep                 # fast hyperparameter diagnostics on a sample
    inpnet cluster-summary               # cluster_summary.jsonl as readable text
    inpnet detect-attributes             # spec 0004: time/place/institution/action spans
    inpnet assemble-relations            # spec 0005: schemaless relation records
    inpnet relations-summary             # relations.jsonl (or low-coverage) as readable text
    inpnet render-graph --institution-contains yale --relations data/interim/relations/{v}

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
from .manifest import read_jsonl
from .nlp import mentions as mentions_stage
from .nlp import resolve as resolve_stage
from .nlp import segment as segment_stage
from .relations import assemble as assemble_stage
from .relations import attributes as attributes_stage
from .relations import cluster as cluster_stage
from .relations import diagnostics as diagnostics_stage
from .versions import layer_dir
from .viz import graph as graph_stage

DEFAULT_QUERY = Path("docs/queries/physicists.rq")
DEFAULT_SEED = Path("data/raw/seed.jsonl")
DEFAULT_RAW = Path("data/raw")
DEFAULT_INTERIM = Path("data/interim")

# Each layer reads/writes its own version directory — see data_versions.json and
# src/inpnet/versions.py. Bump the version by hand there to start a fresh snapshot.
CORPUS_DIR = layer_dir("corpus_acquisition")
ENTITY_MENTION_DIR = layer_dir("entity_mention_layer")
RELATION_DIR = layer_dir("relation_typology")  # spec 0003, superseded -- kept for reruns
ATTRIBUTE_DIR = layer_dir("attribute_spans")  # spec 0004
RELATIONS_DIR = layer_dir("relations")  # spec 0005

DEFAULT_DOCS = CORPUS_DIR / "documents.jsonl"
DEFAULT_SENTENCES = ENTITY_MENTION_DIR / "sentences.jsonl"
DEFAULT_CACHE = ENTITY_MENTION_DIR / "wikidata_cache.jsonl"
DEFAULT_CANDIDATES = ENTITY_MENTION_DIR / "candidates.jsonl"
DEFAULT_MENTIONS = ENTITY_MENTION_DIR / "mentions.jsonl"
DEFAULT_ENTITIES = ENTITY_MENTION_DIR / "entities.jsonl"
DEFAULT_ATTRIBUTE_SPANS = ATTRIBUTE_DIR / "attribute_spans.jsonl"


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
    p_res.add_argument("--out", type=Path, default=ENTITY_MENTION_DIR)
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
    p_men.add_argument("--out", type=Path, default=ENTITY_MENTION_DIR)
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
    p_clu.add_argument("--mentions", type=Path, default=DEFAULT_MENTIONS)
    p_clu.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    p_clu.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    p_clu.add_argument(
        "--no-mask-entities",
        action="store_true",
        help="skip entity masking before embedding -- ablation switch, spec 0003 "
        "Decision 2a (masking is on by default)",
    )
    p_clu.add_argument("--out", type=Path, default=RELATION_DIR)
    p_clu.add_argument("--model", default=cluster_stage.DEFAULT_MODEL)
    p_clu.add_argument(
        "--min-cluster-size", type=int, default=cluster_stage.DEFAULT_MIN_CLUSTER_SIZE
    )
    p_clu.add_argument("--n-neighbors", type=int, default=cluster_stage.DEFAULT_N_NEIGHBORS)
    p_clu.add_argument("--n-components", type=int, default=cluster_stage.DEFAULT_N_COMPONENTS)
    p_clu.add_argument(
        "--random-state", type=int, default=cluster_stage.DEFAULT_RANDOM_STATE
    )
    p_clu.add_argument(
        "--metric",
        default=cluster_stage.DEFAULT_METRIC,
        help="UMAP distance metric (cosine matches how sentence embeddings are trained)",
    )
    p_clu.add_argument(
        "--cluster-selection-method",
        choices=["eom", "leaf"],
        default=cluster_stage.DEFAULT_CLUSTER_SELECTION_METHOD,
        help="HDBSCAN cluster selection: eom favors the most persistent clusters "
        "(can pick one giant cluster), leaf favors more/smaller ones",
    )
    p_clu.add_argument("--limit", type=int, help="only the first N unique sentences")

    p_sweep = sub.add_parser(
        "cluster-sweep",
        parents=[common],
        help="fast UMAP/HDBSCAN hyperparameter diagnostics on a sample -- a dev tool, "
        "not a pipeline stage; see docs/specs/0003-relation-typology.md",
    )
    p_sweep.add_argument("--in", dest="candidates", type=Path, default=DEFAULT_CANDIDATES)
    p_sweep.add_argument("--mentions", type=Path, default=DEFAULT_MENTIONS)
    p_sweep.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    p_sweep.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    p_sweep.add_argument(
        "--no-mask-entities",
        action="store_true",
        help="skip entity masking before embedding -- ablation switch, spec 0003 "
        "Decision 2a (masking is on by default)",
    )
    p_sweep.add_argument("--model", default=cluster_stage.DEFAULT_MODEL)
    p_sweep.add_argument("--limit", type=int, default=5000, help="sentences to sample")
    p_sweep.add_argument(
        "--random-state", type=int, default=cluster_stage.DEFAULT_RANDOM_STATE
    )
    p_sweep.add_argument("--out", type=Path, help="optional: write full results as JSON")
    p_sweep.add_argument(
        "--show-exemplars",
        type=int,
        default=0,
        metavar="N",
        help="expand the N most-balanced combos with real exemplar sentences to read",
    )

    p_csum = sub.add_parser(
        "cluster-summary",
        parents=[common],
        help="render cluster_summary.jsonl as human-readable text",
    )
    p_csum.add_argument(
        "--in", dest="summary", type=Path, default=RELATION_DIR / "cluster_summary.jsonl"
    )
    p_csum.add_argument("--out", type=Path, help="write to a file instead of stdout")
    p_csum.add_argument(
        "--samples", action="store_true", help="also show each cluster's random sample"
    )

    # --- spec 0004: relation attribute detection ----------------------
    p_attr = sub.add_parser(
        "detect-attributes",
        parents=[common],
        help="candidate sentences -> time/place/institution/action spans",
    )
    p_attr.add_argument("--in", dest="candidates", type=Path, default=DEFAULT_CANDIDATES)
    p_attr.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    p_attr.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    p_attr.add_argument("--out", type=Path, default=ATTRIBUTE_DIR)
    p_attr.add_argument("--model", default=attributes_stage.DEFAULT_MODEL)
    p_attr.add_argument("--limit", type=int, help="only the first N candidate pairs")

    # --- spec 0005: schemaless relation attributes ---------------------
    p_asm = sub.add_parser(
        "assemble-relations",
        parents=[common],
        help="participants + attribute spans -> schemaless relation records",
    )
    p_asm.add_argument("--in", dest="candidates", type=Path, default=DEFAULT_CANDIDATES)
    p_asm.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    p_asm.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    p_asm.add_argument("--attributes", type=Path, default=DEFAULT_ATTRIBUTE_SPANS)
    p_asm.add_argument("--out", type=Path, default=RELATIONS_DIR)
    p_asm.add_argument(
        "--min-coverage", type=float, default=assemble_stage.DEFAULT_MIN_COVERAGE
    )
    p_asm.add_argument("--model", default=attributes_stage.DEFAULT_MODEL)
    p_asm.add_argument("--limit", type=int, help="only the first N candidate pairs")

    p_rsum = sub.add_parser(
        "relations-summary",
        parents=[common],
        help="render relations.jsonl (or low_coverage_relations.jsonl) as readable text",
    )
    p_rsum.add_argument(
        "--in", dest="relations", type=Path, default=RELATIONS_DIR / "relations.jsonl"
    )
    p_rsum.add_argument("--out", type=Path, help="write to a file instead of stdout")

    # --- spec 0006: graph visualization -------------------------------
    p_graph = sub.add_parser(
        "render-graph",
        parents=[common],
        help="institution-filtered relation subgraph -> Graphviz SVG",
    )
    p_graph.add_argument(
        "--institution-contains",
        required=True,
        help="case-insensitive substring matched against relations' institution attribute",
    )
    p_graph.add_argument(
        "--relations",
        type=Path,
        default=RELATIONS_DIR,
        help="directory containing relations.jsonl and low_coverage_relations.jsonl (spec 0005)",
    )
    p_graph.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    p_graph.add_argument("--out", type=Path, default=Path("data/processed/graphs"))
    p_graph.add_argument("--name", help="output subdirectory name; defaults to a filter slug")
    p_graph.add_argument(
        "--high-coverage-only",
        action="store_true",
        help="search relations.jsonl only, skip low_coverage_relations.jsonl",
    )
    p_graph.add_argument(
        "--engine", default=graph_stage.DEFAULT_ENGINE, choices=["neato", "fdp", "sfdp"]
    )
    p_graph.add_argument(
        "--graphviz-bin",
        default=os.environ.get("INPNET_GRAPHVIZ_BIN"),
        help="directory containing the Graphviz executables, if not on PATH "
        "(or set INPNET_GRAPHVIZ_BIN)",
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
                mentions_path=None if args.no_mask_entities else args.mentions,
                entities_path=None if args.no_mask_entities else args.entities,
                sentences_path=None if args.no_mask_entities else args.sentences,
                model_name=args.model,
                min_cluster_size=args.min_cluster_size,
                n_neighbors=args.n_neighbors,
                n_components=args.n_components,
                random_state=args.random_state,
                metric=args.metric,
                cluster_selection_method=args.cluster_selection_method,
                limit=args.limit,
            )
        except ImportError as exc:
            sys.exit(f"error: {exc}")
        print(
            f"\n{counts['n_sentences']:,} unique sentences -> {counts['n_clusters']:,} "
            f"clusters, {counts['n_noise']:,} noise ({counts['noise_fraction']:.1%})\n"
            f"  clustering_run_id: {counts['clustering_run_id']}\n"
        )

    elif args.command == "cluster-sweep":
        try:
            results = diagnostics_stage.sweep(
                args.candidates,
                limit=args.limit,
                model_name=args.model,
                random_state=args.random_state,
                mentions_path=None if args.no_mask_entities else args.mentions,
                entities_path=None if args.no_mask_entities else args.entities,
                sentences_path=None if args.no_mask_entities else args.sentences,
                expand_top_n=args.show_exemplars,
            )
        except ImportError as exc:
            sys.exit(f"error: {exc}")
        print(diagnostics_stage.format_table(results))
        if args.show_exemplars:
            print()
            print(diagnostics_stage.format_exemplars(results))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nwrote {args.out}")
        return 0

    elif args.command == "cluster-summary":
        rows = read_jsonl(args.summary)
        text = cluster_stage.format_cluster_summary(rows, show_samples=args.samples)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            print(f"wrote {args.out} ({len(rows):,} clusters)")
        else:
            print(text)
        return 0

    elif args.command == "detect-attributes":
        counts = attributes_stage.run(
            args.candidates,
            args.entities,
            args.sentences,
            args.out,
            model_name=args.model,
            limit=args.limit,
        )
        print(
            f"\n{counts['n_sentences']:,} unique sentences, {counts['n_candidates']:,} "
            f"candidate pairs -> {counts['n_spans']:,} attribute spans "
            f"{counts['spans_by_type']}\n"
            f"  action found for {counts['n_action_found']:,} of {counts['n_candidates']:,} "
            f"pairs ({counts['action_found_fraction']:.1%})\n"
        )

    elif args.command == "assemble-relations":
        counts = assemble_stage.run(
            args.candidates,
            args.entities,
            args.sentences,
            args.attributes,
            args.out,
            min_coverage=args.min_coverage,
            model_name=args.model,
            limit=args.limit,
        )
        print(
            f"\n{counts['n_candidates']:,} candidate pairs -> {counts['n_relations']:,} "
            f"relations, {counts['n_low_coverage']:,} low-coverage "
            f"({counts['low_coverage_fraction']:.1%}, threshold {counts['min_coverage']})\n"
            f"  mean coverage: {counts['mean_coverage']:.3f}\n"
            f"  wrote {args.out / 'relations_summary.txt'} and "
            f"{args.out / 'low_coverage_relations_summary.txt'}\n"
        )

    elif args.command == "relations-summary":
        rows = read_jsonl(args.relations)
        text = assemble_stage.format_relations_summary(rows)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            print(f"wrote {args.out} ({len(rows):,} relations)")
        else:
            print(text)
        return 0

    elif args.command == "render-graph":
        try:
            counts = graph_stage.run(
                args.relations,
                args.entities,
                args.out,
                institution_contains=args.institution_contains,
                name=args.name,
                high_coverage_only=args.high_coverage_only,
                engine=args.engine,
                graphviz_bin=args.graphviz_bin,
            )
        except FileNotFoundError as exc:
            sys.exit(f"error: {exc}")
        print(
            f"\n{counts['matched_relations']:,} matched relations -> "
            f"{counts['nodes']:,} nodes, {counts['edges']:,} edges\n"
            f"  ({counts['skipped_null_participant']:,} skipped: null participant qid)\n"
        )

    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
