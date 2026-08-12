# NLP-Project — Interpersonal networks from open text

A data science / NLP research project: extract **interpersonal networks** from
open-source text (primary candidate corpus: Wikipedia) and represent them as a graph —
nodes are people, edges are typed social relations.

> *A disagreed with B · C worked with D*

FH-SWF, Angewandte Künstliche Intelligenz (AKI). The deliverable is a paper plus a
reproducible pipeline.

## Status

**Scaffolding.** No pipeline code yet, and the technical approach is not yet decided.

## Where to start

| You are | Read |
|---------|------|
| An AI agent | **[AGENTS.md](AGENTS.md)** — first, always |
| A human contributor | [AGENTS.md](AGENTS.md), then [docs/specs/](docs/specs/) |

## How this repo works

It is **spec-driven**: decisions get written down in [`docs/specs/`](docs/specs/) before
code is written against them. For a research project this means the paper's Methods
section is largely written in advance, and every reported number traces back to a
recorded decision. See [docs/specs/README.md](docs/specs/README.md).

## Ethics

This project makes claims about real, named people. Extracted relations are model output,
not established fact. Source text is CC BY-SA 4.0. See [AGENTS.md](AGENTS.md) §6.
