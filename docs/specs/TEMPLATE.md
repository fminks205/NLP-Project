# NNNN — <Title>

| | |
|---|---|
| **Status** | Draft \| Accepted \| Implemented \| Superseded |
| **Depends on** | e.g. 0002, 0003 — or `none` |
| **Superseded by** | — |
| **Owner** | <name> |
| **Last updated** | YYYY-MM-DD |

## Context

Why this decision needs making now. What exists already, what hurts, what the
constraints are (time, compute, licensing, the paper's deadline). Two or three
paragraphs at most.

## Goal

One or two sentences. What is true once this is done that isn't true today.

## Non-goals

Explicitly out of scope. This section prevents the most expensive kind of drift — it is
usually the most valuable part of the spec, so do not leave it empty.

## Decision

What we are doing, concretely enough to implement. Name the libraries, the models, the
thresholds, the algorithm. Where a number is arbitrary, say so and say how it will be
tuned.

## Interface

The data contract this stage produces or consumes. For a pipeline stage:

**Input:** `data/<...>` — schema, or a reference to the spec that defines it.

**Output:** `data/<...>` — one JSON object per line:

```jsonc
{
  "field": "type — meaning"
}
```

**CLI:** `python -m inpnet.<module> --in <...> --out <...>`

Omit this section only for specs that decide something non-computational.

## Alternatives considered

| Option | Why not (for now) |
|--------|-------------------|
| | |

## Risks

What could make this the wrong call, and what the early warning sign would be.

## Acceptance criteria

- [ ] Checkable, ideally by running a command.
- [ ] Includes a test requirement.
- [ ] Includes whatever number the paper will quote.

## Open questions

- Unresolved. Owner and by-when if known.

## References

- Papers, dataset cards, library docs.

## Changelog

- YYYY-MM-DD — created.
