# Specs

This directory is the source of truth for **what we are building and why**. Code
implements specs; specs do not document code after the fact.

If you are an AI agent, start at [`../../AGENTS.md`](../../AGENTS.md).

---

## Why spec-driven here

This is a research project with a paper as its deliverable. That makes decisions —
which relation types count, what the gold standard is, which baseline we compare
against — more important and more expensive to reverse than the code that implements
them. Writing them down first means:

- The paper's Methods section is mostly already written.
- A result is reproducible because the decisions behind it are recorded, not remembered.
- Humans and agents can work on different stages without silently disagreeing about the
  interfaces between them.

## Lifecycle

| Status | Meaning |
|--------|---------|
| `Draft` | Being written or discussed. Do not implement against it yet. |
| `Accepted` | Agreed. Implement this. Changing it needs a deliberate edit + note in Changelog. |
| `Implemented` | Code exists, tests pass, acceptance criteria are checked off. |
| `Superseded` | Replaced. Must name its successor in `Superseded by:`. Never delete a spec. |

A spec moves `Draft → Accepted` when a human says so. An agent may **write** and
**revise** drafts freely; an agent may not promote its own draft to `Accepted`.

## File naming

```
NNNN-short-kebab-title.md
```

Four-digit, zero-padded, monotonically increasing. Numbers are never reused, even if a
spec is superseded. Grab the next free number from the index in
[`../../AGENTS.md`](../../AGENTS.md).

## Writing a good spec

Copy [`TEMPLATE.md`](TEMPLATE.md). Then:

- **Decide something.** A spec that only describes is a wiki page. Each one should close
  at least one open question and say what we are *not* doing.
- **Be concrete about interfaces.** If the stage writes a file, give the record schema
  with field names and types. That schema is a contract other specs depend on.
- **Make acceptance criteria checkable.** "Extraction works well" is not a criterion.
  "≥0.70 precision on the 200-sentence dev set, measured by `scripts/eval_relations.py`"
  is.
- **Keep alternatives.** The paper needs to justify choices; the discarded option and the
  reason are worth more later than they feel now.
- **Park uncertainty in Open questions** rather than guessing in prose.

Length: one screen of decisions beats five screens of context. Cite external sources
rather than restating them.

## Changing an accepted spec

Small correction → edit in place, add a line to the spec's **Changelog**.
Reversing a decision → new spec that supersedes the old one, and set the old one's
`Status: Superseded` and `Superseded by:`.

## Index

The authoritative index lives in [`../../AGENTS.md`](../../AGENTS.md) §3. Update it in the
same commit as any new spec.
