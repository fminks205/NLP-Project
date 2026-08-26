"""Relation clustering and the typology plug-in interface. Implements spec 0003.

This package deliberately does not define relation labels. `cluster` groups candidate
sentences into a field-agnostic intermediate layer (embedding + UMAP + HDBSCAN);
`typology` defines the contract a domain-specific label set must satisfy to consume that
layer. The concrete physics-exemplary label set is a follow-on spec's job.

Heavy dependencies (sentence-transformers, umap-learn, scikit-learn) are declared under
the optional `relations` extra (`uv sync --extra relations`) and imported lazily inside
the functions that need them, so importing this package never requires them.
"""
