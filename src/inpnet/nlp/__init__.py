"""Entity and mention layer. Implements spec 0002.

    resolve  -> data/interim/wikidata_cache.jsonl   (which link targets are people)
    segment  -> data/interim/sentences.jsonl        (sentence boundaries)
    mentions -> data/interim/mentions.jsonl, entities.jsonl, candidates.jsonl

Model-free by design: the mentions here come from human-curated wiki links and from the
article subject, so this layer carries no NER or coreference error. Model-derived
mentions are added later by the GPU stage, using enum values the schema already reserves.
"""
