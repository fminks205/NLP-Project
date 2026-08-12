"""Corpus acquisition stages. Implements spec 0001.

Three stages, deliberately separate commands so a failed fetch never forces a
re-query and a cleaning bug never forces a re-fetch:

    seed  -> data/raw/seed.jsonl
    fetch -> data/raw/html/{qid}.html  + data/raw/fetch_log.jsonl
    clean -> data/interim/documents.jsonl

`estimate` is a read-only helper that projects download size before fetching.
"""
