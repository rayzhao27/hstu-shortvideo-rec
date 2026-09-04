"""KuaiRand processing pipeline. This directory is Python, not the dumps.

Dumps live in ``datasets/`` (gitignored):

    datasets/raw/KuaiRand-Pure/     downloaded CSVs
    datasets/processed/             parquet caches, sequences, encoders

Default release is KuaiRand-Pure (27k users, 7.5k videos, 194MB). That is not
the full log — KuaiRand-1K and KuaiRand-27K exist and are opt-in via
``--dataset``. Run everything from the repo root:

    python -m data.explore          # Stage 0
    python -m data.preprocess       # Stage 1
    python -m data.verify
"""
