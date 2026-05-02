# Grounded Scientometrics for Solar Physics Retrieval

This repository contains the execution notebooks, configuration files, and lightweight documentation used to reproduce the retrieval workflow for the Solar Physics scientometrics study.

The repository is organized around a staged pipeline that preserves the historical study design while making the execution order, corpus roles, and output conventions explicit.

## Repository scope

This repository keeps only versionable assets:

- execution notebooks;
- configuration files;
- repository documentation;
- lightweight helper material needed to understand the workflow.

Large raw corpora, intermediate tables, vector indices, and other heavy outputs remain in the project storage hierarchy and are not tracked here.

## Frozen temporal design

- `core`: `2020-01-01` to `2025-09-30`
- `holdout`: `2025-10-01` to `2026-03-31`

## Methodological rules

- Preserve the historical method wherever possible.
- Apply the same data-engineering treatment to `core` and `holdout`.
- Train the specialized retriever only on thematic `core` corpora.
- Use `ML_core` and `ML_holdout` only as application and evaluation corpora.
- Mirror the same analytical query family from `core` to `holdout`.
- Keep `SciBERT` as the generic dense baseline.
- Keep `BM25` as the lexical baseline.

## Repository structure

- `docs/` repository-level documentation and execution conventions
- `notebooks/` staged notebooks for consolidation, semantic structuring, retriever training, application, and validation
- `config/` lightweight configuration assets

## Notebook order

1. `00_consolidacao_rebuild_core_holdout.ipynb`
2. `01_abstract_llm_nucleo_core_holdout.ipynb`
3. `02_abstract_llm_piml_core_holdout.ipynb`
4. `03_abstract_llm_combfinal_core_holdout.ipynb`
5. `04_abstract_llm_ml_core_holdout.ipynb`
6. `05_scibert_solarphysics_search_rebuild.ipynb`
7. `06_pipe2_retriever_analytics_rebuild.ipynb`
8. `07_pipe3_agent_scientometrics_rebuild.ipynb`
9. `08_pipe4_statistical_validation_rebuild.ipynb`
10. `09_methodological_storyline_pack.ipynb`

## Reproducibility notes

- The notebooks are designed for explicit, stepwise execution.
- Inputs and outputs are anchored to the project storage hierarchy rather than temporary session storage.
- Progress logs, audit tables, and intermediate artifacts are written by stage so that execution remains traceable.
