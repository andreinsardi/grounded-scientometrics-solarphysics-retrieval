# Repository Overview

## Purpose

This repository supports the staged reconstruction of the Solar Physics retrieval workflow while preserving comparability with the historical study design.

## Input sources

The working inputs live in the project storage hierarchy:

- `BASES_UNIFICADAS_POR_TEMA/<corpus>/01_historico_bruto_disponivel`
- `BASES_UNIFICADAS_POR_TEMA/<corpus>/03_complemento_bruto_2025-09_2026-03`

Historical consolidated references live in:

- `BASES_UNIFICADAS_POR_TEMA/<corpus>/02_historico_consolidado`

## Study corpora

1. `Nucleo`
2. `PIML`
3. `CombFinal`
4. `ML_Multimodal`

## Study regimes

- `core`: comparable reconstruction of the main study regime
- `holdout`: temporally disjoint validation regime

## Mirroring rule

- The original family of analytical applications must be reproduced on `core`.
- The same family of applications must be mirrored on `holdout` whenever that does not imply training or tuning on future data.
- Specialized retriever training remains exclusive to `core`.

## Approved baselines

- generic `SciBERT`
- `SciBERT_SolarPhysics_Search`
- `BM25`

## Baseline rationale

- Generic `SciBERT` is the main dense baseline because it isolates the effect of domain specialization while keeping the encoder family fixed.
- `SciBERT_SolarPhysics_Search` is the domain-adapted specialized variant.
- `BM25` is included as a standard lexical baseline.
- Plain `BERT` is not used as the main generic control because `SciBERT` is already the appropriate scientific-text baseline.
- Additional external dense retrievers are excluded to preserve focus and comparability.

## Central rule

The repository should preserve:

1. a clean replication regime on `core`;
2. a temporal mirroring regime on `holdout`;
3. an explicit `core` versus `holdout` comparison as the main validation layer.
