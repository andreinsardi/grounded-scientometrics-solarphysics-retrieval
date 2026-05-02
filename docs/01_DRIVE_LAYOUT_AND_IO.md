# Drive Layout and I/O

## Repository location

This repository is stored at:

`C:\Users\andre\odrive\Google Drive\Unicamp\artigo bibliometria\grounded-scientometrics-solarphysics-retrieval`

## Data root

The notebooks are expected to read from and write to the project storage hierarchy rather than temporary session storage.

Primary data root:

`C:\Users\andre\odrive\Google Drive\Unicamp\artigo bibliometria\base de dados\Artigo_Bibliometria Base Bruta\BASES_UNIFICADAS_POR_TEMA`

## Corpus-level organization

Each corpus contains:

- `01_historico_bruto_disponivel`
- `02_historico_consolidado`
- `03_complemento_bruto_2025-09_2026-03`

## Output policy

The staged notebooks should write outputs into corpus-specific directories within the same hierarchy, organized by stage and period.

Expected examples:

- `Nucleo/04_rebuild_outputs/00_consolidacao`
- `Nucleo/04_rebuild_outputs/01_abstract_llm`
- `Nucleo/04_rebuild_outputs/02_retriever_inputs`

The same convention applies to `PIML`, `CombFinal`, and `ML_Multimodal`.

## Rule

No notebook should depend exclusively on ephemeral local files from a transient execution session.
