# Notebook Conventions

## General rules

All notebooks should:

- run manually in Colab Pro;
- mount project storage explicitly;
- be executed stage by stage;
- include clear comments for each block;
- print intermediate metrics and progress updates;
- persist logs and outputs to project storage;
- keep processing steps traceable.

## Naming convention

The notebooks follow an explicit execution order:

1. `00_consolidacao_rebuild_core_holdout.ipynb`
2. `01_abstract_llm_nucleo_core_holdout.ipynb`
3. `02_abstract_llm_piml_core_holdout.ipynb`
4. `03_abstract_llm_combfinal_core_holdout.ipynb`
5. `04_abstract_llm_ml_core_holdout.ipynb`
6. `05_scibert_solarphysics_search_rebuild.ipynb`
7. `06_pipe2_retriever_analytics_rebuild.ipynb`
8. `07_pipe3_agent_scientometrics_rebuild.ipynb`
9. `08_pipe4_statistical_validation_rebuild.ipynb`

## GPU policy

### GPU recommended

- `01_abstract_llm_nucleo_core_holdout.ipynb`
- `02_abstract_llm_piml_core_holdout.ipynb`
- `03_abstract_llm_combfinal_core_holdout.ipynb`
- `04_abstract_llm_ml_core_holdout.ipynb`
- `05_scibert_solarphysics_search_rebuild.ipynb`

### CPU is sufficient

- `00_consolidacao_rebuild_core_holdout.ipynb`
- `06_pipe2_retriever_analytics_rebuild.ipynb`
- `07_pipe3_agent_scientometrics_rebuild.ipynb`
- `08_pipe4_statistical_validation_rebuild.ipynb`

## Required outputs by notebook

Each notebook should save:

- textual logs;
- progress metrics;
- auxiliary tables;
- consolidated intermediate files;
- manifests or audit artifacts when applicable.
