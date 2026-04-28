# Notebook Conventions

## Regras gerais

Todos os notebooks devem:

- ser executáveis manualmente no Colab Pro;
- montar o Google Drive explicitamente;
- trabalhar célula por célula;
- ter comentários claros em cada bloco;
- imprimir métricas intermediárias;
- salvar logs e saídas no Drive;
- deixar rastreável o andamento do processamento.

## Convenção de nomes

Os notebooks novos devem seguir uma ordem explícita:

1. `00_consolidacao_rebuild_core_holdout.ipynb`
2. `01_abstract_llm_nucleo_core_holdout.ipynb`
3. `02_abstract_llm_piml_core_holdout.ipynb`
4. `03_abstract_llm_combfinal_core_holdout.ipynb`
5. `04_abstract_llm_ml_core_holdout.ipynb`
6. `05_scibert_solarphysics_search_rebuild.ipynb`
7. `06_pipe2_retriever_analytics_rebuild.ipynb`
8. `07_pipe3_agent_scientometrics_rebuild.ipynb`
9. `08_pipe4_statistical_validation_rebuild.ipynb`

## Política de GPU

### Usar GPU

- `01_abstract_llm_nucleo_core_holdout.ipynb`
- `02_abstract_llm_piml_core_holdout.ipynb`
- `03_abstract_llm_combfinal_core_holdout.ipynb`
- `04_abstract_llm_ml_core_holdout.ipynb`
- `05_scibert_solarphysics_search_rebuild.ipynb`

### Não precisa de GPU

- `00_consolidacao_rebuild_core_holdout.ipynb`
- `06_pipe2_retriever_analytics_rebuild.ipynb` em princípio
- `07_pipe3_agent_scientometrics_rebuild.ipynb` em princípio
- `08_pipe4_statistical_validation_rebuild.ipynb`

## Saídas obrigatórias por notebook

Cada notebook deve salvar:

- log textual;
- métricas de progresso;
- tabelas auxiliares;
- arquivos consolidados intermediários;
- manifestos ou auditorias de paridade quando aplicável.
