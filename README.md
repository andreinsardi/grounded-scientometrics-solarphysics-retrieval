# Grounded Scientometrics Solar Physics Retrieval

Rebuild controlado do pipeline do artigo de Scientometrics sobre retrieval especializado em Solar Physics.

Este repositorio foi reiniciado para concentrar apenas:

- notebooks novos do rebuild;
- documentacao tecnica do rebuild;
- convencoes de entrada e saida;
- artefatos leves de apoio ao desenvolvimento.

O objetivo nao e reinventar o metodo. O objetivo e reconstruir o pipeline do zero, mantendo comparabilidade com a versao submetida do paper.

## Regras do projeto

- Preservar o metodo historico sempre que possivel.
- Melhorar o pipeline sem eliminar regras antigas.
- Tratar `01_historico_bruto_disponivel` e `03_complemento_bruto_2025-09_2026-03` como entradas reais.
- Tratar `02_historico_consolidado` como referencia de auditoria e comparacao.
- Separar sempre `core` e `holdout`.
- Replicar no `core` a familia de aplicacoes do artigo original.
- Espelhar essa mesma familia de aplicacoes no `holdout` sempre que isso nao implicar treino ou tuning com dados futuros.
- Manter o treino do retriever especializado exclusivo do `core`.
- Nao commitar outputs pesados do processamento neste repositorio.

## Janela temporal congelada

- `core`: `2020-01-01` a `2025-09-30`
- `holdout`: `2025-10-01` a `2026-03-31`

## Estrutura inicial

- [docs](C:/Users/andre/odrive/Google%20Drive/Unicamp/artigo%20bibliometria/grounded-scientometrics-solarphysics-retrieval/docs)
- [notebooks](C:/Users/andre/odrive/Google%20Drive/Unicamp/artigo%20bibliometria/grounded-scientometrics-solarphysics-retrieval/notebooks)
- [scripts](C:/Users/andre/odrive/Google%20Drive/Unicamp/artigo%20bibliometria/grounded-scientometrics-solarphysics-retrieval/scripts)
- [config](C:/Users/andre/odrive/Google%20Drive/Unicamp/artigo%20bibliometria/grounded-scientometrics-solarphysics-retrieval/config)

## Ordem de trabalho

1. Construir o notebook de consolidacao e deduplicacao.
2. Reconstruir as bases `core` e `holdout`.
3. Recriar os notebooks `Abstract_LLM_*` para `core` e `holdout`.
4. Recriar o treino do `SciBERT_SolarPhysics_Search` apenas no `core`.
5. Recriar os pipes 2, 3 e 4 para replicacao no `core` e espelhamento temporal no `holdout`.
6. So depois reescrever o artigo.
