# Rebuild Overview

## Objetivo

Reconstruir o pipeline do paper do zero, mas usando o codigo e as regras historicas como ancora metodologica.

## Fontes de entrada

As entradas reais do rebuild estao no Google Drive do projeto:

- `BASES_UNIFICADAS_POR_TEMA/<corpus>/01_historico_bruto_disponivel`
- `BASES_UNIFICADAS_POR_TEMA/<corpus>/03_complemento_bruto_2025-09_2026-03`

As referencias historicas consolidadas estao em:

- `BASES_UNIFICADAS_POR_TEMA/<corpus>/02_historico_consolidado`

## Corpora do estudo

1. `Nucleo`
2. `PIML`
3. `CombFinal`
4. `ML_Multimodal`

## Regimes do estudo

- `core`: reconstrucao comparavel ao estudo original
- `holdout`: validacao temporal sem sobreposicao

## Regra de espelhamento

- A familia de aplicacoes do artigo original deve ser reproduzida no `core`.
- Essa mesma familia de aplicacoes deve ser espelhada no `holdout` sempre que isso nao implicar treino ou tuning com dados futuros.
- O que permanece exclusivo do `core` e o treino do retriever especializado.

## Baselines aprovados

- `SciBERT` generico
- `SciBERT_SolarPhysics_Search`
- `BM25`

## Racional dos baselines

- `SciBERT` generico e o baseline denso principal porque permite medir o efeito da especializacao mantendo constante a familia do encoder.
- `SciBERT-SolarPhysics-Search` e a variante especializada por adaptacao de dominio.
- `BM25` entra como baseline lexical classico e interpretavel.
- `BERT` puro nao entra como controle principal, porque para texto cientifico `SciBERT` ja e o baseline generico mais apropriado.
- `Sentence-BERT` e outros retrievers densos externos ficam fora deste ciclo para preservar foco e comparabilidade.

## Referencias para futura redacao do paper

- Beltagy, Lo, Cohan (2019), `SciBERT: A Pretrained Language Model for Scientific Text`.
- Gururangan et al. (2020), `Don't Stop Pretraining: Adapt Language Models to Domains and Tasks`.
- Thakur et al. (2021), `BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models`.
- Wang et al. (2022), `Unsupervised Dense Retrieval for Scientific Articles`.
- Singh et al. (2023), `SciRepEval: A Multi-Format Benchmark for Scientific Document Representations`.

## Regra central

Nao substituir o estudo antigo por um estudo novo. O rebuild precisa manter:

1. uma replicacao limpa no `core`;
2. um espelhamento temporal das aplicacoes no `holdout`;
3. uma comparacao `core vs holdout` como validacao principal.
