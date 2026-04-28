# Drive Layout And IO

## Local de trabalho do repo

Este repositório fica em:

`C:\Users\andre\odrive\Google Drive\Unicamp\artigo bibliometria\grounded-scientometrics-solarphysics-retrieval`

## Raiz dos dados

Os notebooks devem ler e escrever na estrutura do Google Drive, não em armazenamento temporário do Colab.

Raiz principal dos dados:

`C:\Users\andre\odrive\Google Drive\Unicamp\artigo bibliometria\base de dados\Artigo_Bibliometria Base Bruta\BASES_UNIFICADAS_POR_TEMA`

## Organização por corpus

Cada corpus tem:

- `01_historico_bruto_disponivel`
- `02_historico_consolidado`
- `03_complemento_bruto_2025-09_2026-03`

## Política de saída

Os notebooks novos devem gerar outputs em pastas de saída dentro dessa mesma hierarquia do Drive, por corpus, por etapa e por período.

Exemplo esperado:

- `Nucleo/04_rebuild_outputs/00_consolidacao`
- `Nucleo/04_rebuild_outputs/01_abstract_llm`
- `Nucleo/04_rebuild_outputs/02_retriever_inputs`

O mesmo vale para `PIML`, `CombFinal` e `ML_Multimodal`.

## Regra

Nenhum notebook deve depender apenas de arquivos locais efêmeros do Colab.
