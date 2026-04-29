# Plano de Reescrita do Manuscrito para o Major Review

Data de referencia: `2026-04-29`

## Objetivo

Este plano existe para orientar a reescrita do manuscrito atual em `sn-article.tex` a partir dos resultados finais congelados do rebuild `00-09`, preservando:

- a comparabilidade com o paper original;
- as respostas metodologicas exigidas pelo `major review`;
- a distincao entre `replicacao no core` e `validacao temporal no holdout`;
- a narrativa correta sobre ganhos do retriever especializado e robustez do pipeline.

---

## Diagnostico executivo

### O que o rebuild demonstrou

1. A historia metodologica do paper original foi recuperada e tornada auditavel.
2. O treinamento do retriever especializado foi refeito de forma temporalmente limpa.
3. A familia de aplicacoes do paper original foi replicada no `core` e espelhada no `holdout`.
4. `BM25` foi incorporado como unico baseline extra aprovado.
5. O pacote final ganhou rastreabilidade, manifests, logs, `docs_master`, `index_map`, indices FAISS e auditoria estatistica.
6. O `Pipe 09` tornou explicita a costura entre metodo historico, rebuild canonico e narrativa para o manuscrito.

### Resultado central que deve guiar a nova narrativa

O **principal achado positivo** do rebuild permanece no `Pipe 05`:

- o retriever especializado continua superando o baseline generico em metricas paper-facing;
- o comportamento central do paper original foi preservado;
- o rebuild ficou suficientemente proximo do original para sustentar continuidade metodologica, nao apenas uma extensao exploratoria.

### Resultado central que exige cuidado narrativo

O `Pipe 08` **nao deve ser usado como manchete de eficacia entre retrievers** com base em `raw top1 score`, porque:

- os scores brutos entre `generic`, `specialized` e `BM25` nao estao na mesma escala;
- o valor do `Pipe 08` e de **robustez temporal e estatistica**, nao de benchmark principal de superioridade.

### Conclusao editorial

O manuscrito revisado deve ser organizado em torno de quatro mensagens:

1. **replicamos o paper original em um regime limpo (`core`)**;
2. **adicionamos validacao temporal explicita (`holdout`)**;
3. **preservamos o ganho do retriever especializado em benchmark controlado (`Pipe 05`)**;
4. **aumentamos rastreabilidade, comparabilidade e robustez do pipeline inteiro (`06-09`)**.

---

## Respostas diretas aos avaliadores

### 1. Possivel vazamento temporal

**Resposta direta**:

- o retriever especializado foi treinado apenas em `Nucleo_core + PIML_core + CombFinal_core`;
- `ML` nao entrou no treino;
- `ML_core` e `ML_holdout` foram usados apenas para aplicacao e avaliacao;
- o `holdout` foi definido como `2025-10-01` a `2026-03-31`, sem sobreposicao com o `core`.

**Onde isso entra no manuscrito**:

- `Methodology`
- `Experimental Design and Data`
- `Reproducibility details`

### 2. Falta de validacao temporal

**Resposta direta**:

- o estudo passou a ter dois regimes explicitos:
  - `core`: replicacao principal do paper original
  - `holdout`: espelhamento temporal da mesma familia de aplicacoes
- a comparacao `core vs holdout` agora e parte constitutiva do estudo revisado.

**Onde isso entra no manuscrito**:

- `Abstract`
- `Introduction`
- `Methodology`
- `Results`
- `Discussion`

### 3. Falta de baseline adicional

**Resposta direta**:

- mantivemos `SciBERT` generico como baseline denso principal para isolar o efeito da especializacao;
- adicionamos `BM25` como baseline lexical classico, interpretable e barato, exatamente como previsto no plano.

**Onde isso entra no manuscrito**:

- `Methodology`
- `Retrieval experimental conditions`
- `Results`
- `Discussion`

### 4. Rastreabilidade e reproducibilidade insuficientes

**Resposta direta**:

- o rebuild passou a produzir `run_metadata`, manifests, logs, `docs_master`, `index_map`, indices FAISS, bundles grounded e pacotes tabulares por camada;
- o modelo especializado foi republicado no Hugging Face.

**Onde isso entra no manuscrito**:

- `Methodology`
- `Experimental flow`
- `Reproducibility details`
- `Conclusion`

### 5. Claims mais fortes do que o pacote sustentava

**Resposta direta**:

- a narrativa revisada deve separar claramente:
  - evidencia principal de eficacia do retriever (`Pipe 05`);
  - aplicacao e interpretabilidade (`Pipe 07`);
  - robustez temporal e estatistica (`Pipe 08`);
  - costura metodologica e comparativa (`Pipe 09`).

**Onde isso entra no manuscrito**:

- `Results`
- `Discussion`
- `Conclusion`

---

## Resultados congelados que devem orientar a reescrita

## Camada 0-1: corpora e estrutura semantica

### Historico -> rebuild

- `Nucleo total`: `7349 -> 9977`
- `PIML total`: `12866 -> 28425`
- `CombFinal total`: `982 -> 1858`
- `ML total`: `135489` no rebuild consolidado

### Leitura correta

- houve **expansao real do corpus**, nao uma replicacao numerica 1:1;
- isso deve ser descrito como efeito de rebuild mais completo e recorte temporal explicito;
- a comparabilidade principal deve ser defendida no nivel **metodologico e funcional**, nao no nivel de identidade bruta de volume.

### Abstract_LLM

- `Nucleo_core_rows_used`: `5036 -> 5203`
- `PIML_core_rows_used`: `8955 -> 9976`
- `CombFinal_core_rows_used`: `780 -> 853`
- `ML_core_rows_used`: `83771 -> 99666`
- `k=30` foi preservado no `core` para os quatro corpora tematicos.

### Leitura correta

- o upstream semantico ficou proximo do paper original no `core`;
- o `holdout` deve ser apresentado como espelhamento temporal e nao como tentativa de identidade estrutural com o `core`.

## Camada 1: SciBERT-SolarPhysics-Search

### Historico -> rebuild

- `DAPT docs`: `21197 -> 22047`
- `perplexity`: `3.12 -> 3.27`
- `contrastive pairs`: `36424 -> 43314`
- `A/B docs`: `4555 -> 4557`
- `A/B classes`: `15 -> 16`

### Benchmark principal (`Pipe 05`)

- `MRR`: `0.6085 -> 0.6594`
- `Recall@10`: `0.8642 -> 0.8931`
- `Recall@50`: `0.9849 -> 0.9897`
- `Recall@100`: `0.9945 -> 0.9963`
- `NMI`: `0.0860 -> 0.0984`
- `ARI`: `0.0318 -> 0.0437`
- `Silhouette`: `0.1237 -> 0.1502`
- `NearestCentroidAcc`: `0.2888 -> 0.4531`

### Leitura correta

- este e o **resultado central de eficacia** do paper revisado;
- o retriever especializado continua melhor que o baseline generico em metricas rank-based, clustering e centroid;
- a republicacao no Hugging Face reforca a reproducibilidade.

## Camada 2: aplicacao e rastreabilidade

- `pipe2 docs_master historico`: `108383`
- `pipe2 docs_master rebuild principal`: `110513`
- `Pipe 06` reconstruiu `docs_master`, `index_map`, query bank e candidatos de white space por periodo.

### Leitura correta

- esta camada sustenta a resposta a criticas de rastreabilidade;
- deve ser apresentada como infraestrutura de aplicacao e auditoria.

## Camada 3: bundles grounded e baseline lexical

- `60` queries agregadas no `Pipe 07`
- bundles grounded e saidas agent-like produzidos para `core` e `holdout`
- `BM25` incluido no downstream

### Leitura correta

- o foco aqui nao e "provar" sozinho a superioridade do retriever;
- o foco e mostrar o que a infraestrutura recuperada permite observar, comparar e sustentar com evidencia rastreavel.

## Camada 4: robustez estatistica

- `2000` rodadas de bootstrap
- comparacoes por periodo
- auditoria manual pequena

### Leitura correta

- o `Pipe 08` responde a necessidade de robustez e sanity check;
- ele nao deve substituir o `Pipe 05` como benchmark principal de eficacia entre retrievers.

---

## Mudanca de entendimento antes e depois dos resultados

## Antes do rebuild final

A narrativa implicita era:

- o paper original tinha uma boa intuicao experimental;
- faltava principalmente limpeza metodologica e uma melhor explicacao da cadeia de passos;
- o major review parecia exigir sobretudo reorganizacao e um baseline adicional.

## Depois dos resultados congelados

O entendimento ficou mais forte e mais preciso:

1. o paper original tinha, de fato, **substancia experimental real**;
2. o maior problema era a **compressao da historia metodologica** e a **mistura entre evidencias de natureza diferente**;
3. a contribuicao mais forte do rebuild nao foi "inventar um estudo novo", mas **separar replicacao, extensao temporal, baseline lexical, robustez estatistica e rastreabilidade**;
4. o major review pode ser respondido sem abandonar a tese original, desde que a narrativa mude.

---

## Plano de reescrita secao por secao

## 1. Titulo

### Manter

- a ideia central de que LLMs grounded falham sem retrieval especializado.

### Ajustar

- nao precisa mudar o titulo se o foco continuar em retrieval especializado + grounding;
- opcionalmente inserir no subtitulo ou no abstract a ideia de `core replication + temporal holdout`.

## 2. Abstract

### Reescrever quase por completo

O abstract novo deve ter quatro movimentos:

1. problema: retrieval e grounding moldam o evidence space;
2. metodo: rebuild em `core` e `holdout`, treino `core-only`, `BM25` extra;
3. resultado principal: specialized retriever supera baseline no benchmark controlado do `Pipe 05`;
4. contribuicao do major review: validacao temporal, rastreabilidade e robustez estatistica.

### Frases que devem aparecer

- explicitacao de `core` e `holdout`;
- explicacao de que `BM25` foi adicionado;
- nota de que o specialized retriever foi republicado no HF;
- linguagem cautelosa sobre o papel do `Pipe 08`.

## 3. Introduction

### Reescrever parcialmente

Objetivo:

- deslocar a introducao de "um estudo pontual sobre LLMs" para "um estudo experimental com replicacao e validacao temporal".

### Inserir

- porque o `major review` exigiu separacao temporal explicita;
- porque retrieval e baselines devem ser tratados como variaveis causais na construcao do evidence space;
- porque o estudo revisado e mais defensavel do que a versao submetida.

### Evitar

- prometer que toda melhoria estatistica downstream prova superioridade universal do specialized retriever;
- vender o `holdout` como se fosse um novo treino ou um novo paper.

## 4. Background and Related Work

### Reescrever pontualmente

### Fortalecer

- justificativa para `SciBERT` generico como baseline denso;
- justificativa para `BM25` como baseline lexical;
- relacao entre DAPT, retrieval especializado e avaliacao cientifica.

### Inserir

- a distincao entre representational adequacy e epistemic validity;
- a ideia de validacao temporal como resposta metodologica, nao como expansao arbitraria de escopo.

## 5. Methodology

### Esta e a secao mais importante a reescrever

### Nova estrutura recomendada

1. **Historical method recovered**
   - scripts R
   - `Abstract_LLM_*`
   - `SciBERT-SolarPhysics-Search`
   - aplicacao em `ML`
   - `Piepe_3`
   - `PIPE_4`

2. **Canonical rebuild design**
   - `core`: `2020-01-01` a `2025-09-30`
   - `holdout`: `2025-10-01` a `2026-03-31`
   - mesmas regras de engenharia de dados
   - treino `core-only`
   - aplicacao `ML_core` e `ML_holdout`

3. **Baselines and controls**
   - `SciBERT` generico
   - `SciBERT-SolarPhysics-Search`
   - `BM25`

4. **Layered pipeline**
   - `00`: consolidacao
   - `01-04`: estrutura semantica
   - `05`: Phase I benchmark principal
   - `06`: aplicacao / corpus bridge
   - `07`: bundles grounded e tabelas paper-ready
   - `08`: robustez temporal e estatistica
   - `09`: storyline e comparacoes historico vs rebuild

### O que retirar ou rebaixar

- qualquer formulacao que ainda trate o estudo como um pipeline unico e linear sem separar `replicacao` de `holdout`;
- qualquer passagem que sugira que o agente LLM sozinho e o foco causal principal do estudo.

## 6. Experimental Design and Data

### Reescrever quase toda a secao

### Precisa passar a conter

- os quatro corpora e seus papeis;
- o split `core/holdout`;
- a regra de treino `core-only`;
- o papel do `ML` como application corpus;
- a distincao entre benchmark principal (`Pipe 05`) e aplicacao downstream (`06-08`).

### Tabelas novas ou atualizadas

- tabela de volumes por corpus e por periodo;
- tabela de papel metodologico de cada corpus;
- tabela curta de baselines e regimes (`generic`, `specialized`, `BM25`).

### Explicacao obrigatoria

- o aumento de volume nos corpora nao invalida a comparabilidade, porque a comparabilidade principal e de desenho e familia de operacoes.

## 7. Results

### Esta secao deve ser reorganizada

### Estrutura recomendada

1. **Results of core replication**
   - `Abstract_LLM` proximo do historico
   - `k=30` preservado no `core`

2. **Results of rebuilding SciBERT-SolarPhysics-Search**
   - usar o `Pipe 05` como benchmark principal
   - mostrar as metricas rank-based, clustering e centroid
   - destacar a proximidade com o historico

3. **Application-layer reconstruction**
   - `docs_master`, `index_map`, FAISS, query bank
   - bundles grounded e exemplos de gap

4. **Temporal holdout results**
   - mostrar o `holdout` como espelho temporal
   - enfatizar que a estrutura principal de gaps veio do `core`

5. **Robustness results**
   - bootstrap
   - deltas por periodo
   - auditoria manual pequena

### Ponto critico

A subsecao atual de `Results at the LLM-based agency level` provavelmente precisa ser **reescrita ou rebaixada**. Ela ainda esta fortemente ancorada na narrativa antiga de `A1-A4`, `ICR` e `UCR`. Antes de manter esses numeros como centro do paper revisado, deve-se verificar se eles foram de fato reconstruidos na trilha final congelada. Se nao tiverem sido, o caminho seguro e:

- manter H1-H3 como intuicao teorica;
- deslocar a evidencia quantitativa principal para `Pipe 05`, `06`, `07` e `08`;
- mover a narrativa antiga de agency-level para contexto historico ou apendice, se necessario.

## 8. Discussion

### Reescrever com foco em quatro ideias

1. retrieval e infraestrutura sao variaveis causais;
2. especializacao de dominio melhora o benchmark principal;
3. grounding e bundles aumentam interpretabilidade, mas nao dispensam desenho metodologico;
4. robustez temporal e estatistica tornam o paper mais defensavel.

### O que evitar

- usar `raw top1 score` do `Pipe 08` como argumento de que o specialized piorou;
- misturar resultados de benchmark com resultados de aplicacao como se fossem a mesma coisa.

## 9. Implications for Scientometrics

### Reescrever parcialmente

### Nova enfase

- a contribuicao nao e apenas "LLMs grounded falham sem retrieval especializado";
- a contribuicao e tambem que **um workflow scientometrico com retriever especializado precisa ser desenhado, auditado e temporalmente validado**;
- retrieval passa a ser defendido explicitamente como parte causal da epistemologia aplicada da scientometrics.

## 10. Conclusion, limitations, and future work

### Reescrever parcialmente

### Conclusoes a manter

- retrieval especializado importa;
- grounding sozinho nao resolve tudo;
- estrutura e controle metodologico continuam centrais.

### Novas conclusoes

- o rebuild produziu uma replicacao mais auditavel do estudo;
- o `holdout` respondeu a critica temporal;
- `BM25` fortaleceu o baseline;
- a narrativa revisada precisa ser mais cautelosa sobre o que e prova principal e o que e robustez complementar.

### Limitacoes a explicitar

- corpora ampliados em relacao ao historico;
- `holdout` como espelhamento temporal, nao novo treino;
- comparacoes de `raw score` entre sistemas heterogeneos devem ser interpretadas com cautela;
- a velha narrativa agency-level pode precisar ser rebaixada se nao for totalmente reproduzida no rebuild final.

## 11. Reproducibility details

### Ampliar

Inserir ou reforcar:

- janelas `core` e `holdout`;
- corpora usados no treino;
- `ML` apenas como aplicacao;
- HF publish do retriever especializado;
- `docs_master`, `index_map`, FAISS;
- baseline `BM25`;
- bootstrap e auditoria manual pequena.

---

## Ponto que ainda exige decisao editorial

Este e o unico ponto que eu ainda trataria como **risco real de manuscrito**:

### Compatibilizacao da narrativa atual de H1-H3 / A1-A4 com o rebuild final

O manuscrito atual ainda esta fortemente escrito em torno de:

- `A1-A4`
- `ICR`
- `UCR`
- uma leitura mais direta da camada de agency-level

Ja o rebuild congelado ficou mais forte em:

- benchmark controlado do retriever (`Pipe 05`)
- reconstrucao de infraestrutura e bundles (`06-07`)
- robustez temporal e bootstrap (`08`)
- narrativa metodologica explicita (`09`)

### Recomendacao

Adotar uma das duas rotas:

1. **Rota conservadora e recomendada**
   - manter H1-H3 como moldura interpretativa;
   - reconstruir a secao de Results/Discussion com base primaria em `05-09`;
   - tratar a velha matriz A1-A4 como antecedente historico, nao como eixo unico do paper revisado.

2. **Rota de alto risco**
   - tentar manter a narrativa antiga como eixo central sem verificar equivalencia integral com os novos artefatos.

Recomendacao final: **seguir a rota 1**.

---

## Checklist final antes da reescrita no Overleaf

- atualizar `Abstract`
- reescrever `Methodology`
- reescrever `Experimental Design and Data`
- reorganizar `Results`
- reescrever `Discussion`
- ajustar `Implications` e `Conclusion`
- revisar `Reproducibility details`
- alinhar tabelas e figuras aos resultados congelados
- explicitar `core replication` vs `holdout validation`
- deixar `Pipe 05` como evidencia principal de eficacia
- deixar `Pipe 08` como robustez, nao como benchmark principal

---

## Decisao final

Do ponto de vista metodologico, o `major review` foi respondido.

Do ponto de vista de resultados, o paper revisado tem base suficiente para ser defendido.

O trabalho que resta agora e **editorial e argumentativo**, nao de reconstruir um novo experimento:

- costurar a narrativa correta;
- substituir valores e tabelas antigas;
- explicitar melhor a cadeia metodologica;
- responder os avaliadores com base nos resultados ja congelados.
