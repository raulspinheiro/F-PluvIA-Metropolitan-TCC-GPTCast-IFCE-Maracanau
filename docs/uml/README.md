# 🗂️ Diagramas UML do F-PluvIA Metropolitan (ambiente Cronus)

Diagramas gerados a partir do código salvo de cada célula do notebook [`exec_Cronus_F_PluvIA_VQGAN_RMF.ipynb`](../../notebooks/exec_Cronus_F_PluvIA_VQGAN_RMF.ipynb): um diagrama do fluxo de dados entre as células e um diagrama por célula (48 células, 10.594 linhas de código).

## Fluxo de dados entre as células

![Fluxo de dados entre as células do notebook Cronus](00_fluxo_entre_celulas.png)

O diagrama segue a notação de um modelo relacional:

- **Tabela = célula.** O cabeçalho tem a cor do módulo da célula. As três primeiras linhas trazem as linhas de código, as classes e as funções; abaixo vêm até cinco variáveis que a célula publica para as outras, marcadas com **OUT** (e "+ N variáveis" quando há mais).
- **Conector = dependência de dados.** Liga a célula que cria uma variável (‖, cardinalidade **1**) às células que a usam (pé de galinha, **0..\***). O rótulo da linha mostra as variáveis passadas; "+N" indica outras além das listadas.
- **Configuração global.** `ENV_LABEL`, `ENV_TAG`, `RESULTS_ENV_DIR` e `SERIES_CACHE_DIR` são lidas por quase todas as células e ficam numa tabela à parte, para não poluir o desenho.
- **Células sem ligações** (diagnóstico de retomada, Machine id, Security info, janelas observadas de 2026 e Extra 1) ficam em coluna, abaixo da legenda.

| Cor | Módulo |
| :--- | :--- |
| Vermelho | Configuração, ambiente e cronômetro |
| Azul | Ingestão IMERG e séries espaço-temporais |
| Verde | Ajuste fino e inferência do GPTCast |
| Roxo | Verificação, métricas e alertas por município |
| Laranja | Documentação, proveniência e *deploy* |

O diagrama está em três formatos: [PNG](00_fluxo_entre_celulas.png), [SVG](00_fluxo_entre_celulas.svg) (vetorial, melhor para ampliar) e [`.dot`](00_fluxo_entre_celulas.dot) (Graphviz, com posições fixas; renderize com `neato -n2 -Tpng 00_fluxo_entre_celulas.dot -o 00_fluxo_entre_celulas.png`).

## Diagramas por célula

Cada diagrama mostra as classes, funções, bibliotecas importadas e variáveis da célula. Os arquivos ficam em [`celulas/`](celulas/), nos formatos PNG, SVG e `.dot`.

| # | Célula | Módulo | Linhas | Classes | Funções | Bibliotecas | Diagrama |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 01 | Célula 0 | Configuração | 295 | 0 | 10 | 11 | [PNG](celulas/01_celula_0.png) · [SVG](celulas/01_celula_0.svg) |
| 02 | Célula auxiliar (Cronus) — Diagnóstico de retomada após desconexão (só leitura) | Configuração | 63 | 0 | 1 | 6 | [PNG](celulas/02_celula_auxiliar_cronus_diagnostico_de_retomada.png) · [SVG](celulas/02_celula_auxiliar_cronus_diagnostico_de_retomada.svg) |
| 03 | Célula (Machine id) | Configuração | 23 | 0 | 0 | 1 | [PNG](celulas/03_celula_machine_id.png) · [SVG](celulas/03_celula_machine_id.svg) |
| 04 | Célula (Security info) | Configuração | 38 | 0 | 0 | 2 | [PNG](celulas/04_celula_security_info.png) · [SVG](celulas/04_celula_security_info.svg) |
| 05 | Célula 1 | Configuração | 182 | 0 | 0 | 13 | [PNG](celulas/05_celula_1.png) · [SVG](celulas/05_celula_1.svg) |
| 06 | Célula auxiliar (Cronus - Somente Leitura) — Cópia dos caches .npz da V1 | Ingestão IMERG | 48 | 0 | 0 | 4 | [PNG](celulas/06_celula_auxiliar_cronus_somente_leitura_copia_do.png) · [SVG](celulas/06_celula_auxiliar_cronus_somente_leitura_copia_do.svg) |
| 07 | Célula (Limpeza de Saídas Obsoletas) | Configuração | 145 | 0 | 0 | 8 | [PNG](celulas/07_celula_limpeza_de_saidas_obsoletas.png) · [SVG](celulas/07_celula_limpeza_de_saidas_obsoletas.svg) |
| 08 | Célula 2 | Ingestão IMERG | 426 | 1 | 0 | 12 | [PNG](celulas/08_celula_2.png) · [SVG](celulas/08_celula_2.svg) |
| 09 | Célula 3 | Ingestão IMERG | 373 | 3 | 0 | 12 | [PNG](celulas/09_celula_3.png) · [SVG](celulas/09_celula_3.svg) |
| 10 | Célula 4 | Ajuste fino e inferência | 401 | 0 | 0 | 15 | [PNG](celulas/10_celula_4.png) · [SVG](celulas/10_celula_4.svg) |
| 11 | Célula 4 (conferência rápida) | Ajuste fino e inferência | 6 | 0 | 0 | 2 | [PNG](celulas/11_celula_4_conferencia_rapida.png) · [SVG](celulas/11_celula_4_conferencia_rapida.svg) |
| 12 | Célula 4.1 | Ajuste fino e inferência | 241 | 0 | 2 | 8 | [PNG](celulas/12_celula_4_1.png) · [SVG](celulas/12_celula_4_1.svg) |
| 13 | Célula 4.2 | Ajuste fino e inferência | 152 | 0 | 1 | 6 | [PNG](celulas/13_celula_4_2.png) · [SVG](celulas/13_celula_4_2.svg) |
| 14 | Célula 4.3 (Opcional - Fine-Tuning Suave - Adds Another Dimension of Search) | Ajuste fino e inferência | 346 | 0 | 4 | 15 | [PNG](celulas/14_celula_4_3_opcional_fine_tuning_suave_adds_anot.png) · [SVG](celulas/14_celula_4_3_opcional_fine_tuning_suave_adds_anot.svg) |
| 15 | Célula 4.4 | Ajuste fino e inferência | 263 | 0 | 6 | 8 | [PNG](celulas/15_celula_4_4.png) · [SVG](celulas/15_celula_4_4.svg) |
| 16 | Célula 4.5 | Ajuste fino e inferência | 340 | 0 | 11 | 16 | [PNG](celulas/16_celula_4_5.png) · [SVG](celulas/16_celula_4_5.svg) |
| 17 | Célula 4.6 (Cronus - Fine-tuning VQGAN) | Ajuste fino e inferência | 259 | 0 | 5 | 10 | [PNG](celulas/17_celula_4_6_cronus_fine_tuning_vqgan.png) · [SVG](celulas/17_celula_4_6_cronus_fine_tuning_vqgan.svg) |
| 18 | Célula 4.7 (Cronus - Fine-tuning Transformer) | Ajuste fino e inferência | 123 | 0 | 2 | 11 | [PNG](celulas/18_celula_4_7_cronus_fine_tuning_transformer.png) · [SVG](celulas/18_celula_4_7_cronus_fine_tuning_transformer.svg) |
| 19 | Célula 4.8 (Cronus - V1 vs VQ - Validação 2024) | Ajuste fino e inferência | 235 | 0 | 9 | 14 | [PNG](celulas/19_celula_4_8_cronus_v1_vs_vq_validacao_2024.png) · [SVG](celulas/19_celula_4_8_cronus_v1_vs_vq_validacao_2024.svg) |
| 20 | Célula (Activates Checkpoint) | Ajuste fino e inferência | 151 | 0 | 4 | 6 | [PNG](celulas/20_celula_activates_checkpoint.png) · [SVG](celulas/20_celula_activates_checkpoint.svg) |
| 21 | Célula 5 | Verificação e alertas | 174 | 0 | 5 | 9 | [PNG](celulas/21_celula_5.png) · [SVG](celulas/21_celula_5.svg) |
| 22 | Célula 6 | Verificação e alertas | 153 | 0 | 5 | 9 | [PNG](celulas/22_celula_6.png) · [SVG](celulas/22_celula_6.svg) |
| 23 | Célula 7 | Verificação e alertas | 161 | 0 | 3 | 8 | [PNG](celulas/23_celula_7.png) · [SVG](celulas/23_celula_7.svg) |
| 24 | Célula 8 | Verificação e alertas | 159 | 0 | 4 | 7 | [PNG](celulas/24_celula_8.png) · [SVG](celulas/24_celula_8.svg) |
| 25 | Célula 9 | Verificação e alertas | 216 | 0 | 3 | 8 | [PNG](celulas/25_celula_9.png) · [SVG](celulas/25_celula_9.svg) |
| 26 | Célula 10 | Verificação e alertas | 158 | 1 | 0 | 8 | [PNG](celulas/26_celula_10.png) · [SVG](celulas/26_celula_10.svg) |
| 27 | Célula 11 | Verificação e alertas | 144 | 0 | 4 | 6 | [PNG](celulas/27_celula_11.png) · [SVG](celulas/27_celula_11.svg) |
| 28 | Célula auxiliar (Cronus) — Cópia somente-leitura das janelas observadas de 2026 da V1 (sem as previsões) | Ingestão IMERG | 28 | 0 | 0 | 3 | [PNG](celulas/28_celula_auxiliar_cronus_copia_somente_leitura_da.png) · [SVG](celulas/28_celula_auxiliar_cronus_copia_somente_leitura_da.svg) |
| 29 | Célula 12 | Verificação e alertas | 465 | 0 | 11 | 16 | [PNG](celulas/29_celula_12.png) · [SVG](celulas/29_celula_12.svg) |
| 30 | Célula Estatísticas (Célula 12.1) | Verificação e alertas | 152 | 0 | 3 | 4 | [PNG](celulas/30_celula_estatisticas_celula_12_1.png) · [SVG](celulas/30_celula_estatisticas_celula_12_1.svg) |
| 31 | Célula 13 | Verificação e alertas | 150 | 0 | 3 | 6 | [PNG](celulas/31_celula_13.png) · [SVG](celulas/31_celula_13.svg) |
| 32 | Célula 13.1 | Verificação e alertas | 595 | 0 | 17 | 20 | [PNG](celulas/32_celula_13_1.png) · [SVG](celulas/32_celula_13_1.svg) |
| 33 | Célula Métricas (Célula 13.2) | Verificação e alertas | 259 | 0 | 4 | 4 | [PNG](celulas/33_celula_metricas_celula_13_2.png) · [SVG](celulas/33_celula_metricas_celula_13_2.svg) |
| 34 | Célula 13.3 | Verificação e alertas | 249 | 0 | 4 | 6 | [PNG](celulas/34_celula_13_3.png) · [SVG](celulas/34_celula_13_3.svg) |
| 35 | Célula 13.4 | Verificação e alertas | 291 | 0 | 8 | 9 | [PNG](celulas/35_celula_13_4.png) · [SVG](celulas/35_celula_13_4.svg) |
| 36 | Célula 13.5 | Verificação e alertas | 142 | 0 | 2 | 7 | [PNG](celulas/36_celula_13_5.png) · [SVG](celulas/36_celula_13_5.svg) |
| 37 | Célula 13.6 (Cronus) | Ajuste fino e inferência | 64 | 0 | 1 | 1 | [PNG](celulas/37_celula_13_6_cronus.png) · [SVG](celulas/37_celula_13_6_cronus.svg) |
| 38 | Célula 13.7 (Cronus) | Verificação e alertas | 299 | 0 | 9 | 14 | [PNG](celulas/38_celula_13_7_cronus.png) · [SVG](celulas/38_celula_13_7_cronus.svg) |
| 39 | Célula 13.7.1 (Cronus) | Verificação e alertas | 379 | 0 | 8 | 10 | [PNG](celulas/39_celula_13_7_1_cronus.png) · [SVG](celulas/39_celula_13_7_1_cronus.svg) |
| 40 | Célula 13.7.2 (Cronus) | Verificação e alertas | 324 | 0 | 8 | 6 | [PNG](celulas/40_celula_13_7_2_cronus.png) · [SVG](celulas/40_celula_13_7_2_cronus.svg) |
| 41 | Célula Extra 1 | Documentação e deploy | 268 | 0 | 6 | 14 | [PNG](celulas/41_celula_extra_1.png) · [SVG](celulas/41_celula_extra_1.svg) |
| 42 | Célula auxiliar (Cronus - Retomada do Fechamento, Somente Leitura) | Documentação e deploy | 63 | 0 | 0 | 2 | [PNG](celulas/42_celula_auxiliar_cronus_retomada_do_fechamento_s.png) · [SVG](celulas/42_celula_auxiliar_cronus_retomada_do_fechamento_s.svg) |
| 43 | Célula Full UMLs | Documentação e deploy | 397 | 0 | 13 | 14 | [PNG](celulas/43_celula_full_umls.png) · [SVG](celulas/43_celula_full_umls.svg) |
| 44 | Célula Extra 2 | Documentação e deploy | 284 | 0 | 5 | 13 | [PNG](celulas/44_celula_extra_2.png) · [SVG](celulas/44_celula_extra_2.svg) |
| 45 | Célula Base (Pseudo-vista) | Documentação e deploy | 364 | 1 | 5 | 14 | [PNG](celulas/45_celula_base_pseudo_vista.png) · [SVG](celulas/45_celula_base_pseudo_vista.svg) |
| 46 | Célula (Deploy Automatizado) | Documentação e deploy | 324 | 0 | 4 | 12 | [PNG](celulas/46_celula_deploy_automatizado.png) · [SVG](celulas/46_celula_deploy_automatizado.svg) |
| 47 | Célula Backup | Documentação e deploy | 190 | 0 | 1 | 12 | [PNG](celulas/47_celula_backup.png) · [SVG](celulas/47_celula_backup.svg) |
| 48 | Célula (Encerramento do Ambiente Cronus) | Documentação e deploy | 32 | 0 | 0 | 2 | [PNG](celulas/48_celula_encerramento_do_ambiente_cronus.png) · [SVG](celulas/48_celula_encerramento_do_ambiente_cronus.svg) |

## Como são gerados

A **Célula Full UMLs** lê o código de cada célula, gera os diagramas por célula e o fluxo entre células, e a célula de *Deploy* os publica nesta pasta. A pasta é **regravada a cada execução do Deploy**: o estilo relacional do fluxo foi aplicado depois da última execução e precisa ser incorporado à Célula Full UMLs para não voltar ao formato anterior.
