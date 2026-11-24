# Manifesto de integridade e proveniência — F-PluvIA Metropolitan

**Emissão:** 24/11/2026 08:57:49 (Fortaleza) · 2026-11-24 11:57:49 UTC  
**Ambiente:** NVIDIA RTX PRO 6000 Blackwell Server Edition (95.0 GB) · Cronus · GPU NVIDIA RTX PRO 6000 Blackwell Server Edition · Python 3.13.16  
**Autor:** Raul Santiago Pinheiro · **Orientador:** Amauri Holanda de Souza Júnior · Instituto Federal de Educação, Ciência e Tecnologia do Ceará (IFCE) — Campus Maracanaú · 2026.2

## O que este manifesto comprova

Que os arquivos listados não mudaram desde a emissão: basta recalcular o SHA-256 de cada um (ou rodar "sha256sum -c" no arquivo SHA256SUMS). Não comprova autoria nem o hardware usado; esses campos são declarados pelo ambiente no momento da emissão.

**Hash raiz (SHA-256 da lista de arquivos):** `8648c0fac27b204ca677796a80882d4921d3a63f3b4fc6453b16f3fa9e7edc90`

## Artefatos por pasta

| Pasta | Arquivos | Tamanho |
|---|---:|---:|
| checkpoints | 10 | 20,686.1 MB |
| comparativo_hardware | 3 | 0.2 MB |
| diagramas_uml | 98 | 8.6 MB |
| notebook | 1 | 12.7 MB |
| resultados | 72 | 334.4 MB |
| resultados_G4_RTX6000_Blackwell_95GB_Cronus | 339 | 474.2 MB |
| series_2km | 31 | 1,330.1 MB |

## Execução do modelo

- **checkpoint:** `fpluvia_vq_rmf_2km_G4_RTX6000_Blackwell_95GB_Cronus_20261123T0717Z_best-00-4.8038.ckpt`

## Bibliotecas

- gpm-api 0.4.0
- pysteps 1.21.5
- torch 2.14.1+cu130
- lightning 2.6.6
- pytorch-lightning 2.6.6
- numpy 2.4.6
- pandas 2.2.3
- xarray 2025.12.0
- scikit-image 0.25.2
- cartopy 0.26.0
- geopandas 1.2.0

## Código das células (SHA-256 do código executado)

| Célula | Linhas | SHA-256 |
|---|---:|---|
| Célula 0 | 295 | `99e1953080610bd0…` |
| Célula auxiliar (Cronus) — Diagnóstico de retomada após desconexão (só leitura) | 63 | `9d90e9db476409b7…` |
| Célula (Machine id) | 23 | `e23b39c7329a8aae…` |
| Célula (Security info) | 38 | `95d1ef372ae138e1…` |
| Célula 1 | 182 | `570545f72b6898c1…` |
| Célula auxiliar (Cronus - Somente Leitura) — Cópia dos caches .npz da V1 | 48 | `83b3619627af8409…` |
| Célula (Limpeza de Saídas Obsoletas) | 145 | `582e4aff7db521d6…` |
| Célula 2 | 426 | `395d509c894286b8…` |
| Célula 3 | 373 | `fa150491f5dbb360…` |
| Célula 4 | 401 | `ee961e3a1ddeb225…` |
| Célula 4 (conferência rápida) | 6 | `e87d93247e5d9a6b…` |
| Célula 4.1 | 241 | `284d91c2d62b0e96…` |
| Célula 4.2 | 152 | `2788e9d1cb292486…` |
| Célula 4.3 (Opcional - Fine-Tuning Suave - Adds Another Dimension of Search) | 346 | `90db637d3177bb69…` |
| Célula 4.4 | 263 | `05a2cf599175dfe0…` |
| Célula 4.5 | 340 | `e7950c6d8651d1bd…` |
| Célula 4.6 (Cronus - Fine-tuning VQGAN) | 259 | `d0b0808012b4acb2…` |
| Célula 4.7 (Cronus - Fine-tuning Transformer) | 123 | `1ce9473d3f54598f…` |
| Célula 4.8 (Cronus - V1 vs VQ - Validação 2024) | 235 | `60fb2dabaa7aa34a…` |
| Célula (Activates Checkpoint) | 151 | `e68095f8d3ea7907…` |
| Célula 5 | 174 | `88f7888458faac68…` |
| Célula 6 | 153 | `f4d33840367afd03…` |
| Célula 7 | 161 | `f5a4b268449c107d…` |
| Célula 8 | 159 | `569989cc12cc004d…` |
| Célula 9 | 216 | `ea5a5255f8862609…` |
| Célula 10 | 158 | `1a597bd8bbdc39f6…` |
| Célula 11 | 144 | `2846484678410dc1…` |
| Célula auxiliar (Cronus) — Cópia somente-leitura das janelas observadas de 2026 da V1 (sem as previsões) | 28 | `b1e5df6b417f17bd…` |
| Célula 12 | 465 | `ac2911693f8934c1…` |
| Célula Estatísticas (Célula 12.1) | 152 | `68d64b917aae5ad2…` |
| Célula 13 | 150 | `8722ee31b33407e9…` |
| Célula 13.1 | 595 | `9aad0edd13b12a10…` |
| Célula Métricas (Célula 13.2) | 259 | `1f7928abcd4f413c…` |
| Célula 13.3 | 249 | `ecde5bd5ebea23c0…` |
| Célula 13.4 | 291 | `9c0c913d121c9bec…` |
| Célula 13.5 | 142 | `2a0989088119d29d…` |
| Célula 13.6 (Cronus) | 64 | `174b5102af7eafb9…` |
| Célula 13.7 (Cronus) | 299 | `0bdbd11aaf21285a…` |
| Célula 13.7.1 (Cronus) | 379 | `6714b15f62fe1ab4…` |
| Célula 13.7.2 (Cronus) | 324 | `8424ac410f6f8c2c…` |
| Célula Extra 1 | 268 | `680ce0aa797a2f77…` |
| Célula auxiliar (Cronus - Retomada do Fechamento, Somente Leitura) | 63 | `a74da50440a9a5d0…` |
| Célula Full UMLs | 397 | `5594e3bc5494d5e8…` |
| Célula Extra 2 | 284 | `2377423ce28680a3…` |
| Célula Base (Pseudo-vista) | 364 | `02e17bd28ebb7e56…` |
| Célula (Deploy Automatizado) | 324 | `d8307b2cff0a8e2f…` |
| Célula Backup | 190 | `5e8b9e9349072fc0…` |
| Célula (Encerramento do Ambiente Cronus) | 32 | `6f60ae115c3f6877…` |

## Lista completa

Todos os 554 arquivos e hashes estão em `SHA256SUMS_20261124_085749.txt` e `manifesto_proveniencia_NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition_20261124_085749.json`.
Para conferir: `sha256sum -c SHA256SUMS_20261124_085749.txt` (rodado na pasta que contém resultados/, checkpoints/ e series_2km/).
