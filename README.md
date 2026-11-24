# F-PluvIA Metropolitan

Nowcasting de precipitação na Região Metropolitana de Fortaleza com o GPTCast-8x8 ajustado em séries GPM IMERG reamostradas a 2 km.

Trabalho de Conclusão de Curso — Bacharelado em Ciência da Computação, IFCE Campus Maracanaú (2026.2).  
Autor: Raul Santiago Pinheiro · Orientador: Amauri Holanda de Souza Júnior

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `notebooks/` | Notebook do Google Colab com todas as células |
| `src/celulas/` | Código de cada célula, um arquivo por célula |
| `docs/uml/` | Diagramas UML de cada célula e do fluxo entre células |
| `results/<ambiente>/tables/` | Tabelas de métricas (.csv) |
| `results/<ambiente>/figures/` | Figuras e mapas (.png) |
| `docs/provenance/` | Manifesto SHA-256 dos artefatos (conferir com `sha256sum -c`) |

Os checkpoints do modelo e as séries .npz não estão no repositório (tamanho); ficam no Google Drive do autor.
