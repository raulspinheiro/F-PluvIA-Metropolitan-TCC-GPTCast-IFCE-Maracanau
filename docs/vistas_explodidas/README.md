# 🧩 Vistas Explodidas do F-PluvIA

Diagramas em vista explodida do **GPTCast-8x8** (Franch et al., 2025) como adaptado neste trabalho: cada etapa do modelo é desenhada como fatias separadas de um bloco, da entrada IMERG à verificação. Servem de apoio visual às Seções 2 e 3 do artigo e ao [README da raiz](../../README.md).

As dimensões e os números vêm do código e dos resultados deste repositório e foram conferidos com o *checkpoint* oficial do GPTCast. Os poucos elementos apenas ilustrativos estão indicados na própria imagem e na tabela abaixo.

## Convenção de cores

| Cor | Significado |
| :--- | :--- |
| Cinza | Dados, etapas de pré-processamento e componentes congelados |
| Verde | Codificador da VQGAN e quantização |
| Azul com linha amarela (1/3 da espessura) | Transformer GPT-2 e escolha do *token* |
| Vermelho | Decodificador da VQGAN |
| Preto com linha vermelha (1/4 da espessura) | CNN auxiliar da perda (LPIPS · VGG16, congelada) |

## Ordem de leitura

| # | Imagem | Conteúdo | Observação |
| :---: | :--- | :--- | :--- |
| 1 | [`vista_explodida_figura2_fluxo.png`](vista_explodida_figura2_fluxo.png) | Fluxo completo da previsão e variantes G, C8 e VQ (versão explodida da Figura 2 do artigo) | Visão geral; as demais detalham cada bloco |
| 2 | [`vista_explodida_entrada_imerg.png`](vista_explodida_entrada_imerg.png) | IMERG → grade de 2 km → Marshall–Palmer → dBZ → [−1, 1] → janelas de 11 quadros → divisão por ano → pilha de 8 quadros | A 1ª fatia ilustra a grade nativa (~11 km) |
| 3 | [`vista_explodida_encoder_VQGAN.png`](vista_explodida_encoder_VQGAN.png) | Codificador: 1 canal · 128×128 → 8 canais · 8×8, com o caso convectivo de 10/09/2026 na entrada | 5 níveis, 2 ResnetBlocks por nível, atenção a 8×8 |
| 4 | [`vista_explodida_quantizacao_tokens.png`](vista_explodida_quantizacao_tokens.png) | 64 vetores comparados aos 1.024 do *codebook* por distância euclidiana (argmin) e tokenização empilhada × quadro a quadro | Ranking e porcentagens são ilustrativos |
| 5 | [`vista_explodida_transformer_GPTCast.png`](vista_explodida_transformer_GPTCast.png) | Transformer GPT-2: embedding, 24 blocos (1.024 de largura, 16 cabeças, MLP de 4.096), ln_f e LM head | 304,93 M parâmetros |
| 6 | [`vista_explodida_decoder_VQGAN.png`](vista_explodida_decoder_VQGAN.png) | Decodificador: 8 canais · 8×8 → 1 canal · 128×128, com a inferência de +30 min do F-PluvIA-VQ na saída | 3 ResnetBlocks por nível; *upsample* nearest ×2 + conv 3×3 |
| 7 | [`vista_explodida_finetuning_vqgan.png`](vista_explodida_finetuning_vqgan.png) | Ajuste fino da VQGAN com o *codebook* fixo: MWAE + LPIPS + MWAE por quadro isolado + termo de quantização, sem discriminador | Célula 4.6 |
| 8 | [`vista_explodida_finetuning_transformer.png`](vista_explodida_finetuning_transformer.png) | Ajuste fino do Transformer sobre os *tokens* da VQGAN congelada: entropia cruzada do próximo *token* | Células 4 e 4.7 |
| 9 | [`vista_explodida_verificacao.png`](vista_explodida_verificacao.png) | Verificação por pixel (CSI, BIAS, MAE, FSS, SAL) e por município (F1 nos níveis Atenção, Alerta e Severo), com *bootstrap* em blocos de dias | Contingência e barras municipais são ilustrativas |

## Notas

- O caso de estudo exibido nas imagens 1, 3, 6 e 9 é o evento convectivo de **10/09/2026** (t₀ às 10:30 UTC), escolhido entre os eventos de 2026 noticiados pela imprensa, junto com um caso mais estratiforme. Ele fica fora da quadra chuvosa avaliada nos testes (fevereiro a maio) e serve só como ilustração. O quadro previsto é o de +30 min do F-PluvIA-VQ, o horizonte de maior sobreposição com o observado.
- O **discriminador PatchGAN** da receita original da VQGAN não foi usado neste trabalho, nem na V1 nem no Cronus, e por isso não aparece nas vistas. Na configuração dos autores do GPTCast (`disc_start: 50001`), essa CNN auxiliar só entra na perda depois do passo **100.002** de treino. Esse requisito não seria alcançado na reprodução deste artigo: o ajuste da VQGAN (Célula 4.6) teve 1.243 passos por época (9.940 janelas em lotes de 8 pilhas), cerca de **12,4 mil passos** nas 10 épocas executadas e no máximo 24,9 mil no limite de 20 épocas. Chegar ao passo 100.002 exigiria cerca de 81 épocas. A VQGAN foi então ajustada sem o discriminador, só com MWAE, LPIPS e o termo de quantização.
- Esta pasta **não é regravada** pela célula de Deploy do Cronus.
