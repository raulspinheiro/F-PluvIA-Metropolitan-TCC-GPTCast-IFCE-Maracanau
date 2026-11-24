# @title
# Célula 13.7.2 (Cronus): Volume de Tokens do Transformer por Etapa e por Modelo no ambiente Cronus — F-PluvIA-G (V1)
# × F-PluvIA-VQ × Zero-Shot, com e sem KV cache. Versão Cronus da Célula 13.3: conta só o que de fato rodou nesta
# execução, lendo os registros das duas pastas de resultados (V1 e VQ). Não carrega modelos nem usa a GPU, e não
# altera nenhum arquivo existente: grava só as duas tabelas e a figura novas na pasta do duelo.
#
# "Token" é um índice do codebook da VQGAN (bloco de 16 x 16 pixels, ~32 km): é o volume de trabalho do Transformer.
#   - treino/validação: cada amostra tem 8 quadros x 64 tokens = 512 tokens, lidos de uma vez;
#   - previsão sem cache: cada quadro futuro tem 64 tokens gerados um a um, e cada um relê o contexto inteiro
#     (SOS + 7 quadros = 449 posições no 1º token): 30.752 posições por quadro;
#   - previsão com KV cache (Células 13.7 e 13.7.1): o contexto passa uma vez e cada token novo processa 1 posição:
#     449 + 63 = 512 posições por quadro.
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if 'CRONUS' not in globals() or 'vq' not in CRONUS:
  raise RuntimeError('Rode antes a Célula 13.6 (ou, depois de uma desconexão, a célula auxiliar de retomada do '
                     'fechamento): a célula precisa das pastas de resultados da V1 e do F-PluvIA-VQ.')
TAG_V1, TAG_VQ = CRONUS['v1']['env_tag'], CRONUS['vq']['env_tag']
DIR_V1, DIR_VQ = CRONUS['v1']['results_dir'], CRONUS['vq']['results_dir']
ROTULO = globals().get('ENV_LABEL', TAG_V1)

N_CTX, N_FUT, TOK_QUADRO, BLOCK = 7, 4, 64, 512
CTX_INI = 1 + N_CTX * TOK_QUADRO                              # 449
POS_SEM_CACHE = sum(CTX_INI + k for k in range(TOK_QUADRO))   # 30.752 posições por quadro
POS_KV = CTX_INI + (TOK_QUADRO - 1)                           # 512 posições por quadro
TOK_JANELA = N_FUT * TOK_QUADRO                               # 256 tokens gerados por janela
N_MEMBROS = 8
fontes, avisos = {}, []


def achar(pasta, padrao):
  """Arquivo mais recente que casa com o padrão, em qualquer subpasta (ou None)."""
  a = sorted(glob.glob(os.path.join(pasta, '**', padrao), recursive=True), key=os.path.getmtime)
  return a[-1] if a else None


def registrar(chave, caminho):
  fontes[chave] = os.path.relpath(caminho, os.path.dirname(DIR_V1))


# ------------------------------------------------------------- 1. partições (da V1 do Cronus)
f = achar(DIR_V1, f'resumo_particoes_{TAG_V1}_*.csv')
if not f:
  raise RuntimeError(f'resumo_particoes_{TAG_V1}_*.csv não encontrado em {DIR_V1}.')
d = pd.read_csv(f).set_index('Particao')['Sequencias']
N_TREINO, N_VAL, N_TESTE25 = int(d['Treino']), int(d['Validação']), int(d['Teste'])
registrar('partições', f)
f = achar(DIR_V1, f'log_tempos_treino_e_inferencia_{TAG_V1}_*.json')
reg_v1 = json.load(open(f, encoding='utf-8')) if f else {}
if f:
  registrar('registro do treino da V1', f)
LOTE = int(reg_v1.get('batch_size_treino', 16))
PASSOS_EPOCA = -(-N_TREINO // LOTE)

linhas = []


def add(etapa, modelo, variante, unidade, n, tok_unidade, pos_por_token=1.0, obs=''):
  """pos_por_token: posições processadas por token (1 no treino; 480,5 na previsão sem cache; 8 com KV cache)."""
  if n <= 0:
    return
  tipo = 'lidos (treino/validação)' if pos_por_token == 1.0 else 'gerados (previsão)'
  linhas.append({'Etapa': etapa, 'Modelo': modelo, 'Variante': variante, 'Unidade': unidade, 'Quantidade': int(n),
                 'Tokens por unidade': int(tok_unidade), 'Tokens': int(n * tok_unidade),
                 'Posições processadas pelo Transformer': int(round(n * tok_unidade * pos_por_token)),
                 'Tipo': tipo, 'Observação': obs})


SEM_CACHE, COM_KV = POS_SEM_CACHE / TOK_QUADRO, POS_KV / TOK_QUADRO


def curva(metrics_csv):
  """Amostras de treino vistas, validações completas e passos, do metrics.csv do Lightning."""
  m = pd.read_csv(metrics_csv)
  passos = int(m['step'].max()) + 1
  ep, resto = divmod(passos, PASSOS_EPOCA)
  return ep * N_TREINO + min(resto * LOTE, N_TREINO), int(m['val/loss_epoch'].notna().sum()), passos


# ------------------------------------------------------------- 2. treino e validação
treinos = [('Ajuste fino da V1 (Célula 4)', 'F-PluvIA-G (V1)', f'curva_treino_{TAG_V1}', 'validação a cada época'),
           ('Ajuste fino suave (Célula 4.3)', 'F-PluvIA suave (lr 1e-5)', f'curva_treino_alternativo_{TAG_V1}',
            'validação a cada 1/4 de época'),
           ('Ajuste fino do VQ (Célula 4.7)', 'F-PluvIA-VQ', f'curva_treino_fpluvia_vq_{TAG_V1}', 'validação a cada época')]
for etapa, modelo, pasta, rot_val in treinos:
  achados = sorted(glob.glob(os.path.join(DIR_V1, '**', pasta, '*', 'metrics.csv'), recursive=True))
  for p in achados:
    a, v, ps = curva(p)
    rodada = os.path.basename(os.path.dirname(p))
    add(etapa, modelo, 'treino', 'amostra de 8 quadros', a, BLOCK,
        obs=f'rodada {rodada[:14]}: {ps} passos, {a / N_TREINO:.2f} épocas de {N_TREINO} janelas'
            + (' (interrompida)' if 'alternativo' in pasta else ''))
    add(etapa, modelo, rot_val, 'amostra de 8 quadros', v * N_VAL, BLOCK, obs=f'{v} validações de {N_VAL} janelas de 2024')
    registrar(f'curva {pasta}/{rodada}', p)
  if not achados and 'alternativo' not in pasta:
    avisos.append(f'curva de treino "{pasta}" não encontrada: a etapa "{etapa}" ficou de fora.')

f = achar(DIR_V1, f'validacao_resumo_finetuned_vs_zeroshot_{TAG_V1}.csv')
if f:
  for modelo in ('F-PluvIA-G (V1)', 'Zero-Shot'):
    add('Validação Fine-Tuned × Zero-Shot (Célula 4.1)', modelo, 'perda nos tokens', 'amostra de 8 quadros', N_VAL,
        BLOCK, obs='uma passada sobre a validação de 2024')
  registrar('Célula 4.1', f)

# ------------------------------------------------------------- 3. previsões (tokens gerados)
f = achar(DIR_V1, f'sensibilidade_amostragem_validacao2024_{TAG_V1}.csv')
if f:
  N_45 = 240                                              # amostra fixa da Célula 4.5 (semente fixa)
  s = pd.read_csv(f)
  s = s[s['Modelo'].isin(['Fine-Tuned', 'Zero-Shot'])]
  for _, r in s.iterrows():
    membros = N_MEMBROS if str(r['Amostragem']).startswith('Média') else 1
    add('Escolha da decodificação (Célula 4.5, validação 2024)',
        'F-PluvIA-G (V1)' if r['Modelo'] == 'Fine-Tuned' else 'Zero-Shot', r['Amostragem'], 'janela (4 quadros)',
        N_45 * membros, TOK_JANELA, SEM_CACHE, obs=f'{N_45} janelas' + (f' x {membros} membros' if membros > 1 else ''))
  registrar('Célula 4.5', f)

f = achar(DIR_V1, f'cronus_previsao_v1_vs_vq_validacao2024_{TAG_V1}.json')
if f:
  n48 = int(json.load(open(f, encoding='utf-8')).get('n_janelas', 240))
  for modelo in ('F-PluvIA-G (V1)', 'F-PluvIA-VQ'):
    add('Validação V1 × VQ (Célula 4.8, 2024)', modelo, 'gulosa', 'janela (4 quadros)', n48, TOK_JANELA, SEM_CACHE,
        obs=f'{n48} janelas sorteadas de 2024')
  registrar('Célula 4.8', f)


def velocidade(pasta, tag):
  f = achar(pasta, f'velocidade_inferencia_teste2025_{tag}_N*.csv')
  if not f:
    return None, None
  return pd.read_csv(f).set_index('Modelo'), f


vel_v1, f1 = velocidade(DIR_V1, TAG_V1)
vel_vq, f2 = velocidade(DIR_VQ, TAG_VQ)
FT, ZS = 'GPTCast-8x8 (Fine-Tuned RMF)', 'GPTCast-8x8 (Zero-Shot)'
if vel_v1 is not None:
  registrar('tempos do teste 2025 (V1)', f1)
  add('Teste 2025 (Célula 12)', 'F-PluvIA-G (V1)', 'gulosa', 'janela (4 quadros)',
      int(vel_v1.loc[FT, 'Amostras_Teste_2025']), TOK_JANELA, SEM_CACHE, obs='decodificação escolhida em 2024')
  add('Teste 2025 (Célula 12)', 'Zero-Shot', 'top-k 10', 'janela (4 quadros)',
      int(vel_v1.loc[ZS, 'Amostras_Teste_2025']), TOK_JANELA, SEM_CACHE, obs='decodificação escolhida em 2024')
else:
  avisos.append('tempos do teste 2025 da V1 não encontrados: a Célula 12 da V1 ficou de fora.')
if vel_vq is not None:
  registrar('tempos do teste 2025 (VQ)', f2)
  add('Teste 2025 (Célula 12)', 'F-PluvIA-VQ', 'gulosa', 'janela (4 quadros)',
      int(vel_vq.loc[FT, 'Amostras_Teste_2025']), TOK_JANELA, SEM_CACHE, obs='decodificação escolhida em 2024')
  # O Zero-Shot é reaproveitado da V1 pela Célula 13.6: só conta se o tempo total mostrar que foi gerado de novo
  if vel_v1 is not None and ZS in vel_vq.index and \
     abs(float(vel_vq.loc[ZS, 'Tempo_Total_Lote_s']) - float(vel_v1.loc[ZS, 'Tempo_Total_Lote_s'])) > 1e-6:
    add('Teste 2025 (Célula 12)', 'Zero-Shot', 'top-k 10 (execução VQ)', 'janela (4 quadros)',
        int(vel_vq.loc[ZS, 'Amostras_Teste_2025']), TOK_JANELA, SEM_CACHE, obs='gerado de novo na execução VQ')
else:
  avisos.append('tempos do teste 2025 do VQ não encontrados: a Célula 12 do VQ ficou de fora.')


def teste_2026(pasta, tag):
  f = achar(pasta, f'teste_cego_2026_meses_e_produtos_{tag}*.csv')
  if not f:
    return None
  g = achar(pasta, f'teste_cego_2026_estudos_de_caso_{tag}*.csv')
  casos = int(pd.read_csv(g)['Caso'].nunique()) if g else 0
  return int(pd.read_csv(f)['Janelas'].sum()), casos, f


for pasta, tag, modelos in ((DIR_V1, TAG_V1, (('F-PluvIA-G (V1)', 'gulosa'), ('Zero-Shot', 'top-k 10'))),
                            (DIR_VQ, TAG_VQ, (('F-PluvIA-VQ', 'gulosa'),))):
  r = teste_2026(pasta, tag)
  if r is None:
    avisos.append(f'teste cego de 2026 não encontrado em {os.path.basename(pasta)}.')
    continue
  n26, casos, f = r
  registrar(f'teste cego 2026 ({os.path.basename(pasta)})', f)
  for modelo, var in modelos:
    add('Teste cego 2026 (Célula 13.1)', modelo, var, 'janela (4 quadros)', n26 + casos, TOK_JANELA, SEM_CACHE,
        obs=f'{n26} janelas da série + {casos} estudos de caso')

f = achar(DIR_VQ, f'cronus_kvcache_resumo_{TAG_VQ}.csv')
if f:
  for _, r in pd.read_csv(f).iterrows():
    add('Duelo: variante com KV cache (Célula 13.7)', r['Modelo'], f'gulosa com KV cache, {r["Conjunto"]}',
        'janela (4 quadros)', int(r['Janelas']), TOK_JANELA, COM_KV,
        obs=f'{int(r["Janelas"])} janelas de {r["Conjunto"]}; a previsão sem cache é a das Células 12 e 13.1')
  registrar('Célula 13.7 (KV cache)', f)
  kv = pd.read_csv(f)
else:
  kv = None
  avisos.append('resumo do KV cache da Célula 13.7 não encontrado.')

f = achar(DIR_VQ, f'tabela_complexidade_bigO_v1_vs_vq_{TAG_VQ}.csv')
if f:
  c = pd.read_csv(f)
  for modelo, g in c.groupby('Modelo', sort=False):
    n_q = int((g['Passos (S)'] * g['Instâncias (m)']).sum())          # quadros previstos em cada versão
    add('Análise de complexidade (Célula 13.7.1)', modelo, 'recálculo (sem cache)', 'quadro', n_q, TOK_QUADRO,
        SEM_CACHE, obs=f'{len(g)} tamanhos x {int(g["Instâncias (m)"].iloc[0])} instâncias, medições cronometradas')
    add('Análise de complexidade (Célula 13.7.1)', modelo, 'KV cache', 'quadro', n_q, TOK_QUADRO, COM_KV,
        obs='as mesmas instâncias')
  registrar('Célula 13.7.1', f)

df = pd.DataFrame(linhas)
if df.empty:
  raise RuntimeError('Nenhum registro encontrado nas pastas da V1 e do VQ.')

# ------------------------------------------------------------- 4. custo de um quadro (tempos medidos)
unid = []


def t_quadro(v):
  return float(v) / N_FUT if v is not None and v == v else np.nan


if vel_v1 is not None:
  unid.append(('F-PluvIA-G (V1), sem cache', TOK_QUADRO, POS_SEM_CACHE, t_quadro(vel_v1.loc[FT, 'Tempo_Medio_4Passos_s'])))
if vel_vq is not None:
  unid.append(('F-PluvIA-VQ, sem cache', TOK_QUADRO, POS_SEM_CACHE, t_quadro(vel_vq.loc[FT, 'Tempo_Medio_4Passos_s'])))
if vel_v1 is not None:
  unid.append(('Zero-Shot top-k 10, sem cache', TOK_QUADRO, POS_SEM_CACHE, t_quadro(vel_v1.loc[ZS, 'Tempo_Medio_4Passos_s'])))
if kv is not None:
  for modelo, g in kv.groupby('Modelo', sort=False):
    unid.append((f'{modelo}, com KV cache', TOK_QUADRO, POS_KV, t_quadro(g['Tempo médio com KV cache (s)'].mean())))
df_unid = pd.DataFrame([{'Previsão de 1 quadro (+30 min)': n, 'Tokens gerados': t, 'Posições processadas': p,
                         'Tempo medido por quadro [s]': round(s, 3),
                         'Tokens gerados por segundo': round(t / s, 1) if s == s else np.nan} for n, t, p, s in unid])

T_AJUSTE_V1 = float(reg_v1.get('finetuning_tempo_total_s', np.nan))
tok_ajuste_v1 = df[df['Etapa'] == 'Ajuste fino da V1 (Célula 4)']['Tokens'].sum()
vazao = tok_ajuste_v1 / T_AJUSTE_V1 if T_AJUSTE_V1 == T_AJUSTE_V1 and T_AJUSTE_V1 > 0 else np.nan

# ------------------------------------------------------------- 5. saídas
fmt = lambda x: f'{x:,.0f}'.replace(',', '.')


def bonito(d, cols):
  d = d.copy()
  for c in cols:
    d[c] = d[c].map(fmt)
  return d


print(f'=== Volume de tokens do Transformer no ambiente Cronus | {ROTULO} ===')
print(f'{TOK_QUADRO} tokens por quadro | block_size {BLOCK} | contexto de {N_CTX} quadros | {N_FUT} quadros previstos')
print(f'Sem cache: {fmt(POS_SEM_CACHE)} posições por quadro ({SEM_CACHE:.1f} por token gerado). '
      f'Com KV cache: {POS_KV} posições por quadro ({COM_KV:.0f} por token), {POS_SEM_CACHE / POS_KV:.0f}x menos.')
for a in avisos:
  print(f'[Aviso] {a}')

print('\n--- Previsão de um único quadro (tempos medidos) ---')
display(bonito(df_unid, ['Tokens gerados', 'Posições processadas']).set_index('Previsão de 1 quadro (+30 min)'))
if vazao == vazao:
  print(f'No ajuste fino da V1, a vazão foi de ~{fmt(vazao)} tokens lidos por segundo; na previsão, ~'
        f'{df_unid["Tokens gerados por segundo"].iloc[0]:.0f} tokens gerados por segundo.')

cols = ['Quantidade', 'Tokens por unidade', 'Tokens', 'Posições processadas pelo Transformer', 'Observação']
print('\n--- Por etapa e variante ---')
with pd.option_context('display.max_rows', None, 'display.max_colwidth', 80):
  display(bonito(df, cols[:4]).set_index(['Etapa', 'Modelo', 'Variante'])[cols])

resumo = df.pivot_table(index='Modelo', columns='Tipo', values='Tokens', aggfunc='sum', fill_value=0, sort=False)
resumo['Total de tokens'] = resumo.sum(axis=1)
resumo['Posições processadas'] = df.groupby('Modelo', sort=False)['Posições processadas pelo Transformer'].sum()
print('\n--- Por modelo (todas as etapas) ---')
display(bonito(resumo, list(resumo.columns)))
lidos = df[df['Tipo'].str.startswith('lidos')]
gerados = df[df['Tipo'].str.startswith('gerados')]
print(f'Total: {fmt(lidos["Tokens"].sum())} tokens lidos no treino e na validação e {fmt(gerados["Tokens"].sum())} '
      f'tokens gerados nas previsões ({fmt(df["Posições processadas pelo Transformer"].sum())} posições processadas).')
kvp = gerados[gerados['Variante'].str.contains('KV cache')]
if len(kvp):
  ref = kvp['Tokens'].sum() * SEM_CACHE
  print(f'As previsões com KV cache geraram {fmt(kvp["Tokens"].sum())} tokens processando '
        f'{fmt(kvp["Posições processadas pelo Transformer"].sum())} posições; sem o cache seriam {fmt(ref)}.')
print('Fora da conta: o ajuste da VQGAN (Célula 4.6, que não passa pelo Transformer), os estudos de caso de '
      '18/03/2024 e da Célula 9, os aquecimentos e a verificação em fp32 da Célula 13.7.1 e a checagem de sanidade do '
      'Lightning. O Zero-Shot, o LINDA, o Lucas-Kanade e a persistência da execução VQ foram reaproveitados da V1.')
print(f'Fontes: {json.dumps(fontes, ensure_ascii=False, indent=1)}')

p_det = os.path.join(DIR_VQ, f'tokens_por_etapa_e_variante_cronus_{TAG_VQ}.csv')
p_uni = os.path.join(DIR_VQ, f'tokens_por_quadro_cronus_{TAG_VQ}.csv')
df.to_csv(p_det, index=False)
df_unid.to_csv(p_uni, index=False)

# figura: tokens (esquerda) e posições processadas (direita) por etapa, escala log, cores por modelo
cores = {'F-PluvIA-G (V1)': '#E69F00', 'F-PluvIA-VQ': '#56B4E9', 'F-PluvIA suave (lr 1e-5)': '#F0C873',
         'Zero-Shot': '#999999'}
g = df.groupby(['Etapa', 'Modelo', 'Variante'], sort=False)[['Tokens', 'Posições processadas pelo Transformer']] \
      .sum().reset_index()
g['rótulo'] = [f"{e.split(' (')[0]} | {m}, {v}" for e, m, v in zip(g['Etapa'], g['Modelo'], g['Variante'])]
fig, axs = plt.subplots(1, 2, figsize=(16, 0.3 * len(g) + 1.8), sharey=True, constrained_layout=True)
y = np.arange(len(g))[::-1]
for ax, col, titulo in ((axs[0], 'Tokens', 'Tokens (lidos no treino e validação, gerados na previsão)'),
                        (axs[1], 'Posições processadas pelo Transformer', 'Posições processadas pelo Transformer')):
  barras = ax.barh(y, g[col], color=[cores.get(m, '#7f7f7f') for m in g['Modelo']], edgecolor='white', linewidth=0.5)
  for b, yi, t, v in zip(barras, y, g[col], g['Variante']):
    ax.text(t * 1.1, yi, fmt(t), va='center', fontsize=7)
    if 'KV cache' in v:
      b.set_hatch('///')
  ax.set_xscale('log')
  ax.set_xlim(left=max(1, g[col].min() / 3), right=g[col].max() * 8)
  ax.set_xlabel(f'{titulo} (escala logarítmica)', fontsize=9)
  ax.grid(True, axis='x', ls=':', alpha=0.6)
axs[0].set_yticks(y)
axs[0].set_yticklabels(g['rótulo'], fontsize=7.5)
alcas = [plt.Rectangle((0, 0), 1, 1, color=c) for m, c in cores.items() if m in set(g['Modelo'])]
nomes = [m for m in cores if m in set(g['Modelo'])]
alcas.append(plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='#555555', hatch='///'))
nomes.append('com KV cache')
axs[1].legend(alcas, nomes, loc='upper right', fontsize=8)
fig.suptitle(f'Volume de tokens do Transformer do GPTCast-8x8 no ambiente Cronus (V1 × VQ) | {ROTULO}',
             fontweight='bold')
p_fig = os.path.join(DIR_VQ, f'painel_tokens_por_etapa_cronus_{TAG_VQ}.png')
fig.savefig(p_fig, dpi=300, bbox_inches='tight')
plt.show()

CELL13_7_2_TOKENS = {'por_etapa': df, 'por_quadro': df_unid, 'por_modelo': resumo}
print(f'[Célula 13.7.2] Tabelas -> {p_det} | {p_uni}\n[Célula 13.7.2] Figura -> {p_fig}')
