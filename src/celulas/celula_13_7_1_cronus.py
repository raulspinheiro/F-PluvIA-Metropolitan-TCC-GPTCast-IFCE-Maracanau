# @title
# Célula 13.7.1 (Cronus): Análise de Complexidade (Big O) dos dois modelos do duelo da Célula 13.7 — F-PluvIA-G (V1,
# VQGAN original) e F-PluvIA-VQ (VQGAN ajustada à RMF) — como estudo de caso comparativo entre duas implementações
# equivalentes do gerador guloso: recálculo integral (sem cache) × memoização das chaves e valores da atenção (KV cache).
# Metodologia do estudo de caso de Programação Dinâmica (Edit Distance, top-down com memoização × bottom-up) da disciplina
# ANALG 2051.1 IFCE Maracanaú Profº Thiago Alves: semente 42, k tamanhos sorteados e igualmente espaçados, m instâncias
# sorteadas e IGUAIS para os dois modelos e as duas versões, tempo médio por instância com perf_counter e verificação
# de correção com assert. Complementa a Célula 13.4 (que analisa só o F-PluvIA-G).
import math
import os
import random
import statistics
import time

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------- 1. parâmetros do experimento
SEED = 42                           # reprodutibilidade, como no estudo de caso
MIN_S, MAX_S = 1, 12                # intervalo dos tamanhos de entrada: passos de 30 min (+30 a +360 min)
K_RANGE = (6, 10)                   # nº de tamanhos distintos (sorteado)
M_RANGE = (10, 20)                  # nº de instâncias por tamanho (sorteado, o mesmo para todos os tamanhos)
CONTEXTO = 7                        # quadros observados, como nas Células 12 e 13.7
AQUECIMENTO = 2                     # execuções descartadas antes de medir (compilação de kernels na GPU)
TOL_EMPATE = 1e-3                   # diferença entre os 2 maiores logits abaixo da qual há empate numérico

faltam = [n for n in ('CRONUS', 'gpt8_v1', 'gpt8_vq', 'md_rmf') if n not in globals()]
if faltam or 'vq' not in globals().get('CRONUS', {}):
  raise RuntimeError('Rode antes a Célula 13.6 e a 13.7 (duelo) nesta sessão. '
                     f'Faltam: {", ".join(faltam) or "a execução VQ em CRONUS"}')
ENV_LABEL = globals().get('ENV_LABEL', 'Hardware Ativo')
SAIDA = CRONUS['vq']['results_dir']                       # mesma pasta do duelo (nada da V1 é sobrescrito)
TAG = CRONUS['vq']['env_tag']
os.makedirs(SAIDA, exist_ok=True)
MODELOS = {'F-PluvIA-G (V1)': gpt8_v1, 'F-PluvIA-VQ': gpt8_vq}
for _m in MODELOS.values():
  _m.eval()
DEVICE = next(gpt8_v1.parameters()).device
if DEVICE.type != 'cuda':
  print('[aviso] Os modelos estão na CPU: os tempos não representam a execução na GPU usada no artigo.')

# Mesma arquitetura nos dois modelos: a complexidade teórica é idêntica; só os pesos mudam.
_cfg = {nome: (int(m.transformer.config.n_layer), int(m.transformer.config.n_embd), int(m.transformer.config.block_size),
               sum(p.numel() for p in m.transformer.parameters()),
               sum(p.numel() for p in m.first_stage_model.parameters())) for nome, m in MODELOS.items()}
assert len(set(_cfg.values())) == 1, f'Os dois modelos deveriam ter a mesma arquitetura: {_cfg}'
L_CAM, D_LARG, BLOCK, P_TRANSFORMER, P_VQGAN = next(iter(_cfg.values()))
WS = int(math.sqrt(BLOCK // 8))                           # janela espacial em tokens (8 → 8×8 = 64 por quadro)
T_QUADRO = WS * WS
seqs = md_rmf.splits['test'].sequences_dbz                # teste de 2025, o conjunto que decide o duelo


def br(n) -> str:
  return f'{int(n):,}'.replace(',', '.')


def cronometro():
  if DEVICE.type == 'cuda':
    torch.cuda.synchronize()
  return time.perf_counter()


# ------------------------------------------------------------------- 2. as duas implementações (para qualquer modelo)
def _proximo_token(M, logits):
  """Mesma regra de predict_next_index com temperatura 1 e top-k 1 (decodificação gulosa)."""
  lg = M.top_k_logits(logits[:, -1, :] / 1.0, 1)
  _, res = torch.topk(F.softmax(lg, dim=-1), k=1, dim=-1)
  return res


def quadro_recalculo(M, ind_ctx, c_idx):
  """Versão 1, sem cache: o gerador original do pacote. Cada um dos T tokens do quadro
  reprocessa o contexto inteiro (condicionamento + quadros passados + tokens já gerados)."""
  return M.predict_next_frame_indices(ind_ctx, c_idx, WS, temperature=1.0, top_k=1, show_progress=False)


def quadro_kv_cache(M, ind_ctx, c_idx):
  """Versão 2, com memoização: o contexto passa uma única vez pelo Transformer e as chaves e valores
  de cada camada ficam guardados (KV cache); cada token novo processa só 1 posição."""
  TR = M.transformer
  b, _, h, w = ind_ctx.shape
  if (h, w) != (WS, WS):
    raise RuntimeError(f'O KV cache exige que a janela ({WS}×{WS}) cubra o quadro inteiro; quadro de {h}×{w} tokens.')
  prefixo = torch.cat((c_idx.reshape(b, -1), ind_ctx.reshape(b, -1)), dim=1)
  logits, _, cache = TR.forward_with_past(prefixo)
  n = prefixo.shape[1]
  tok = _proximo_token(M, logits)
  tokens = [tok]
  for _ in range(h * w - 1):
    logits, _, novo = TR.forward_with_past(tok, past=[cache], past_length=n)
    cache = torch.cat((cache, novo), dim=-2)                # o pacote devolve só k e v da posição nova
    n += 1
    tok = _proximo_token(M, logits)
    tokens.append(tok)
  return torch.cat(tokens, dim=1).reshape(b, h, w)


VERSOES = {'Recálculo (sem cache)': quadro_recalculo, 'KV cache (memoização)': quadro_kv_cache}


@torch.inference_mode()
def codificar(M, ctx_dbz):
  x = torch.tensor(np.clip(np.nan_to_num(ctx_dbz, nan=0.0), 0, 60) / 30.0 - 1.0, dtype=torch.float32, device=DEVICE)
  s = x.shape[0]
  xs = einops.rearrange(x, 's h w -> (s h) w')[None, None].contiguous()
  quant, idx = M.encode_to_z(xs)                          # cada modelo usa a SUA VQGAN (original ou ajustada)
  _, c_idx = M.encode_to_c(xs)
  qs = quant.shape
  return idx.reshape(qs[0], s, qs[2] // s, qs[3]), c_idx, qs


@torch.inference_mode()
def prever(M, ctx_dbz, passos: int, gerador):
  """Previsão completa de S passos (a mesma das Células 12 e 13.7): codifica, gera quadro a quadro com a janela
  deslizante dos últimos CONTEXTO quadros e decodifica cada quadro. Só o gerador de tokens muda."""
  ind, c_idx, qs = codificar(M, ctx_dbz)
  s, ih, iw = ind.shape[1], ind.shape[2], ind.shape[3]
  quadros = []
  for _ in range(passos):
    with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
      p = gerador(M, ind[:, -s:], c_idx)
    M.decode_to_img(p.reshape(1, -1), torch.Size([1, qs[1], ih, iw]))
    ind = torch.cat((ind, p[:, None]), dim=1)
    quadros.append(p)
  return torch.stack(quadros, dim=1)


@torch.inference_mode()
def verificar_token_a_token(M, ctx_dbz):
  """Verificação de correção em fp32, sem divergência em cascata: em cada um dos T tokens do quadro, as duas
  versões recebem o MESMO contexto e devem escolher o mesmo token (logits iguais a menos de arredondamento)."""
  TR = M.transformer
  ind, c_idx, _ = codificar(M, ctx_dbz)
  ind_ctx = ind[:, -CONTEXTO:]
  b = ind_ctx.shape[0]
  prefixo = torch.cat((c_idx.reshape(b, -1), ind_ctx.reshape(b, -1)), dim=1)
  logits_kv, _, cache = TR.forward_with_past(prefixo)
  n, gerados, dif_max, divergentes, empates = prefixo.shape[1], [], 0.0, 0, 0
  for t in range(T_QUADRO):
    contexto = torch.cat([prefixo] + gerados, dim=1)
    l_rc = TR(contexto)[0][:, -1, :].float()
    l_kv = logits_kv[:, -1, :].float()
    dif_max = max(dif_max, float((l_rc - l_kv).abs().max()))
    if int(l_rc.argmax(-1)) != int(l_kv.argmax(-1)):
      divergentes += 1
      top2 = l_rc.topk(2, dim=-1).values[0]
      empates += int(float(top2[0] - top2[1]) < TOL_EMPATE)
    tok = _proximo_token(M, logits_kv)
    gerados.append(tok)
    if t < T_QUADRO - 1:
      logits_kv, _, novo = TR.forward_with_past(tok, past=[cache], past_length=n)
      cache = torch.cat((cache, novo), dim=-2)
      n += 1
  return dif_max, divergentes, empates


# ------------------------------------------------------------------- 3. sorteio de k, m e dos tamanhos
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
k = random.randint(*K_RANGE)                              # sorteia k
m = random.randint(*M_RANGE)                              # sorteia m (o mesmo para todos os tamanhos)
passo = (MAX_S - MIN_S) / (k - 1)                          # tamanhos igualmente espaçados
tamanhos = sorted({int(MIN_S + i * passo + 1e-9) for i in range(k)})
_amostra = np.clip(seqs[0, :CONTEXTO], 0, 60)
P0 = int(codificar(gpt8_v1, _amostra)[1].numel() + CONTEXTO * T_QUADRO)   # tokens do contexto antes do 1º token
POS_RC = T_QUADRO * P0 + T_QUADRO * (T_QUADRO - 1) // 2   # posições por quadro, recálculo: Σ (P0 + t)
POS_KV = P0 + T_QUADRO - 1                                 # posições por quadro, KV cache: P0 + (T − 1)
print(f'Arquitetura (idêntica nos dois modelos): Transformer {P_TRANSFORMER / 1e6:.2f} M de parâmetros, L = {L_CAM} '
      f'camadas, d = {D_LARG}, contexto máximo {BLOCK}, {T_QUADRO} tokens por quadro, {P0} tokens de contexto inicial | '
      f'VQGAN {P_VQGAN / 1e6:.2f} M')
print(f'Sorteio (semente {SEED}): k = {k} tamanhos {tamanhos}, m = {m} instâncias por tamanho, as MESMAS para os dois '
      f'modelos e as duas versões | posições por quadro: {br(POS_RC)} (recálculo) × {br(POS_KV)} (KV cache), '
      f'{POS_RC / POS_KV:.0f}× menos')

t0_total = time.perf_counter()

# ------------------------------------------------------------------- 4. verificação de correção (fp32, token a token)
_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
janelas_verif = random.sample(range(len(seqs)), m)
verif = {}
try:
  for nome, M in MODELOS.items():
    verif[nome] = [verificar_token_a_token(M, np.clip(seqs[jv, :CONTEXTO], 0, 60)) for jv in janelas_verif]
finally:
  torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = _tf32
for nome, v in verif.items():
  dmax, ndiv, nemp = max(x[0] for x in v), sum(x[1] for x in v), sum(x[2] for x in v)
  assert ndiv == nemp, (f'{nome}: as versões divergem fora de empate numérico em {ndiv - nemp} de {m * T_QUADRO} '
                        'tokens: o KV cache não reproduz o recálculo.')
  print(f'Verificação (fp32) {nome}: {m} instâncias × {T_QUADRO} tokens, mesmo token em {m * T_QUADRO - ndiv} de '
        f'{m * T_QUADRO} ({nemp} empates numéricos), |Δ logit| máx. = {dmax:.2e}  OK')

# ------------------------------------------------------------------- 5. tempo médio por instância × tamanho
for _ in range(AQUECIMENTO):
  for M in MODELOS.values():
    for g in VERSOES.values():
      prever(M, _amostra, 1, g)

linhas = []
for S in tamanhos:
  janelas = random.sample(range(len(seqs)), m)            # *uma única* lista de instâncias para este tamanho
  totais = {(mod, ver): 0.0 for mod in MODELOS for ver in VERSOES}
  tempos = {(mod, ver): [] for mod in MODELOS for ver in VERSOES}   # por instância (média e desvio)
  identicos = {mod: 0 for mod in MODELOS}
  for j in janelas:                                       # os dois modelos e as duas versões sobre as MESMAS instâncias
    ctx = np.clip(seqs[j, :CONTEXTO], 0, 60)
    for mod, M in MODELOS.items():
      saidas = {}
      for ver, g in VERSOES.items():
        inicio = cronometro()
        saidas[ver] = prever(M, ctx, S, g)
        dt = cronometro() - inicio
        totais[(mod, ver)] += dt
        tempos[(mod, ver)].append(dt)
      identicos[mod] += int(torch.equal(*saidas.values()))
  for mod in MODELOS:
    t_rc, t_kv = (totais[(mod, ver)] / m for ver in VERSOES)
    linhas.append({'Ambiente_Execucao': ENV_LABEL, 'Modelo': mod, 'Passos (S)': S, 'Horizonte [min]': 30 * S,
                   'Instâncias (m)': m, 'Tempo médio recálculo [s]': round(t_rc, 4),
                   'Tempo médio KV cache [s]': round(t_kv, 4),
                   'Desvio recálculo [s]': round(statistics.pstdev(tempos[(mod, 'Recálculo (sem cache)')]), 4),
                   'Desvio KV cache [s]': round(statistics.pstdev(tempos[(mod, 'KV cache (memoização)')]), 4),
                   'Speedup (recálculo / KV)': round(t_rc / t_kv, 2),
                   'Posições recálculo': S * POS_RC, 'Posições KV cache': S * POS_KV,
                   'TFLOP/s recálculo': round(2 * P_TRANSFORMER * S * POS_RC / t_rc / 1e12, 1),
                   'TFLOP/s KV cache': round(2 * P_TRANSFORMER * S * POS_KV / t_kv / 1e12, 1),
                   'Previsões idênticas (bf16)': f'{identicos[mod]}/{m}',
                   'Divergências (bf16)': m - identicos[mod]})
  print(f'S = {S:2d} ({30 * S:3d} min)  OK  ' + ' | '.join(
      f'{mod}: recálculo {totais[(mod, "Recálculo (sem cache)")] / m:.3f} s, KV {totais[(mod, "KV cache (memoização)")] / m:.3f} s'
      f' ({totais[(mod, "Recálculo (sem cache)")] / totais[(mod, "KV cache (memoização)")]:.2f}×), idênticas {identicos[mod]}/{m}'
      for mod in MODELOS))
df = pd.DataFrame(linhas)

# Ajuste t = a + b·S para cada modelo e versão (O(S) em todos: a janela desliza e o contexto nunca passa de BLOCK)
COLS = {'Recálculo (sem cache)': 'Tempo médio recálculo [s]', 'KV cache (memoização)': 'Tempo médio KV cache [s]'}
ajustes = {}
for mod in MODELOS:
  sub = df[df['Modelo'] == mod]
  S_arr = sub['Passos (S)'].to_numpy(float)
  for ver, col in COLS.items():
    y = sub[col].to_numpy(float)
    b_, a_ = np.polyfit(S_arr, y, 1)
    r2 = 1 - np.sum((y - (a_ + b_ * S_arr)) ** 2) / np.sum((y - y.mean()) ** 2)
    ajustes[(mod, ver)] = (a_, b_, r2, np.polyfit(S_arr, y, 2)[0])
csv_path = os.path.join(SAIDA, f'tabela_complexidade_bigO_v1_vs_vq_{TAG}.csv')
df.to_csv(csv_path, index=False)
verif_path = os.path.join(SAIDA, f'tabela_verificacao_kvcache_v1_vs_vq_{TAG}.csv')
pd.DataFrame([{'Ambiente_Execucao': ENV_LABEL, 'Modelo': nome, 'Janela': jv, 'Tokens': T_QUADRO, 'Max |Δ logit|': x[0],
               'Tokens divergentes': x[1], 'Empates numéricos': x[2]}
              for nome, v in verif.items() for jv, x in zip(janelas_verif, v)]).to_csv(verif_path, index=False)
ajustes_path = os.path.join(SAIDA, f'tabela_ajustes_bigO_v1_vs_vq_{TAG}.csv')
pd.DataFrame([{'Modelo': mod, 'Versão': ver, 'a [s]': round(a, 4), 'b [s por passo]': round(b, 4), 'R²': round(r2, 5),
               'Termo quadrático': round(c2, 5)} for (mod, ver), (a, b, r2, c2) in ajustes.items()]
             ).to_csv(ajustes_path, index=False)

# ------------------------------------------------------------------- 6. painel
CORES = {'Recálculo (sem cache)': '#c0392b', 'KV cache (memoização)': '#2471a3'}
ESTILO = {'F-PluvIA-G (V1)': ('o', '-'), 'F-PluvIA-VQ': ('s', '--')}
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
ax = axes[0]
xs_fit = np.linspace(0, max(tamanhos), 50)
for mod in MODELOS:
  sub = df[df['Modelo'] == mod]
  mk, ls = ESTILO[mod]
  for ver, col in COLS.items():
    a_, b_, r2, _ = ajustes[(mod, ver)]
    ax.plot(sub['Passos (S)'], sub[col], marker=mk, ls=ls, color=CORES[ver],
            label=f'{mod} · {ver}: t ≈ {a_:.2f} + {b_:.3f}·S (R² = {r2:.4f})')
ax.set_xlabel('Tamanho da entrada: passos de previsão S (× 30 min)')
ax.set_ylabel('Tempo médio por instância [s]')
ax.set_title(f'(a) Tempo × tamanho, dois modelos e duas versões – k = {k}, m = {m}', fontweight='bold')
ax.set_xticks(tamanhos)
ax.grid(True, ls=':', alpha=0.6)
ax.legend(fontsize=7.5)

ax = axes[1]
razao_pos = POS_RC / POS_KV
speed = {mod: float(df.loc[df['Modelo'] == mod, 'Speedup (recálculo / KV)'].mean()) for mod in MODELOS}
rotulos = ['Posições processadas\n(teórico, Θ, os dois)'] + [f'Tempo medido\n{mod}' for mod in MODELOS]
valores = [razao_pos] + list(speed.values())
barras = ax.bar(rotulos, valores, color=['#7f8c8d', '#2471a3', '#5dade2'])
for bar_, v in zip(barras, valores):
  ax.annotate(f'{v:.1f}×', (bar_.get_x() + bar_.get_width() / 2, v), ha='center', va='bottom',
              fontsize=11, fontweight='bold')
ax.set_yscale('log')
ax.set_ylim(min(0.5, min(speed.values()) / 2), razao_pos * 1.8)
ax.axhline(1, color='#555555', lw=0.8)
ax.set_ylabel('Razão recálculo / KV cache (escala log)')
ax.set_title('(b) Ganho assintótico × ganho real nos dois modelos', fontweight='bold')
ax.grid(True, axis='y', ls=':', alpha=0.6)
fig.suptitle(f'F-PluvIA-G (V1) × F-PluvIA-VQ: recálculo O(S·T·L·(n·d² + n²·d)) × KV cache O(S·L·(n·d² + n²·d)), '
             f'n ≤ {BLOCK}, T = {T_QUADRO}, L = {L_CAM}, d = {D_LARG} | {ENV_LABEL}', fontsize=10.5, fontweight='bold')
fig_path = os.path.join(SAIDA, f'painel_complexidade_bigO_v1_vs_vq_{TAG}.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()


# ------------------------------------------------------------------- 6b. gráfico no estilo do estudo de caso de ANALG
# Mesmo formato da Figura 1 do repositório ANALG-2025.2 (Comparação Experimental, eixos log-log, uma linha com
# marcador por versão e a ordem de complexidade na legenda), agora com as duas versões nos dois modelos.
ESTILO_ANALG = {
    ('F-PluvIA-G (V1)', 'Recálculo (sem cache)'): dict(color='#E69F00', marker='o', ls='-'),
    ('F-PluvIA-G (V1)', 'KV cache (memoização)'): dict(color='#56B4E9', marker='s', ls='-'),
    ('F-PluvIA-VQ', 'Recálculo (sem cache)'): dict(color='#B26E00', marker='o', ls='--', mfc='white'),
    ('F-PluvIA-VQ', 'KV cache (memoização)'): dict(color='#1F77B4', marker='s', ls='--', mfc='white'),
}
ORDEM = {'Recálculo (sem cache)': 'O(S·T·L·(n·d² + n²·d))', 'KV cache (memoização)': 'O(S·L·(n·d² + n²·d))'}
fig2, ax2 = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
for mod in MODELOS:
  sub = df[df['Modelo'] == mod]
  for ver, col in COLS.items():
    desvio = sub['Desvio recálculo [s]' if ver.startswith('Recálculo') else 'Desvio KV cache [s]']
    ax2.errorbar(sub['Passos (S)'], sub[col], yerr=desvio, lw=2, ms=7, capsize=3, elinewidth=0.8,
                 label=f'{mod} · {ver} ({ORDEM[ver]})', **ESTILO_ANALG[(mod, ver)])
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xticks(tamanhos)
ax2.set_xticklabels([f'{S}\n(+{30 * S} min)' for S in tamanhos])
ax2.minorticks_off()
ax2.set_xlabel('Tamanho da entrada: passos de previsão S (escala log)')
ax2.set_ylabel('Tempo médio (segundos)')
ax2.set_title('Comparação Experimental — Recálculo × KV Cache no F-PluvIA-G (V1) e no F-PluvIA-VQ',
              fontsize=14, color='#333333')
ax2.grid(True, which='major', ls='--', alpha=0.6)
ax2.legend(fontsize=9.5, loc='upper left')
fig2_path = os.path.join(SAIDA, f'figura_comparacao_experimental_bigO_v1_vs_vq_{TAG}.png')
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.show()

# Tabela no mesmo formato da Tabela 1 do estudo de caso (média, desvio e divergências por tamanho)
TAB_ANALG = df[['Modelo', 'Passos (S)', 'Horizonte [min]', 'Instâncias (m)', 'Tempo médio recálculo [s]',
                'Desvio recálculo [s]', 'Tempo médio KV cache [s]', 'Desvio KV cache [s]', 'Speedup (recálculo / KV)',
                'Divergências (bf16)']].copy()
print('Tabela no formato do estudo de caso (divergências = previsões bf16 não idênticas; a correção é conferida '
      'em fp32, token a token, na seção 4):')
display(TAB_ANALG.set_index(['Modelo', 'Passos (S)']))

# ------------------------------------------------------------------- 7. resultados e discussão
display(df.drop(columns=['Ambiente_Execucao']).set_index(['Modelo', 'Passos (S)']))
t_total = time.perf_counter() - t0_total
print('\nResumo do estudo de caso (decodificação gulosa, os dois modelos do duelo da Célula 13.7):')
print(f'  • Arquitetura: os dois modelos têm o mesmo Transformer ({P_TRANSFORMER / 1e6:.2f} M, L = {L_CAM}, d = {D_LARG}) e a '
      'mesma VQGAN em formato; só os pesos mudam. Logo, a complexidade teórica é a mesma: recálculo '
      'O(S·T·L·(n·d² + n²·d)) e KV cache O(S·L·(n·d² + n²·d)), um fator T a menos.')
for mod in MODELOS:
  a_rc, b_rc, r2_rc, c2_rc = ajustes[(mod, 'Recálculo (sem cache)')]
  a_kv, b_kv, r2_kv, c2_kv = ajustes[(mod, 'KV cache (memoização)')]
  print(f'  • {mod}: recálculo t ≈ {a_rc:.3f} + {b_rc:.3f}·S s (R² = {r2_rc:.4f}, quadrático {c2_rc:+.4f}); '
        f'KV cache t ≈ {a_kv:.3f} + {b_kv:.3f}·S s (R² = {r2_kv:.4f}, quadrático {c2_kv:+.4f}); '
        f'speedup médio {speed[mod]:.2f}×.')
_rc = {mod: ajustes[(mod, 'Recálculo (sem cache)')][1] for mod in MODELOS}
_kv = {mod: ajustes[(mod, 'KV cache (memoização)')][1] for mod in MODELOS}
_m1, _m2 = list(MODELOS)
print(f'  • Custo por passo, {_m2} / {_m1}: {_rc[_m2] / _rc[_m1]:.3f} (recálculo) e {_kv[_m2] / _kv[_m1]:.3f} (KV cache). '
      'Perto de 1 confirma que ajustar a VQGAN não muda o custo da previsão: o ganho de acerto do VQ vem sem custo de tempo.')
print(f'  • Trabalho: {br(POS_RC)} → {br(POS_KV)} posições por quadro ({razao_pos:.0f}× menos); tempo: '
      + ' e '.join(f'{speed[mod]:.2f}× ({mod})' for mod in MODELOS)
      + '. A distância entre a razão teórica e a medida são as constantes: com lote 1, cada passagem de 1 token é '
        'limitada pelo lançamento de kernels e pela leitura dos pesos da memória, não pelos FLOPs.')
_tab_kv = globals().get('TAB_KV')
if isinstance(_tab_kv, pd.DataFrame) and len(_tab_kv):
  print('\nConferência com a Célula 13.7 (todas as janelas de 2025 e 2026, horizonte fixo de 4 passos):')
  display(_tab_kv[[c for c in ('Modelo', 'Conjunto', 'Janelas', 'Tempo médio sem cache (s)',
                                'Tempo médio com KV cache (s)', 'Aceleração (×)', 'Janelas idênticas (%)')
                   if c in _tab_kv.columns]])
else:
  print('\n(TAB_KV da Célula 13.7 não está na sessão: sem a conferência com todas as janelas.)')
print(f'[Célula 13.7.1] Tabela -> {csv_path}\n[Célula 13.7.1] Ajustes -> {ajustes_path}\n'
      f'[Célula 13.7.1] Verificação -> {verif_path}\n[Célula 13.7.1] Painel -> {fig_path}\n'
      f'[Célula 13.7.1] Figura (estilo ANALG) -> {fig2_path}\n'
      f'[Célula 13.7.1] Tempo total de execução: {t_total:.0f} s')
