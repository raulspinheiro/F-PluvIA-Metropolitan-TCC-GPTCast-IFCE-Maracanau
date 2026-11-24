# @title
# Célula 13.7 (Cronus): Duelo final — F-PluvIA-G (V1, VQGAN original) × F-PluvIA-VQ (VQGAN ajustada à RMF), ambos com
# decodificação gulosa, no teste de 2025 (1.729 janelas) e no teste cego de 2026 (janelas regulares de fev–mai), com
# IC 95 % da diferença por bootstrap pareado em blocos diários. O vencedor é decidido em 2025; 2026 é relatado com a
# ressalva de que a VQGAN foi ajustada depois de conhecido o resultado cego da V1. Em seguida, a variante com KV cache
# (memoização das chaves e valores da atenção, Célula 13.4) de CADA modelo (V1 e VQ) roda em todas as janelas dos dois anos:
# mede o ganho de tempo e confere se as previsões e as métricas ficam iguais às do gerador original.
import contextlib
import glob
import io
import json
import math
import os
import re
import time
import warnings

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gptcast.utils.converters import dbz_to_rainfall

N_BOOT = 1000
SEED_DUELO = 42
NIVEIS_ALERTA = {'Atenção': 0.5, 'Alerta': 1.0, 'Severo': 5.0}
LIMIARES_PIXEL = (0.5, 1.0, 5.0)
STEP_MIN = 30
RODAR_KV = True                 # False = só o duelo, sem a variante com KV cache
KV_MODELOS = ('v1', 'vq')       # KV cache nos dois modelos (equivalência e aceleração de cada um)
TOL_IGUAL_MMH = 1e-2            # diferença máxima (mm/h) para considerar duas previsões iguais

faltam = [n for n in ('CRONUS', 'gpt8_v1', 'gpt8_vq', 'md_rmf', 'SERIES_CACHE_DIR', 'CONTEXT_STEPS', 'FORECAST_STEPS')
          if n not in globals()]
if faltam or 'vq' not in globals().get('CRONUS', {}):
  raise RuntimeError('Rode antes a Célula 13.6 e, com o F-PluvIA-VQ ativo, as Células 12, 12.1, 13.1 e 13.2. '
                     f'Faltam: {", ".join(faltam) or "a execução VQ"}')
for n in ('CKPT_DIR', 'PASTA_PREV'):
  if n not in globals():
    raise RuntimeError(f'"{n}" não existe: rode as Células 12 e 13.1 com o F-PluvIA-VQ ativo.')
CRONUS['vq'].update(cell12_dir=CKPT_DIR, pasta_2026=PASTA_PREV)
DEVICE = 'cuda'
SAIDA = CRONUS['vq']['results_dir']
TAG = CRONUS['vq']['env_tag']
_cron = globals().get('etapa')


def _rain(dbz):
  return np.clip(dbz_to_rainfall(np.clip(np.asarray(dbz, dtype=np.float64), 0.0, 60.0)), 0.0, 205.0).astype(np.float32)


# ------------------------------------------------------------------ 1. máscaras dos municípios
if 'mascaras_municipais_rmf' not in globals():
  raise RuntimeError('Rode a Célula 4.8 nesta sessão (ela define as máscaras municipais).')
M_STACK = np.stack(list(mascaras_municipais_rmf().values()))


def medias_municipais(r):
  return np.stack([r[..., m].mean(axis=-1) for m in M_STACK], axis=-1)


# ------------------------------------------------------------------ 2. previsões salvas (somente leitura)
def _carregar_2025(pasta):
  """{idx_janela: (pred_ft, tempo_ft)} a partir dos blocos da Célula 12."""
  out = {}
  for f in sorted(glob.glob(os.path.join(pasta, 'bloco_*.npz'))):
    if f.endswith('.tmp.npz'):
      continue
    with np.load(f) as z:
      for j, i in enumerate(z['idx']):
        out[int(i)] = (z['pred_ft'][j].astype(np.float32), float(abs(z['tempo_ft'][j])))
  return out


def _carregar_2026(pasta):
  """{(produto, ts_fim): (pred_ft, tempo_ft)} a partir dos arquivos por janela da Célula 13.1."""
  out = {}
  pad = re.compile(r'^(?P<prod>.+)_(?P<ts>\d{8}T\d{4})\.npz$')
  for f in glob.glob(os.path.join(pasta, '*.npz')):
    m = pad.match(os.path.basename(f))
    if not m or f.endswith('.tmp.npz'):
      continue
    ts = np.datetime64(pd.Timestamp(m['ts']).to_datetime64(), 'm')
    with np.load(f) as z:
      out[(m['prod'], ts)] = (z['pred_ft'].astype(np.float32), float(z['tempo_ft']))
  return out


# 2025: observação da partição de teste
te = md_rmf.splits['test']
P25 = {k: _carregar_2025(CRONUS[k]['cell12_dir']) for k in ('v1', 'vq')}
I25 = sorted(set(P25['v1']) & set(P25['vq']))
OBS25 = _rain(np.asarray(te.sequences_dbz)[I25][:, CONTEXT_STEPS:CONTEXT_STEPS + FORECAST_STEPS])
DIAS25 = np.asarray(te.timestamps)[I25].astype('datetime64[D]')
CTX25 = np.asarray(te.sequences_dbz)[I25][:, :CONTEXT_STEPS]

# 2026: só as janelas regulares de fev–mai (os caches RMF_2026_*); estudos de caso fora, como na Célula 13.2
DIR_2026 = os.path.join(SERIES_CACHE_DIR, 'teste_cego_2026')
SEQ26 = {}
for f in sorted(glob.glob(os.path.join(DIR_2026, 'RMF_2026_*.npz'))):
  with np.load(f) as z:
    for s, t in zip(z['seqs_dbz'], z['ts_fim'].astype('datetime64[m]')):
      SEQ26.setdefault(np.datetime64(t, 'm'), s)
P26 = {k: _carregar_2026(CRONUS[k]['pasta_2026']) for k in ('v1', 'vq')}
T26 = sorted(k for k in set(P26['v1']) & set(P26['vq']) if k[1] in SEQ26)   # (produto, ts_fim)
OBS26 = _rain(np.stack([SEQ26[t][CONTEXT_STEPS:CONTEXT_STEPS + FORECAST_STEPS] for _, t in T26]))
DIAS26 = np.array([t for _, t in T26], dtype='datetime64[m]').astype('datetime64[D]')
CTX26 = np.stack([SEQ26[t][:CONTEXT_STEPS] for _, t in T26])
print(f'[Célula 13.7] 2025: {len(I25)} janelas | 2026: {len(T26)} janelas regulares com as duas previsões.')

CONJ = {
    '2025': {'obs': OBS25, 'dias': DIAS25, 'ctx': CTX25,
             'pred': {k: np.stack([P25[k][i][0] for i in I25]) for k in ('v1', 'vq')},
             'tempo': {k: np.array([P25[k][i][1] for i in I25]) for k in ('v1', 'vq')}},
    '2026': {'obs': OBS26, 'dias': DIAS26, 'ctx': CTX26,
             'pred': {k: np.stack([P26[k][t][0] for t in T26]) for k in ('v1', 'vq')},
             'tempo': {k: np.array([P26[k][t][1] for t in T26]) for k in ('v1', 'vq')}},
}
NOME = {'v1': 'F-PluvIA-G (V1)', 'vq': 'F-PluvIA-VQ', 'kv_v1': 'F-PluvIA-G (V1) com KV cache',
        'kv_vq': 'F-PluvIA-VQ com KV cache'}


# ------------------------------------------------------------------ 3. métricas com bootstrap pareado por dia
def _contagens(o, p, thr):
  ob, pb = o >= thr, p >= thr
  eixo = tuple(range(1, o.ndim))
  return np.stack([np.sum(ob & pb, axis=eixo), np.sum(~ob & pb, axis=eixo), np.sum(ob & ~pb, axis=eixo)], -1).astype(float)


def _f1(c):
  tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
  with np.errstate(invalid='ignore', divide='ignore'):
    return 100 * 2 * tp / (2 * tp + fp + fn)


def _csi(c):
  tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
  with np.errstate(invalid='ignore', divide='ignore'):
    return 100 * tp / (tp + fp + fn)


def comparar(conj, a, b, preds=None):
  """Tabela de métricas de a e b e da diferença b − a, com IC 95 % por bootstrap em blocos diários."""
  P = preds or conj['pred']
  o, dias = conj['obs'], conj['dias']
  du, inv = np.unique(dias, return_inverse=True)
  amostras = np.random.default_rng(SEED_DUELO).integers(0, len(du), size=(N_BOOT, len(du)))
  o_mun = medias_municipais(o)
  linhas = []

  def _linha(nome_m, fn_cont, fn_met):
    cd = {}
    for k in (a, b):
      c = fn_cont(k)                                       # (N, 3) por janela
      d = np.zeros((len(du), 3))
      np.add.at(d, inv, c)
      cd[k] = d
    pa, pb = fn_met(cd[a].sum(0)), fn_met(cd[b].sum(0))
    dif = fn_met(cd[b][amostras].sum(1)) - fn_met(cd[a][amostras].sum(1))
    lo, hi = np.nanpercentile(dif, [2.5, 97.5])
    linhas.append({'Métrica': nome_m, NOME[a]: pa, NOME[b]: pb, 'Δ': pb - pa, 'IC 95% Δ inf': lo, 'IC 95% Δ sup': hi})

  for nivel, thr in NIVEIS_ALERTA.items():
    _linha(f'F1 municipal {nivel} (%)', lambda k: _contagens(o_mun, medias_municipais(P[k]), thr), _f1)
  for h in range(FORECAST_STEPS):
    _linha(f'F1 municipal Atenção +{(h + 1) * STEP_MIN} min (%)',
           lambda k: _contagens(o_mun[:, h], medias_municipais(P[k][:, h]), 0.5), _f1)
  for thr in LIMIARES_PIXEL:
    _linha(f'CSI pixel ≥ {thr:g} mm/h (%)', lambda k: _contagens(o, P[k], thr), _csi)
  linhas.append({'Métrica': 'MAE pixel (mm/h)', NOME[a]: float(np.mean(np.abs(P[a] - o))),
                 NOME[b]: float(np.mean(np.abs(P[b] - o)))})
  linhas.append({'Métrica': 'Razão de volume (prev./obs.)', NOME[a]: float(P[a].sum() / max(o.sum(), 1e-9)),
                 NOME[b]: float(P[b].sum() / max(o.sum(), 1e-9))})
  for r in linhas[-2:]:
    r['Δ'] = r[NOME[b]] - r[NOME[a]]
  return pd.DataFrame(linhas)


DUELO = {ano: comparar(CONJ[ano], 'v1', 'vq') for ano in ('2025', '2026')}


def _veredito(df):
  r = df.iloc[0]
  if r['IC 95% Δ inf'] > 0:
    return 'vq', 'F-PluvIA-VQ melhor (IC 95% do ganho no F1 de Atenção todo acima de zero)'
  if r['IC 95% Δ sup'] < 0:
    return 'v1', 'F-PluvIA-G (V1) melhor (IC 95% da diferença todo abaixo de zero)'
  return 'v1', 'empate técnico (o IC 95% contém o zero); fica a V1, que tem o teste cego legítimo'


VENCEDOR, TEXTO_2025 = _veredito(DUELO['2025'])
_, TEXTO_2026 = _veredito(DUELO['2026'])
for ano, df in DUELO.items():
  df.to_csv(os.path.join(SAIDA, f'cronus_duelo_v1_vs_vq_{ano}_{TAG}.csv'), index=False)
  print(f'\n=== Célula 13.7: {NOME["v1"]} × {NOME["vq"]} — {"teste 2025" if ano == "2025" else "teste cego 2026 (com ressalva)"} ===')
  display(df)
print(f'\nVeredito em 2025 (decide): {TEXTO_2025}.')
print(f'Em 2026 (só relatado): {TEXTO_2026}.')
print(f'Modelo vencedor em 2025: {NOME[VENCEDOR]}. KV cache medido em: {", ".join(NOME[m] for m in KV_MODELOS)}.')

# ------------------------------------------------------------------ 4. variante com KV cache do vencedor
RES_KV, TAB_KV = None, None
if RODAR_KV:
  RES_KV, TAB_KV = {}, []
  for MOD in KV_MODELOS:
    M = gpt8_v1 if MOD == 'v1' else gpt8_vq
    M.eval()
    TR = M.transformer
    WS = int(math.sqrt(int(TR.config.block_size) // 8))

    def _proximo_token(logits, M=M):
      lg = M.top_k_logits(logits[:, -1, :], 1)
      return torch.topk(F.softmax(lg, dim=-1), k=1, dim=-1)[1]

    def _quadro_kv(ind_ctx, c_idx, TR=TR, _proximo_token=_proximo_token):
      b, _, h, w = ind_ctx.shape
      prefixo = torch.cat((c_idx.reshape(b, -1), ind_ctx.reshape(b, -1)), dim=1)
      logits, _, cache = TR.forward_with_past(prefixo)
      n = prefixo.shape[1]
      tok = _proximo_token(logits)
      toks = [tok]
      for _ in range(h * w - 1):
        logits, _, novo = TR.forward_with_past(tok, past=[cache], past_length=n)
        cache = torch.cat((cache, novo), dim=-2)
        n += 1
        tok = _proximo_token(logits)
        toks.append(tok)
      return torch.cat(toks, dim=1).reshape(b, h, w)

    @torch.inference_mode()
    def prever_kv(ctx_dbz, M=M, _quadro_kv=_quadro_kv):
      x = torch.tensor(np.clip(np.nan_to_num(ctx_dbz, nan=0.0), 0, 60) / 30.0 - 1.0, dtype=torch.float32, device=DEVICE)
      s = x.shape[0]
      xs = einops.rearrange(x, 's h w -> (s h) w')[None, None].contiguous()
      quant, idx = M.encode_to_z(xs)
      _, c_idx = M.encode_to_c(xs)
      qs = quant.shape
      ind = idx.reshape(qs[0], s, qs[2] // s, qs[3])
      ih, iw = ind.shape[2], ind.shape[3]
      quadros = []
      for _ in range(FORECAST_STEPS):
        with torch.autocast('cuda', dtype=torch.bfloat16):
          p = _quadro_kv(ind[:, -s:], c_idx)
        quadros.append(M.decode_to_img(p.reshape(1, -1), torch.Size([1, qs[1], ih, iw])))
        ind = torch.cat((ind, p[:, None]), dim=1)
      y = torch.cat(quadros, dim=1).float().cpu().numpy().clip(-1, 1)
      return _rain((y + 1.0) * 30.0)[0]

    _ = prever_kv(CONJ['2025']['ctx'][0])                  # aquecimento
    chave = f'kv_{MOD}'
    for ano in ('2025', '2026'):
      ctx = CONJ[ano]['ctx']
      P_kv, T_kv = np.empty_like(CONJ[ano]['obs']), np.empty(len(ctx))
      with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        for j in range(len(ctx)):
          torch.cuda.synchronize()
          t0 = time.perf_counter()
          P_kv[j] = prever_kv(ctx[j])
          torch.cuda.synchronize()
          T_kv[j] = time.perf_counter() - t0
          if (j + 1) % 250 == 0:
            print(f'  KV cache {NOME[MOD]} {ano}: {j + 1}/{len(ctx)} janelas', flush=True)
      P_rc = CONJ[ano]['pred'][MOD]
      dif = np.abs(P_kv - P_rc).reshape(len(ctx), -1).max(axis=1)
      CONJ[ano]['pred'][chave] = P_kv
      cmp = comparar(CONJ[ano], MOD, chave)
      cmp.to_csv(os.path.join(SAIDA, f'cronus_kvcache_{MOD}_{ano}_{TAG}.csv'), index=False)
      t_rc = CONJ[ano]['tempo'][MOD]
      RES_KV[(MOD, ano)] = cmp
      TAB_KV.append({'Modelo': NOME[MOD], 'Conjunto': ano, 'Janelas': len(ctx),
                     'Tempo médio sem cache (s)': float(np.mean(t_rc)),
                     'Tempo médio com KV cache (s)': float(np.mean(T_kv)),
                     'Aceleração (×)': float(np.mean(t_rc) / np.mean(T_kv)),
                     'Janelas idênticas (%)': float(100 * np.mean(dif <= TOL_IGUAL_MMH)),
                     'Maior diferença (mm/h)': float(dif.max()),
                     'Δ F1 Atenção (KV − sem cache)': float(cmp.iloc[0]['Δ']),
                     'IC 95% Δ': f"{cmp.iloc[0]['IC 95% Δ inf']:.2f} a {cmp.iloc[0]['IC 95% Δ sup']:.2f}"})
      CONJ[ano]['pred'].pop(chave)                         # libera memória
  TAB_KV = pd.DataFrame(TAB_KV)
  TAB_KV.to_csv(os.path.join(SAIDA, f'cronus_kvcache_resumo_{TAG}.csv'), index=False)
  print('\n=== Célula 13.7: sem cache × com KV cache, nos dois modelos e nos dois anos (todas as janelas) ===')
  display(TAB_KV)
  print('Esperado: Δ F1 ≈ 0 com o IC contendo o zero (o KV cache é a mesma conta, só sem recálculo). Diferenças '
        'pequenas vêm do bf16: a ordem das somas muda e empates entre os dois maiores logits podem virar. Um ganho '
        'claro de acerto com o cache indicaria erro de implementação, não um modelo melhor.')

CELL13_7_DUELO = {'vencedor_2025': NOME[VENCEDOR], 'veredito_2025': TEXTO_2025, 'veredito_2026': TEXTO_2026,
                  'janelas_2025': len(I25), 'janelas_2026': len(T26),
                  'kv_cache': None if TAB_KV is None else TAB_KV.to_dict(orient='records')}
with open(os.path.join(SAIDA, f'cronus_duelo_resumo_{TAG}.json'), 'w') as f:
  json.dump(CELL13_7_DUELO, f, indent=2, ensure_ascii=False, default=str)
print(f'\nResumo salvo em {SAIDA}. Próximo passo: Células Extra, Deploy, Backup e, por último, o Encerramento.')

# Para o Deploy e o Backup levarem também a pasta da V1 (o modelo ativo agora é o VQ)
CRONUS_PASTAS_RESULTADOS = {CRONUS['v1']['env_tag']: CRONUS['v1']['results_dir']}
CRONUS_PASTAS_EXTRAS = {f"resultados_{CRONUS['v1']['env_tag']}": CRONUS['v1']['results_dir']}
