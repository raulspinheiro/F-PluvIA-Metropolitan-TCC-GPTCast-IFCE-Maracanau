# @title
# Célula 4.8 (Cronus - V1 vs VQ - Validação 2024): F-PluvIA-G (V1) × F-PluvIA-VQ na validação de 2024 — (a) teto de reconstrução das duas VQGAN
# nos quadros observados e (b) previsão gulosa nas mesmas janelas sorteadas, com F1 municipal dos alertas, CSI, MAE e
# razão de volume, e IC 95 % da diferença por bootstrap pareado em blocos diários. Decide se o F-PluvIA-C8 também é
# refeito com a VQGAN nova. Só usa 2024: o teste de 2025 e o teste cego de 2026 continuam intocados.
import contextlib
import io
import json
import math
import os
import urllib.request
import warnings

import einops
import numpy as np
import pandas as pd
import torch
from gptcast.utils.converters import dbz_to_rainfall
from matplotlib.path import Path as MplPath

N_JANELAS_VAL = 240            # janelas de 2024 sorteadas (as mesmas para os dois modelos); None = todas
SEED_VAL = 42
N_BOOT = 1000
NIVEIS_ALERTA = {'Atenção': 0.5, 'Alerta': 1.0, 'Severo': 5.0}   # mm/h, média do município
LIMIARES_PIXEL = (0.5, 1.0, 5.0)
STEP_MIN = 30

faltam = [n for n in ('md_rmf', 'gpt8', 'gpt8_vq', 'vq_rmf', 'CONTEXT_STEPS', 'FORECAST_STEPS', 'RESULTS_ENV_DIR', 'ENV_TAG')
          if n not in globals()]
if faltam:
  raise RuntimeError(f'Rode antes as Células 1 a 4, 4.6 e 4.7. Faltam: {", ".join(faltam)}')
DEVICE = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _rain(dbz):
  return np.clip(dbz_to_rainfall(np.clip(np.asarray(dbz, dtype=np.float64), 0.0, 60.0)), 0.0, 205.0).astype(np.float32)


def mascaras_municipais_rmf():
  """Máscaras dos 19 municípios na malha de 2 km, com a mesma regra das Células 12.1 e 13.2 (≥ 4 pixels)."""
  if globals().get('mun_masks'):
    return dict(globals()['mun_masks'])
  gdf = globals().get('gdf_rmf')
  if gdf is None:
    import geopandas as gpd
    rmf = ['Aquiraz', 'Cascavel', 'Caucaia', 'Chorozinho', 'Eusébio', 'Fortaleza', 'Guaiúba', 'Horizonte', 'Itaitinga',
           'Maracanaú', 'Maranguape', 'Pacajus', 'Pacatuba', 'Paracuru', 'Paraipaba', 'Pindoretama',
           'São Gonçalo do Amarante', 'São Luís do Curu', 'Trairi']
    local = os.path.join(globals().get('BASE_WORK_DIR', '/content'), 'ceara_municipios_ibge.json')
    if not os.path.exists(local):
      urllib.request.urlretrieve('https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-23-mun.json', local)
    gdf_ce = gpd.read_file(local)
    gdf = gdf_ce[gdf_ce['name'].isin(rmf)]
  lon_g, lat_g = np.meshgrid(md_rmf.target_lons, md_rmf.target_lats)
  pts = np.column_stack([lon_g.ravel(), lat_g.ravel()])
  out = {}
  for _, r in gdf.iterrows():
    polys = list(r.geometry.geoms) if r.geometry.geom_type == 'MultiPolygon' else [r.geometry]
    m = np.zeros(pts.shape[0], bool)
    for pg in polys:
      m |= MplPath(np.asarray(pg.exterior.coords)).contains_points(pts)
    if m.sum() >= 4:
      out[r['name']] = m.reshape(lon_g.shape)
  return out


MASCARAS = mascaras_municipais_rmf()
M_STACK = np.stack(list(MASCARAS.values()))
print(f'[Célula 4.8] {len(MASCARAS)} municípios com máscara na malha de 2 km.')


def medias_municipais(r):
  """(..., H, W) -> (..., M)"""
  return np.stack([r[..., m].mean(axis=-1) for m in M_STACK], axis=-1)


# ------------------------------------------------------------------ janelas de 2024 (fixas)
va = md_rmf.splits['val']
n_tot = len(va.sequences_dbz)
rng = np.random.default_rng(SEED_VAL)
IDX = np.sort(rng.choice(n_tot, size=min(N_JANELAS_VAL or n_tot, n_tot), replace=False))
SEQS = np.clip(np.nan_to_num(np.asarray(va.sequences_dbz[IDX], dtype=np.float32), nan=0.0), 0.0, 60.0)
TS_FIM = np.asarray(va.timestamps)[IDX].astype('datetime64[m]')
DIAS = TS_FIM.astype('datetime64[D]')
OBS = _rain(SEQS[:, CONTEXT_STEPS:CONTEXT_STEPS + FORECAST_STEPS])            # (N, 4, H, W)

# ------------------------------------------------------------------ (a) teto de reconstrução das duas VQGAN
@torch.inference_mode()
def _reconstruir(vqm, seqs, empilhado):
  out = np.empty((len(seqs), FORECAST_STEPS) + seqs.shape[-2:], np.float32)
  for i in range(0, len(seqs), 8):
    x = torch.from_numpy(seqs[i:i + 8] / 30.0 - 1.0).to(DEVICE)                 # (b, 11, H, W)
    b, t, h, w = x.shape
    if empilhado:
      z, _, _ = vqm.encode(x.reshape(b, 1, t * h, w))
      rec = vqm.decode(z).reshape(b, t, h, w)
    else:
      z, _, _ = vqm.encode(x.reshape(b * t, 1, h, w))
      rec = vqm.decode(z).reshape(b, t, h, w)
    y = rec[:, CONTEXT_STEPS:CONTEXT_STEPS + FORECAST_STEPS].float().cpu().numpy().clip(-1, 1)
    out[i:i + b] = _rain((y + 1.0) * 30.0)
  return out


def _csi(o, p, thr):
  ob, pb = o >= thr, p >= thr
  tp, fp, fn = np.sum(ob & pb), np.sum(~ob & pb), np.sum(ob & ~pb)
  return float(tp / (tp + fp + fn)) if (tp + fp + fn) else float('nan')


linhas_rec = []
for nome, vqm in (('VQGAN original (V1)', gpt8.first_stage_model), ('VQGAN ajustada à RMF', vq_rmf)):
  for modo in (False, True):
    R = _reconstruir(vqm, SEQS, modo)
    linhas_rec.append({'VQGAN': nome, 'Modo': 'empilhado (11 quadros)' if modo else 'quadro isolado',
                       **{f'CSI ≥ {t:g}': _csi(OBS, R, t) for t in LIMIARES_PIXEL},
                       'MAE (mm/h)': float(np.mean(np.abs(R - OBS))),
                       'Razão de volume': float(R.sum() / max(OBS.sum(), 1e-9))})
df_rec = pd.DataFrame(linhas_rec)

# ------------------------------------------------------------------ (b) previsão gulosa (a mesma da Célula 12)
WS = int(math.sqrt(gpt8.transformer.config.block_size // 8))


@torch.inference_mode()
def prever_guloso(model, ctx_dbz):
  x = torch.tensor(ctx_dbz / 30.0 - 1.0, dtype=torch.float32, device=DEVICE)
  s = x.shape[0]
  xs = einops.rearrange(x, 's h w -> (s h) w')[None, None].contiguous()
  quant, idx = model.encode_to_z(xs)
  _, c_idx = model.encode_to_c(xs)
  qs = quant.shape
  ind = idx.reshape(qs[0], s, qs[2] // s, qs[3])
  ih, iw = ind.shape[2], ind.shape[3]
  quadros = []
  for _ in range(FORECAST_STEPS):
    with torch.autocast('cuda', dtype=torch.bfloat16):
      p = model.predict_next_frame_indices(ind[:, -s:], c_idx, WS, temperature=1.0, top_k=1, show_progress=False)
    quadros.append(model.decode_to_img(p.reshape(1, -1), torch.Size([1, qs[1], ih, iw])))
    ind = torch.cat((ind, p[:, None]), dim=1)
  y = torch.cat(quadros, dim=1).float().cpu().numpy().clip(-1, 1)
  return _rain((y + 1.0) * 30.0)[0]


PRED = {}
for nome, model in (('F-PluvIA-G (V1)', gpt8), ('F-PluvIA-VQ', gpt8_vq)):
  P = np.empty_like(OBS)
  with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
    warnings.simplefilter('ignore')
    for j in range(len(SEQS)):
      torch.manual_seed(SEED_VAL + int(IDX[j]))
      P[j] = prever_guloso(model, SEQS[j, :CONTEXT_STEPS])
  PRED[nome] = P
  print(f'[Célula 4.8] {nome}: {len(SEQS)} previsões gulosas de 2024 prontas.', flush=True)


# ------------------------------------------------------------------ métricas e bootstrap pareado por dia
def _contagens_mun(o_mun, p_mun, thr):
  ob, pb = o_mun >= thr, p_mun >= thr
  return np.array([np.sum(ob & pb), np.sum(~ob & pb), np.sum(ob & ~pb)], float)   # tp, fp, fn por janela somados


def _f1(c):
  tp, fp, fn = c
  return 100 * 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float('nan')


O_MUN = medias_municipais(OBS)                                  # (N, 4, M)
P_MUN = {k: medias_municipais(v) for k, v in PRED.items()}
dias_u, inv = np.unique(DIAS, return_inverse=True)


def _por_dia(fn):
  """Soma por dia de uma estatística aditiva por janela: (D, ...)"""
  vals = np.stack([fn(j) for j in range(len(SEQS))])
  out = np.zeros((len(dias_u),) + vals.shape[1:])
  np.add.at(out, inv, vals)
  return out


nomes = list(PRED)
linhas, boots = [], {}
rng_b = np.random.default_rng(SEED_VAL)
amostras = rng_b.integers(0, len(dias_u), size=(N_BOOT, len(dias_u)))
for nivel, thr in NIVEIS_ALERTA.items():
  cd = {k: _por_dia(lambda j, k=k: _contagens_mun(O_MUN[j], P_MUN[k][j], thr)) for k in nomes}
  pontos = {k: _f1(cd[k].sum(0)) for k in nomes}
  dif = np.array([_f1(cd[nomes[1]][a].sum(0)) - _f1(cd[nomes[0]][a].sum(0)) for a in amostras])
  lo, hi = np.nanpercentile(dif, [2.5, 97.5])
  linhas.append({'Métrica': f'F1 municipal {nivel} (%)', nomes[0]: pontos[nomes[0]], nomes[1]: pontos[nomes[1]],
                 'Δ (VQ − V1)': pontos[nomes[1]] - pontos[nomes[0]], 'IC 95% Δ inf': lo, 'IC 95% Δ sup': hi})
for thr in LIMIARES_PIXEL:
  cd = {k: _por_dia(lambda j, k=k: _contagens_mun(OBS[j], PRED[k][j], thr)) for k in nomes}
  csi = lambda c: 100 * c[0] / (c[0] + c[1] + c[2]) if c.sum() else float('nan')
  pontos = {k: csi(cd[k].sum(0)) for k in nomes}
  dif = np.array([csi(cd[nomes[1]][a].sum(0)) - csi(cd[nomes[0]][a].sum(0)) for a in amostras])
  lo, hi = np.nanpercentile(dif, [2.5, 97.5])
  linhas.append({'Métrica': f'CSI pixel ≥ {thr:g} mm/h (%)', nomes[0]: pontos[nomes[0]], nomes[1]: pontos[nomes[1]],
                 'Δ (VQ − V1)': pontos[nomes[1]] - pontos[nomes[0]], 'IC 95% Δ inf': lo, 'IC 95% Δ sup': hi})
linhas.append({'Métrica': 'MAE pixel (mm/h)', **{k: float(np.mean(np.abs(PRED[k] - OBS))) for k in nomes}})
linhas.append({'Métrica': 'Razão de volume (prev./obs.)', **{k: float(PRED[k].sum() / max(OBS.sum(), 1e-9)) for k in nomes}})
for r in linhas[-2:]:
  r['Δ (VQ − V1)'] = r[nomes[1]] - r[nomes[0]]
df_val = pd.DataFrame(linhas)

f1a = df_val.iloc[0]
if f1a['IC 95% Δ inf'] > 0:
  VEREDITO_2024 = 'F-PluvIA-VQ melhor que a V1 no F1 de Atenção em 2024 (IC 95% do ganho todo acima de zero)'
  INCLUIR_C8_VQ = True
elif f1a['IC 95% Δ sup'] < 0:
  VEREDITO_2024 = 'F-PluvIA-VQ pior que a V1 no F1 de Atenção em 2024 (IC 95% todo abaixo de zero)'
  INCLUIR_C8_VQ = False
else:
  VEREDITO_2024 = 'empate técnico no F1 de Atenção em 2024 (IC 95% da diferença contém o zero)'
  INCLUIR_C8_VQ = False

arq_rec = os.path.join(RESULTS_ENV_DIR, f'cronus_teto_reconstrucao_v1_vs_vq_validacao2024_{ENV_TAG}.csv')
arq_val = os.path.join(RESULTS_ENV_DIR, f'cronus_previsao_v1_vs_vq_validacao2024_{ENV_TAG}.csv')
df_rec.to_csv(arq_rec, index=False)
df_val.to_csv(arq_val, index=False)
CELL4_8_VALIDACAO = {'veredito_2024': VEREDITO_2024, 'incluir_c8_vq': INCLUIR_C8_VQ, 'n_janelas': int(len(SEQS)),
                     'n_dias': int(len(dias_u)), 'reconstrucao_csv': arq_rec, 'previsao_csv': arq_val}
with open(arq_val.replace('.csv', '.json'), 'w') as f:
  json.dump(CELL4_8_VALIDACAO, f, indent=2, ensure_ascii=False)

print('\n=== Célula 4.8 (a): teto de reconstrução em 2024 (quadros-alvo +30 a +120 min) ===')
display(df_rec)
print(f'\n=== Célula 4.8 (b): previsão gulosa em {len(SEQS)} janelas de 2024 ({len(dias_u)} dias) ===')
display(df_val)
print(f'\nVeredito em 2024: {VEREDITO_2024}.')
print(f'F-PluvIA-C8 com a VQGAN nova nos testes: {"SIM" if INCLUIR_C8_VQ else "NÃO (só o guloso, economiza ~8 h)"}.')
print('Próximo passo: rodar o caminho da V1 (Células 5 a 13.2) e depois a Célula 13.6.')
