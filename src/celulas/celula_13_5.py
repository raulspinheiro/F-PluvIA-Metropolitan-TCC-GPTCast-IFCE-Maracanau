# @title
# Célula 13.5: Exportação SOMENTE LEITURA das previsões por município para a interface web (APS 2026.2)
#
# O que faz: lê as previsões JÁ SALVAS no Drive pela Célula 13.1 (um .npz por janela do teste cego de 2026 e dos
# estudos de caso) e as janelas observadas (caches mensais e de caso), calcula a chuva MÉDIA de cada município da
# RMF em cada horizonte (+30, +60, +90, +120 min) e grava dois CSV para a interface web.
#
# O que NÃO faz: não treina, não carrega o modelo, não roda inferência, não usa GPU e não altera nenhum arquivo
# existente. Só escreve na pasta nova  <RESULTS_ENV_DIR>/interface_web/.
#
# Pré-requisitos nesta sessão: Célula 1 (Drive e caminhos) e Célula 5 (limites dos 19 municípios: gdf_rmf, md_rmf).
# Se a Célula 12.1 ou 13.2 já rodou, as máscaras municipais (mun_masks) são reaproveitadas; senão, são refeitas
# com a MESMA regra (contorno externo do polígono, mínimo de 4 pixels de 2 km).

import glob
import os
import re

import numpy as np
import pandas as pd

SERIES_CACHE_DIR = globals()['SERIES_CACHE_DIR']
ENV_TAG = globals().get('ENV_TAG', 'Ambiente_Ativo')
RESULTS_ENV_DIR = globals().get('RESULTS_ENV_DIR', os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}'))
DIR_2026 = os.path.join(SERIES_CACHE_DIR, 'teste_cego_2026')
OUT = os.path.join(RESULTS_ENV_DIR, 'interface_web')
os.makedirs(OUT, exist_ok=True)

CONTEXT_STEPS, FORECAST_STEPS, STEP_MIN = 7, 4, 30
N_DESTAQUE = 30                         # janelas de 2026 de maior chuva observada levadas ao banco da interface
SIGLA = {'ft': 'F-PluvIA-G', 'ft_media': 'F-PluvIA-C8', 'zs': 'Zero-Shot',
         'linda': 'LINDA', 'lk': 'Lucas-Kanade', 'pers': 'Persistência'}
CASOS = {'20260516_20260517': 'Estratiforme longo (16–17/05/2026)',
         '20260910_20260910': 'Convectivo (10/09/2026)'}

try:                                    # mesma conversão dBZ -> mm/h da Célula 13.1
  from gptcast.utils.converters import dbz_to_rainfall
except Exception:                       # Marshall–Palmer, Z = 200 R^1,6 (a mesma relação, sem o pacote)
  def dbz_to_rainfall(dbz):
    return (10 ** (np.asarray(dbz) / 10.0) / 200.0) ** (1 / 1.6)


def _rain(dbz):
  return np.clip(dbz_to_rainfall(np.clip(np.asarray(dbz, dtype=np.float64), 0.0, 60.0)), 0.0, 205.0).astype(np.float32)


# ----------------------------------------------------------- 1. máscaras municipais (mesma regra da Célula 12.1)
mun_masks = globals().get('mun_masks') or {}
if not mun_masks:
  from matplotlib.path import Path as MplPath
  gdf_rmf, md_rmf = globals().get('gdf_rmf'), globals().get('md_rmf')
  if gdf_rmf is None or md_rmf is None:
    raise RuntimeError('Rode a Célula 5 (limites municipais) antes desta célula.')
  lon_g, lat_g = np.meshgrid(md_rmf.target_lons, md_rmf.target_lats)
  pts = np.column_stack([lon_g.ravel(), lat_g.ravel()])
  for _, r in gdf_rmf.iterrows():
    polys = list(r.geometry.geoms) if r.geometry.geom_type == 'MultiPolygon' else [r.geometry]
    m = np.zeros(pts.shape[0], bool)
    for pg in polys:
      m |= MplPath(np.asarray(pg.exterior.coords)).contains_points(pts)
    if m.sum() >= 4:
      mun_masks[r['name']] = m.reshape(lon_g.shape)
NOMES_MUN = list(mun_masks)
MASKS = np.stack([mun_masks[n] for n in NOMES_MUN])            # (M, H, W)
print(f'[Célula 13.5] {len(NOMES_MUN)} municípios com máscara: {", ".join(NOMES_MUN)}')


def medias_municipais(campo_4xHxW):
  """(4, H, W) mm/h -> (4, M) chuva média na área de cada município."""
  c = np.asarray(campo_4xHxW, dtype=np.float32)
  return np.stack([c[:, m].mean(axis=1) for m in MASKS], axis=1)


# ----------------------------------------------------------- 2. janelas observadas (somente leitura)
OBS = {}                                                      # (produto não importa) ts_fim -> (seq_dbz, caso)
for f in sorted(glob.glob(os.path.join(DIR_2026, '*.npz'))):
  base = os.path.basename(f)
  if not (base.startswith('RMF_2026_') or base.startswith('caso_')):
    continue
  caso = next((v for k, v in CASOS.items() if k in base), None)
  with np.load(f) as z:
    for s, t in zip(z['seqs_dbz'], z['ts_fim'].astype('datetime64[m]')):
      OBS.setdefault(np.datetime64(t, 'm'), (s, caso))
print(f'[Célula 13.5] {len(OBS)} janelas observadas lidas dos caches de 2026.')

# ----------------------------------------------------------- 3. previsões salvas pela Célula 13.1 (somente leitura)
pastas = sorted(glob.glob(os.path.join(DIR_2026, 'previsoes', 'janelas_*_amostragem2024_media8')),
                key=os.path.getmtime)
if not pastas:
  raise RuntimeError('Pasta de previsões da Célula 13.1 (…_amostragem2024_media8) não encontrada no Drive.')
PASTA = pastas[-1]
print(f'[Célula 13.5] Previsões: {PASTA}')

linhas, sem_obs = [], 0
padrao = re.compile(r'^(?P<prod>.+)_(?P<ts>\d{8}T\d{4})\.npz$')
for f in sorted(glob.glob(os.path.join(PASTA, '*.npz'))):
  m = padrao.match(os.path.basename(f))
  if not m or f.endswith('.tmp.npz'):
    continue
  ts_fim = np.datetime64(pd.Timestamp(m['ts']).to_datetime64(), 'm')
  t0 = ts_fim - np.timedelta64(STEP_MIN * FORECAST_STEPS, 'm')
  if ts_fim not in OBS:
    sem_obs += 1
    continue
  seq, caso = OBS[ts_fim]
  obs_mun = medias_municipais(_rain(seq[CONTEXT_STEPS:CONTEXT_STEPS + FORECAST_STEPS]))
  with np.load(f) as z:
    for k, sigla in SIGLA.items():
      if f'pred_{k}' not in z.files:
        continue
      prev_mun = medias_municipais(z[f'pred_{k}'].astype(np.float32))
      for h in range(FORECAST_STEPS):
        for j, nome in enumerate(NOMES_MUN):
          linhas.append((str(t0), m['prod'], caso or '', sigla, nome, (h + 1) * STEP_MIN,
                         round(float(prev_mun[h, j]), 3), round(float(obs_mun[h, j]), 3)))

df = pd.DataFrame(linhas, columns=['t0_utc', 'produto', 'caso', 'metodo', 'municipio', 'horizonte_min',
                                   'chuva_prevista_mm_h', 'chuva_observada_mm_h'])
if df.empty:
  raise RuntimeError('Nenhuma previsão com observação correspondente foi encontrada.')
if sem_obs:
  print(f'[Aviso Célula 13.5] {sem_obs} arquivo(s) de previsão sem janela observada correspondente (ignorados).')

# ----------------------------------------------------------- 4. arquivos para a interface web
completo = os.path.join(OUT, f'previsoes_municipio_teste_cego_2026_completo_{ENV_TAG}.csv.gz')
df.to_csv(completo, index=False, compression='gzip')

serie = df[df['caso'] == '']
chuva_janela = (serie[serie['metodo'] == 'F-PluvIA-G'].groupby('t0_utc')['chuva_observada_mm_h'].mean()
                .sort_values(ascending=False))
destaque_t0 = set(chuva_janela.head(N_DESTAQUE).index) | set(df.loc[df['caso'] != '', 't0_utc'])
dest = df[df['t0_utc'].isin(destaque_t0)].copy()
dest.loc[dest['caso'] == '', 'caso'] = 'Destaque: maior chuva observada no teste cego de 2026'
destaque = os.path.join(OUT, f'previsoes_municipio_destaques_interface_web_{ENV_TAG}.csv')
dest.to_csv(destaque, index=False)

print(f'[Célula 13.5] Completo: {df["t0_utc"].nunique()} janelas × {df["metodo"].nunique()} métodos × '
      f'{len(NOMES_MUN)} municípios × 4 horizontes = {len(df)} linhas → {completo}')
print(f'[Célula 13.5] Para a interface web: {dest["t0_utc"].nunique()} janelas '
      f'({N_DESTAQUE} de maior chuva + estudos de caso), {len(dest)} linhas → {destaque}')
print('[Célula 13.5] Nenhum arquivo existente foi alterado. Envie o CSV de destaques (e, se quiser, o completo).')
