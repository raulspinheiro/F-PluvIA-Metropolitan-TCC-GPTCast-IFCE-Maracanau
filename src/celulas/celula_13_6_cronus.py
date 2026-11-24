# @title
# Célula 13.6 (Cronus): Troca do modelo ativo — F-PluvIA-G (V1) → F-PluvIA-VQ, para rodar de novo as Células 4.4,
# 12, 12.1, 13.1 e 13.2 com a VQGAN ajustada. Guarda onde ficaram os resultados da V1, aponta as saídas para uma pasta
# própria (sufixo _VQ, nada da V1 é sobrescrito) e reaproveita o que não depende da VQGAN nova (Zero-Shot, LINDA,
# Lucas-Kanade e persistência), que é idêntico nas duas execuções.
import os

faltam = [n for n in ('gpt8', 'gpt8_vq', 'best_ckpt_path_vq', 'ENV_TAG', 'RESULTS_ENV_DIR', 'SERIES_CACHE_DIR')
          if n not in globals()]
if faltam:
  raise RuntimeError(f'Rode antes as Células 4.6 a 4.8. Faltam: {", ".join(faltam)}')

if globals().get('CRONUS', {}).get('ativo') == 'vq':
  print('[Célula 13.6] O F-PluvIA-VQ já é o modelo ativo nesta sessão; nada a fazer.')
else:
  for n in ('CKPT_DIR', 'PASTA_PREV'):
    if n not in globals():
      raise RuntimeError(f'"{n}" não existe: rode antes o caminho da V1 até a Célula 13.2 (as Células 12 e 13.1 '
                         'definem onde ficaram as previsões da V1).')
  CRONUS = {
      'ativo': 'vq',
      'v1': {'env_tag': ENV_TAG, 'results_dir': RESULTS_ENV_DIR, 'cell12_dir': CKPT_DIR, 'pasta_2026': PASTA_PREV,
             'checkpoint': os.path.basename(globals().get('best_ckpt_path', '?')),
             'cell12': globals().get('CELL12_FULL_2025_RESULTS')},
  }
  gpt8_v1 = gpt8                                   # o F-PluvIA-G (V1) continua na sessão com este nome
  best_ckpt_path_v1 = globals().get('best_ckpt_path')
  ENV_TAG = f"{CRONUS['v1']['env_tag']}_VQ"
  RESULTS_ENV_DIR = os.path.join(SERIES_CACHE_DIR, f'resultados_{ENV_TAG}')
  os.makedirs(RESULTS_ENV_DIR, exist_ok=True)
  gpt8 = gpt8_vq
  best_ckpt_path = best_ckpt_path_vq
  globals().pop('CELL12_FULL_2025_RESULTS', None)  # a 13.1 deve seguir a Célula 12 da execução VQ, não a da V1

  # Overrides lidos pelas Células 12 e 13.1 (versão Cronus)
  CRONUS_INCLUIR_MEDIA = bool(globals().get('INCLUIR_C8_VQ', False))
  CRONUS_REUSO_12 = [(CRONUS['v1']['cell12_dir'], ('zs', 'linda', 'lk', 'pers'))]
  CRONUS_REUSO_13_1 = [(CRONUS['v1']['pasta_2026'], ('zs', 'linda', 'lk', 'pers'))]
  CRONUS['vq'] = {'env_tag': ENV_TAG, 'results_dir': RESULTS_ENV_DIR, 'checkpoint': os.path.basename(best_ckpt_path_vq),
                  'incluir_c8': CRONUS_INCLUIR_MEDIA}

  print('=== Célula 13.6: modelo ativo agora é o F-PluvIA-VQ ===')
  print(f'V1 (guardada): {CRONUS["v1"]["results_dir"]}')
  print(f'VQ (nova):     {RESULTS_ENV_DIR}')
  print(f'F-PluvIA-C8 com a VQGAN nova: {"sim" if CRONUS_INCLUIR_MEDIA else "não"} (veredito da Célula 4.8).')
  print('Zero-Shot, LINDA, Lucas-Kanade e persistência serão reaproveitados das pastas da V1.')
  print('\nAgora rode, nesta ordem: Célula 4.4 (opcional), 12, 12.1, 13.1, 13.2. Depois, a Célula 13.7.')


def cronus_voltar_para_v1():
  """Desfaz a troca (para rodar de novo alguma célula com a V1)."""
  global gpt8, best_ckpt_path, ENV_TAG, RESULTS_ENV_DIR
  if globals().get('CRONUS', {}).get('ativo') != 'vq':
    print('A V1 já é o modelo ativo.')
    return
  gpt8, best_ckpt_path = gpt8_v1, best_ckpt_path_v1
  ENV_TAG, RESULTS_ENV_DIR = CRONUS['v1']['env_tag'], CRONUS['v1']['results_dir']
  for n in ('CRONUS_INCLUIR_MEDIA', 'CRONUS_REUSO_12', 'CRONUS_REUSO_13_1'):
    globals().pop(n, None)
  if CRONUS['v1'].get('cell12') is not None:
    globals()['CELL12_FULL_2025_RESULTS'] = CRONUS['v1']['cell12']
  CRONUS['ativo'] = 'v1'
  print('Modelo ativo voltou a ser o F-PluvIA-G (V1).')
