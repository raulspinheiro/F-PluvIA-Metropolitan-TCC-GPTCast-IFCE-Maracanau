# @title
# Célula auxiliar (Cronus - Retomada do Fechamento, Somente Leitura): refaz, sem carregar modelos nem usar a GPU,
# as variáveis que as Células 13.6 e 13.7 deixam para as Células Full UMLs, Extra 2, Pseudo-vista, Deploy, Backup e
# Encerramento, depois que o ambiente de execução foi desconectado. Só lê o Drive: não grava, não move e não apaga
# nada. Rode depois das Células 0 e 1.
import os, json

# Etiqueta da execução Cronus que treinou e testou os modelos (G4). Fica fixa para a célula funcionar em qualquer
# GPU ou na CPU: a Célula 1 daria outra etiqueta se o Colab entregasse outro acelerador.
ENV_TAG_BASE = 'G4_RTX6000_Blackwell_95GB_Cronus'

faltam = [n for n in ('SERIES_CACHE_DIR', 'CHECKPOINTS_DIR', 'BASE_WORK_DIR') if n not in globals()]
if faltam:
  raise RuntimeError(f'Rode antes as Células 0 e 1. Faltam: {", ".join(faltam)}')
if 'GPTCast_RMF_Cronus' not in BASE_WORK_DIR:
  raise RuntimeError(f'BASE_WORK_DIR aponta para {BASE_WORK_DIR}; esta célula só serve ao ambiente Cronus.')
if globals().get('CRONUS', {}).get('ativo') == 'vq' and 'CRONUS_PASTAS_RESULTADOS' in globals():
  print('[Retomada] As variáveis do fechamento já existem nesta sessão; nada a fazer.')
else:
  if globals().get('ENV_TAG') != ENV_TAG_BASE:
    print(f'[Retomada] A Célula 1 deu a etiqueta {globals().get("ENV_TAG")}; usando a da execução: {ENV_TAG_BASE}.')

  tag_v1, tag_vq = ENV_TAG_BASE, f'{ENV_TAG_BASE}_VQ'
  ckpt_dir = os.path.join(CHECKPOINTS_DIR, tag_v1)
  res_v1 = os.path.join(SERIES_CACHE_DIR, f'resultados_{tag_v1}')
  res_vq = os.path.join(SERIES_CACHE_DIR, f'resultados_{tag_vq}')
  ARQ_MELHOR = os.path.join(ckpt_dir, 'vqgan_rmf', f'vqgan_rmf_2km_{tag_v1}_melhor.pt')
  REGISTRO_VQT = os.path.join(ckpt_dir, 'fpluvia_vq', f'registro_treino_fpluvia_vq_{tag_v1}.json')
  resumo_duelo = os.path.join(res_vq, f'cronus_duelo_resumo_{tag_vq}.json')

  exigidos = {'Resultados da V1': res_v1, 'Resultados do F-PluvIA-VQ': res_vq,
              'VQGAN ajustada (Célula 4.6)': ARQ_MELHOR, 'Registro do F-PluvIA-VQ (Célula 4.7)': REGISTRO_VQT,
              'Resumo do duelo (Célula 13.7)': resumo_duelo}
  ausentes = [f'{k}: {v}' for k, v in exigidos.items() if not os.path.exists(v)]
  if ausentes:
    raise RuntimeError('Não encontrado no Drive (nada foi alterado):\n  - ' + '\n  - '.join(ausentes))

  with open(REGISTRO_VQT) as f:
    best_ckpt_path_vq = os.path.join(ckpt_dir, 'fpluvia_vq', json.load(f)['checkpoint'])
  with open(resumo_duelo) as f:
    CELL13_7_DUELO = json.load(f)

  # Mesmo estado que as Células 13.6 e 13.7 deixam na sessão (modelo ativo: F-PluvIA-VQ)
  CRONUS = {'ativo': 'vq',
            'v1': {'env_tag': tag_v1, 'results_dir': res_v1},
            'vq': {'env_tag': tag_vq, 'results_dir': res_vq, 'checkpoint': os.path.basename(best_ckpt_path_vq)}}
  ENV_TAG, RESULTS_ENV_DIR, CHECKPOINTS_ENV_DIR = tag_vq, res_vq, ckpt_dir
  best_ckpt_path = best_ckpt_path_vq
  CRONUS_PASTAS_RESULTADOS = {tag_v1: res_v1}
  CRONUS_PASTAS_EXTRAS = {f'resultados_{tag_v1}': res_v1}

  print('=== Retomada do fechamento (somente leitura) ===')
  print(f'ENV_TAG ativo:   {ENV_TAG}')
  print(f'Resultados V1:   {res_v1}')
  print(f'Resultados VQ:   {res_vq}')
  print(f'Checkpoint VQ:   {os.path.basename(best_ckpt_path_vq)} '
        f'({"encontrado" if os.path.exists(best_ckpt_path_vq) else "NÃO encontrado"})')
  print(f'Vencedor (2025): {CELL13_7_DUELO.get("vencedor_2025")}')
  print('Modelos não foram carregados: o manifesto da Extra 2 sai sem a impressão digital do Transformer em memória, '
        'mas os arquivos .ckpt continuam com o SHA-256 listado.')
  print('\nAgora: Ctrl+S → Full UMLs → Extra 2 → Ctrl+S → Pseudo-vista → Ctrl+S → Deploy (SO_PREPARAR=True) '
        '→ Deploy → Backup → Encerramento.')
