# @title
# Célula auxiliar (Cronus - Somente Leitura) — Cópia dos caches .npz da V1
# Lê de GPTCast_RMF (V1) e grava apenas em GPTCast_RMF_Cronus.
# Nunca sobrescreve: arquivo já existente no destino é mantido.
import os, glob, shutil, time

ORIGEM_V1 = '/content/drive/MyDrive/GPTCast_RMF'
DESTINO   = BASE_WORK_DIR  # definido na Célula 1 (GPTCast_RMF_Cronus)

if os.path.realpath(DESTINO) == os.path.realpath(ORIGEM_V1):
    raise RuntimeError('Destino igual à pasta da V1: abortado.')

# Subpastas de cache a copiar (só .npz). Ajuste se o listado abaixo mostrar outro nome.
SUBPASTAS = ['series_2019_2025_2km']
SO_LISTAR = True   # 1ª execução: só lista. Depois mude para False e rode de novo.

print('Pastas de primeiro nível na V1:')
for p in sorted(os.listdir(ORIGEM_V1)):
    if os.path.isdir(os.path.join(ORIGEM_V1, p)):
        n = len(glob.glob(os.path.join(ORIGEM_V1, p, '*.npz')))
        print(f'  {p:45s} {n:5d} .npz')

t0 = time.time()
for sub in SUBPASTAS:
    src_dir = os.path.join(ORIGEM_V1, sub)
    dst_dir = os.path.join(DESTINO, sub)
    arquivos = sorted(glob.glob(os.path.join(src_dir, '*.npz')))
    tam_gb = sum(os.path.getsize(a) for a in arquivos) / 1e9
    print(f'\n[{sub}] {len(arquivos)} arquivos .npz, {tam_gb:.2f} GB')
    if SO_LISTAR or not arquivos:
        continue
    os.makedirs(dst_dir, exist_ok=True)
    copiados = pulados = 0
    for a in arquivos:
        alvo = os.path.join(dst_dir, os.path.basename(a))
        if os.path.exists(alvo) and os.path.getsize(alvo) == os.path.getsize(a):
            pulados += 1
            continue
        shutil.copy2(a, alvo + '.part')
        os.replace(alvo + '.part', alvo)
        copiados += 1
    print(f'  copiados: {copiados} | já existiam: {pulados}')
    n_dst = len(glob.glob(os.path.join(dst_dir, '*.npz')))
    assert n_dst >= len(arquivos), f'Destino com {n_dst} de {len(arquivos)} arquivos.'

print(f'\nTempo: {(time.time() - t0) / 60:.1f} min'
      + (' (modo listagem: nada foi copiado)' if SO_LISTAR else ''))
