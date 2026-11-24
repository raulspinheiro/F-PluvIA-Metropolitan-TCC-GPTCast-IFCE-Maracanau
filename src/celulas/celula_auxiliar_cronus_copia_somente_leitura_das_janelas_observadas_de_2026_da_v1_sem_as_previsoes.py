# @title
# Célula auxiliar (Cronus) — Cópia somente-leitura das janelas observadas de 2026 da V1 (sem as previsões)
# Lê de GPTCast_RMF (V1) e grava só em GPTCast_RMF_Cronus. Nunca sobrescreve.
# Copia só as entradas observadas que a Célula 13.1 reaproveita como cache (RMF_2026_MM_s2.npz e caso_*_s1.npz).
# NÃO copia a pasta previsoes/ (previsões da V1): a 13.1 do Cronus gera as próprias com o modelo desta execução.
# NÃO copia os dia_corrente_* (a 13.1 procura o dia corrente pelo horário de agora; com a NASA PPS fora, ele é omitido).
import os, glob, shutil

ORIGEM = '/content/drive/MyDrive/GPTCast_RMF/series_2019_2025_2km/teste_cego_2026'
DESTINO = os.path.join(SERIES_CACHE_DIR, 'teste_cego_2026')      # SERIES_CACHE_DIR vem da Célula 1 (pasta do Cronus)
if '/GPTCast_RMF/' in DESTINO + '/' and '/GPTCast_RMF_Cronus/' not in DESTINO + '/':
    raise RuntimeError('Destino aponta para a pasta da V1: abortado.')
os.makedirs(DESTINO, exist_ok=True)

arquivos = sorted(glob.glob(os.path.join(ORIGEM, 'RMF_2026_*_s2.npz')) + glob.glob(os.path.join(ORIGEM, 'caso_*_s1.npz')))
copiados = pulados = 0
for a in arquivos:
    alvo = os.path.join(DESTINO, os.path.basename(a))
    if os.path.exists(alvo) and os.path.getsize(alvo) == os.path.getsize(a):
        pulados += 1
        continue
    shutil.copy2(a, alvo + '.part')
    os.replace(alvo + '.part', alvo)
    copiados += 1
    print(f'  copiado: {os.path.basename(a)} ({os.path.getsize(a) / 1e6:.1f} MB)')
print(f'[Cópia 2026] {copiados} copiados | {pulados} já existiam | destino: {DESTINO}')
print('Conteúdo do destino:', sorted(os.listdir(DESTINO)))
