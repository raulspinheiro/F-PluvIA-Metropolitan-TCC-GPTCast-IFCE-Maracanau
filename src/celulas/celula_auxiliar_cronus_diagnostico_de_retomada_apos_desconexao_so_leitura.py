# @title
# Célula auxiliar (Cronus) — Diagnóstico de retomada após desconexão (só leitura)
# Não grava nem apaga nada. Rode logo depois de montar o Drive (Célula 0 ou 1).
import os, glob, json, time
import pandas as pd
from google.colab import drive
if not os.path.ismount('/content/drive'):
    drive.mount('/content/drive')

BASE = '/content/drive/MyDrive/GPTCast_RMF_Cronus'
HORAS = 14   # janela: arquivos alterados nas últimas HORAS horas
agora = time.time()

def ler_json(padrao):
    achados = sorted(glob.glob(os.path.join(BASE, '**', padrao), recursive=True))
    for a in achados:
        print(f'\n--- {a.replace(BASE, "…")}')
        try:
            with open(a) as f:
                d = json.load(f)
            for k, v in d.items():
                if not isinstance(v, (list, dict)):
                    print(f'  {k}: {v}')
        except Exception as e:
            print('  (não foi possível ler)', e)
    if not achados:
        print(f'\n--- nenhum arquivo {padrao}')

print('=== 1. Arquivos alterados nas últimas', HORAS, 'h (mais recentes primeiro) ===')
recentes = []
for raiz, _, arqs in os.walk(BASE):
    for a in arqs:
        p = os.path.join(raiz, a)
        try:
            m = os.path.getmtime(p)
        except OSError:
            continue
        if agora - m < HORAS * 3600:
            recentes.append((m, p, os.path.getsize(p)))
for m, p, s in sorted(recentes, reverse=True)[:40]:
    print(f'{time.strftime("%d/%m %H:%M", time.localtime(m - 3 * 3600))}  {s / 1e6:9.1f} MB  {p.replace(BASE, "…")}')

print('\n=== 2. Célula 4.6 (VQGAN) ===')
for padrao in ('vqgan_rmf_2km_*_melhor.pt', 'vqgan_rmf_2km_*_ultima.pt'):
    for a in glob.glob(os.path.join(BASE, '**', padrao), recursive=True):
        print(f'{padrao.split("_")[-1]:10s} {os.path.getsize(a) / 1e6:8.1f} MB  '
              f'{time.strftime("%d/%m %H:%M", time.localtime(os.path.getmtime(a) - 3 * 3600))}')
ler_json('registro_vqgan_*.json')
for a in glob.glob(os.path.join(BASE, '**', 'curva_ajuste_vqgan_*.csv'), recursive=True):
    print('\nCurva por época:')
    print(pd.read_csv(a).to_string(index=False))

print('\n=== 3. Célula 4.7 (Transformer do F-PluvIA-VQ) ===')
ler_json('registro_treino_fpluvia_vq_*.json')

print('\n=== 4. Célula 4.8 (validação 2024, V1 × VQ) ===')
ler_json('cronus_previsao_v1_vs_vq_validacao2024_*.json')

print('\n=== 5. Cronômetro (últimas células registradas) ===')
for a in glob.glob(os.path.join(BASE, 'cronometro', '*celulas*.csv')):
    df = pd.read_csv(a)
    print(df.tail(15).to_string(index=False))
