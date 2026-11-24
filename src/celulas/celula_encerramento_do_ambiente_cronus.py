# @title
# Célula (Encerramento do Ambiente Cronus): resumo do cronômetro, conferência do que ficou salvo no Drive e
# desligamento do ambiente de execução, para não consumir mais unidades de computação (UC) parado.
import os

ENCERRAR_AO_FINAL = True        # False = só mostra o resumo e não desliga
SALDO_UC_FINAL = None           # opcional: o saldo lido agora no painel "Recursos" do Colab (ex.: 512.3)

if SALDO_UC_FINAL is not None and callable(globals().get('registrar_saldo_uc')):
  registrar_saldo_uc(SALDO_UC_FINAL, 'fim da execução Cronus')
if callable(globals().get('resumo_cronometro')):
  resumo_cronometro()

conferir = {
    'VQGAN ajustada (Célula 4.6)': globals().get('ARQ_MELHOR'),
    'Registro do F-PluvIA-VQ (Célula 4.7)': globals().get('REGISTRO_VQT'),
    'Resultados da V1': (globals().get('CRONUS') or {}).get('v1', {}).get('results_dir'),
    'Resultados do F-PluvIA-VQ': (globals().get('CRONUS') or {}).get('vq', {}).get('results_dir'),
    'Backup completo (.zip)': globals().get('zip_path'),
}
print('=== Encerramento do ambiente Cronus ===')
for rotulo, caminho in conferir.items():
  ok = bool(caminho) and os.path.exists(str(caminho))
  print(f'  [{"ok" if ok else "--"}] {rotulo}: {caminho or "não definido nesta sessão"}')
if globals().get('CELL13_7_DUELO'):
  print(f"\nVencedor em 2025: {CELL13_7_DUELO['vencedor_2025']} — {CELL13_7_DUELO['veredito_2025']}")

if ENCERRAR_AO_FINAL:
  print('\nDesligando o ambiente de execução em seguida (o Drive já tem tudo; a sessão e a RAM serão descartadas).')
  from google.colab import runtime
  runtime.unassign()
