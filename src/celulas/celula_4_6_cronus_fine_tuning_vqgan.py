# @title
# Célula 4.6 (Cronus - Fine-tuning VQGAN): Ajuste da VQGAN do GPTCast-8x8 à RMF — codificador e decodificador treináveis, dicionário
# (codebook) de 1.024 códigos fixo, treino 2019–2023, escolha da época na validação de 2024, FP32 + TF32, sem
# discriminador (GAN). Teste 2025 e teste cego 2026 intocados. Retomável: se a sessão cair, continua da última época.
import copy
import datetime
import json
import os
import time
import types

import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------------ 1. parâmetros
SEED_VQ = 42
QUADROS_EMPILHADOS = 8         # 8 quadros na vertical (1.024 × 128 px → 64 × 8 = 512 tokens), como no treino do Transformer
LOTE_VQ = 8                    # pilhas por lote (8 × 8 = 64 quadros)
LR_VQ = 1e-5                   # baixa: parte da VQGAN original e só a adapta (o GPTCast treinou do zero com 4,5e-6 × lote)
BETAS_VQ = (0.5, 0.9)          # Adam, como no treino original da VQGAN
MAX_EPOCAS_VQ = 20
PACIENCIA_VQ = 3               # épocas sem melhora na validação de 2024 antes de parar
TEMPO_MAX_H_VQ = 5.0           # orçamento de relógio para o ajuste (h); ao estourar, fica a melhor época até ali
PESO_PIXEL, PESO_PERCEPTUAL = 10.0, 0.1   # mesmos pesos da VQGAN "vae_mwae" do GPTCast (perda MWAE + LPIPS)
PESO_QUADRO_ISOLADO = 1.0      # reconstrução de cada quadro decodificado sozinho, como no forecast()
CONGELAR_DICIONARIO = True     # mantém o significado dos 1.024 tokens (o Transformer continua comparável)
N_VAL_VQ = None                # None = todas as janelas de validação de 2024
FORCE_RETRAIN_VQ = False       # True = descarta o ajuste anterior desta pasta e treina de novo

faltam = [n for n in ('md_rmf', 'gpt8_zeroshot', 'CHECKPOINTS_ENV_DIR', 'RESULTS_ENV_DIR', 'ENV_TAG') if n not in globals()]
if faltam:
  raise RuntimeError(f'Rode antes as Células 1 a 4. Faltam: {", ".join(faltam)}')
if not torch.cuda.is_available():
  raise RuntimeError('A Célula 4.6 treina a VQGAN: ative a GPU (Ambiente de execução -> GPU).')
DEVICE = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

PASTA_VQ = os.path.join(CHECKPOINTS_ENV_DIR, 'vqgan_rmf')
os.makedirs(PASTA_VQ, exist_ok=True)
ARQ_MELHOR = os.path.join(PASTA_VQ, f'vqgan_rmf_2km_{ENV_TAG}_melhor.pt')
ARQ_ULTIMA = os.path.join(PASTA_VQ, f'vqgan_rmf_2km_{ENV_TAG}_ultima.pt')
ARQ_REGISTRO = os.path.join(PASTA_VQ, f'registro_vqgan_{ENV_TAG}.json')
ARQ_CURVA = os.path.join(RESULTS_ENV_DIR, f'curva_ajuste_vqgan_{ENV_TAG}.csv')
if FORCE_RETRAIN_VQ:
  for _f in (ARQ_MELHOR, ARQ_ULTIMA, ARQ_REGISTRO, ARQ_CURVA):
    if os.path.exists(_f):
      os.replace(_f, _f + f'.descartado_{datetime.datetime.utcnow():%Y%m%dT%H%MZ}')

_cron = globals().get('etapa')   # Célula 0 (cronômetro), se estiver na sessão
_ctx_cron = (lambda nome, cat: _cron(nome, categoria=cat)) if callable(_cron) else None

# ------------------------------------------------------------------ 2. VQGAN treinável
# O GPTCast troca o train() da VQGAN por uma função que não faz nada (disabled_train) e congela os pesos.
# Aqui se trabalha numa CÓPIA: o Zero-Shot e o F-PluvIA (V1) continuam com a VQGAN original intacta.
vq_orig = gpt8_zeroshot.first_stage_model
vq = copy.deepcopy(vq_orig).to(DEVICE).float()
vq.train = types.MethodType(torch.nn.Module.train, vq)
for p in vq.parameters():
  p.requires_grad = True
if CONGELAR_DICIONARIO:
  vq.quantize.embedding.weight.requires_grad = False
treinaveis = [p for p in vq.parameters() if p.requires_grad]
N_TREINAVEIS = sum(p.numel() for p in treinaveis)
N_TOTAL_VQ = sum(p.numel() for p in vq.parameters())

# Perda de reconstrução do próprio GPTCast (MWAE ponderada + LPIPS). Sem discriminador: a disputa gerador ×
# discriminador fica instável e não caberia no prazo. Se o LPIPS não baixar os pesos, segue só com a MWAE.
from gptcast.models.components.vaegan.losses.discriminator import mwae
try:
  from gptcast.models.components.vaegan.losses import LPIPS
  lpips = LPIPS().eval().to(DEVICE)
  for p in lpips.parameters():
    p.requires_grad = False
except Exception as e:  # noqa: BLE001
  lpips, PESO_PERCEPTUAL = None, 0.0
  print(f'[Aviso Célula 4.6] LPIPS indisponível ({type(e).__name__}: {e}); perda só com MWAE.')


def _norm(dbz):
  """dBZ [0, 60] -> [-1, 1], como no treino e na previsão do GPTCast."""
  return np.clip(np.nan_to_num(dbz, nan=0.0), 0.0, 60.0) / 30.0 - 1.0


def _pilhas(seqs, rng):
  """(N, 11, H, W) dBZ -> (N, 1, 8·H, W): 8 quadros consecutivos empilhados, início sorteado (0 a 3)."""
  n, t = seqs.shape[0], seqs.shape[1]
  ini = rng.integers(0, t - QUADROS_EMPILHADOS + 1, size=n)
  out = np.stack([seqs[i, s:s + QUADROS_EMPILHADOS] for i, s in zip(range(n), ini)])
  return out.reshape(n, 1, QUADROS_EMPILHADOS * out.shape[-2], out.shape[-1])


def _perda(x):
  """Pilha inteira (codifica junto, como no forecast) + cada quadro decodificado sozinho (como no forecast)."""
  quant, dif, _ = vq.encode(x)
  rec = vq.decode(quant)
  l_pix = PESO_PIXEL * mwae(x, rec).mean()
  l_per = PESO_PERCEPTUAL * lpips(x, rec).mean() if lpips is not None and PESO_PERCEPTUAL > 0 else x.new_zeros(())
  # quadros isolados: (B, C, 8·h, w) -> (B·8, C, h, w), decodificados um a um
  b, c, hh, w = quant.shape
  q_iso = quant.reshape(b, c, QUADROS_EMPILHADOS, hh // QUADROS_EMPILHADOS, w).permute(0, 2, 1, 3, 4)
  q_iso = q_iso.reshape(b * QUADROS_EMPILHADOS, c, hh // QUADROS_EMPILHADOS, w)
  rec_iso = vq.decode(q_iso)
  x_iso = x.reshape(b, 1, QUADROS_EMPILHADOS, x.shape[-2] // QUADROS_EMPILHADOS, x.shape[-1]).permute(0, 2, 1, 3, 4)
  x_iso = x_iso.reshape(b * QUADROS_EMPILHADOS, 1, x.shape[-2] // QUADROS_EMPILHADOS, x.shape[-1])
  l_iso = PESO_QUADRO_ISOLADO * PESO_PIXEL * mwae(x_iso, rec_iso).mean()
  l_vq = dif.mean()
  return l_pix + l_per + l_iso + l_vq, {'pixel': float(l_pix), 'perceptual': float(l_per),
                                        'quadro_isolado': float(l_iso), 'quantizacao': float(l_vq)}


# ------------------------------------------------------------------ 3. dados (só treino e validação)
tr_seq = md_rmf.splits['train'].sequences_dbz
va_seq = md_rmf.splits['val'].sequences_dbz
if N_VAL_VQ:
  va_seq = va_seq[:N_VAL_VQ]
rng_val = np.random.default_rng(SEED_VQ + 1)
X_VAL = _pilhas(np.asarray(va_seq), rng_val).astype(np.float32)     # pilhas de validação fixas


@torch.no_grad()
def _avaliar(modelo, X):
  modelo_estava = modelo.training
  modelo.eval()
  tot, n = 0.0, 0
  for i in range(0, len(X), LOTE_VQ):
    x = torch.from_numpy(_norm(X[i:i + LOTE_VQ])).to(DEVICE)
    if modelo is vq:
      l, _ = _perda(x)
    else:  # VQGAN original, mesma perda
      quant, dif, _ = modelo.encode(x)
      rec = modelo.decode(quant)
      l = PESO_PIXEL * mwae(x, rec).mean() + dif.mean()
      if lpips is not None and PESO_PERCEPTUAL > 0:
        l = l + PESO_PERCEPTUAL * lpips(x, rec).mean()
      b, c, hh, w = quant.shape
      q_iso = quant.reshape(b, c, QUADROS_EMPILHADOS, hh // QUADROS_EMPILHADOS, w).permute(0, 2, 1, 3, 4).reshape(
          b * QUADROS_EMPILHADOS, c, hh // QUADROS_EMPILHADOS, w)
      x_iso = x.reshape(b, 1, QUADROS_EMPILHADOS, x.shape[-2] // QUADROS_EMPILHADOS, x.shape[-1]).permute(
          0, 2, 1, 3, 4).reshape(b * QUADROS_EMPILHADOS, 1, x.shape[-2] // QUADROS_EMPILHADOS, x.shape[-1])
      l = l + PESO_QUADRO_ISOLADO * PESO_PIXEL * mwae(x_iso, modelo.decode(q_iso)).mean()
    tot += float(l) * len(x)
    n += len(x)
  if modelo_estava:
    modelo.train()
  return tot / max(n, 1)


# ------------------------------------------------------------------ 4. treino retomável
opt = torch.optim.Adam(treinaveis, lr=LR_VQ, betas=BETAS_VQ)
estado = {'epoca': 0, 'melhor_val': None, 'melhor_epoca': None, 'sem_melhora': 0, 'tempo_s': 0.0, 'curva': []}
if os.path.exists(ARQ_ULTIMA):
  ck = torch.load(ARQ_ULTIMA, map_location=DEVICE)
  vq.load_state_dict(ck['vq'])
  opt.load_state_dict(ck['opt'])
  estado = ck['estado']
  print(f'[Célula 4.6] Retomando depois da época {estado["epoca"]} (melhor val {estado["melhor_val"]:.5f} '
        f'na época {estado["melhor_epoca"]}).')

val_original = estado.get('val_vqgan_original')
if val_original is None:
  val_original = _avaliar(vq_orig.to(DEVICE), X_VAL)
  estado['val_vqgan_original'] = val_original
print(f'[Célula 4.6] VQGAN: {N_TOTAL_VQ:,} parâmetros, {N_TREINAVEIS:,} treináveis '.replace(',', '.') +
      f'({"dicionário fixo" if CONGELAR_DICIONARIO else "dicionário treinável"}) | {len(tr_seq)} janelas de treino, '
      f'{len(X_VAL)} pilhas de validação | perda de validação da VQGAN original: {val_original:.5f}')

terminou = estado.get('terminou', False)
t_ini_sessao = time.perf_counter()
torch.manual_seed(SEED_VQ)
while not terminou and estado['epoca'] < MAX_EPOCAS_VQ:
  ep = estado['epoca'] + 1
  rng = np.random.default_rng(SEED_VQ + ep)
  ordem = rng.permutation(len(tr_seq))
  vq.train()
  t0 = time.perf_counter()
  soma, n, comp = 0.0, 0, {}
  for i in range(0, len(ordem), LOTE_VQ):
    idx = np.sort(ordem[i:i + LOTE_VQ])
    x = torch.from_numpy(_norm(_pilhas(np.asarray(tr_seq[idx]), rng)).astype(np.float32)).to(DEVICE)
    perda, partes = _perda(x)
    opt.zero_grad(set_to_none=True)
    perda.backward()
    torch.nn.utils.clip_grad_norm_(treinaveis, 1.0)
    opt.step()
    soma += float(perda) * len(x)
    n += len(x)
    for k, v in partes.items():
      comp[k] = comp.get(k, 0.0) + v * len(x)
    if (i // LOTE_VQ) % 200 == 0:
      print(f'  época {ep} | {min(i + LOTE_VQ, len(ordem))}/{len(ordem)} | perda {soma / n:.5f}', flush=True)
  val = _avaliar(vq, X_VAL)
  dt = time.perf_counter() - t0
  estado['tempo_s'] += dt
  estado['curva'].append({'epoca': ep, 'perda_treino': soma / n, 'perda_val_2024': val, 'tempo_s': dt,
                          **{f'treino_{k}': v / n for k, v in comp.items()}})
  melhorou = estado['melhor_val'] is None or val < estado['melhor_val']
  if melhorou:
    estado.update(melhor_val=val, melhor_epoca=ep, sem_melhora=0)
    torch.save({k: v.detach().cpu() for k, v in vq.state_dict().items()}, ARQ_MELHOR)
  else:
    estado['sem_melhora'] += 1
  estado['epoca'] = ep
  horas = (time.perf_counter() - t_ini_sessao) / 3600
  terminou = estado['sem_melhora'] >= PACIENCIA_VQ or horas >= TEMPO_MAX_H_VQ
  estado['terminou'] = terminou or ep >= MAX_EPOCAS_VQ
  torch.save({'vq': vq.state_dict(), 'opt': opt.state_dict(), 'estado': estado}, ARQ_ULTIMA)
  pd.DataFrame(estado['curva']).to_csv(ARQ_CURVA, index=False)
  situacao = 'melhor até aqui' if melhorou else f"sem melhora há {estado['sem_melhora']} época(s)"
  print(f'[Célula 4.6] época {ep}: treino {soma / n:.5f} | val 2024 {val:.5f} ({situacao}) | {dt / 60:.1f} min',
        flush=True)

# ------------------------------------------------------------------ 5. melhor época e registro
vq_rmf = copy.deepcopy(vq_orig).to(DEVICE)              # mesma classe e mesmo disabled_train da original
vq_rmf.load_state_dict(torch.load(ARQ_MELHOR, map_location=DEVICE))
for p in vq_rmf.parameters():
  p.requires_grad = False


@torch.no_grad()
def _fracao_tokens_iguais(X):
  iguais, total = 0, 0
  for i in range(0, len(X), LOTE_VQ):
    x = torch.from_numpy(_norm(X[i:i + LOTE_VQ])).to(DEVICE)
    a = vq_orig.encode(x)[2].reshape(-1)
    b = vq_rmf.encode(x)[2].reshape(-1)
    iguais += int((a == b).sum())
    total += a.numel()
  return iguais / max(total, 1)


FRAC_TOKENS_IGUAIS_2024 = _fracao_tokens_iguais(X_VAL)
registro_vq = {
    'ambiente': ENV_TAG, 'data_utc': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M'),
    'arquivo_melhor': os.path.basename(ARQ_MELHOR), 'melhor_epoca': estado['melhor_epoca'],
    'epocas_executadas': estado['epoca'], 'perda_val_2024_melhor': round(estado['melhor_val'], 6),
    'perda_val_2024_vqgan_original': round(val_original, 6),
    'reducao_perda_val_pct': round(100 * (1 - estado['melhor_val'] / val_original), 2),
    'fracao_tokens_iguais_val_2024': round(FRAC_TOKENS_IGUAIS_2024, 4),
    'tempo_total_ajuste_min': round(estado['tempo_s'] / 60, 2),
    'parametros_vqgan': N_TOTAL_VQ, 'parametros_treinados': N_TREINAVEIS,
    'dicionario_congelado': CONGELAR_DICIONARIO, 'lr': LR_VQ, 'lote_pilhas': LOTE_VQ,
    'quadros_por_pilha': QUADROS_EMPILHADOS, 'pesos_perda': {'pixel_mwae': PESO_PIXEL, 'lpips': PESO_PERCEPTUAL,
                                                            'quadro_isolado': PESO_QUADRO_ISOLADO, 'vq': 1.0},
    'precisao': 'float32 com TF32 (sem bf16/fp16, sem discriminador)', 'semente': SEED_VQ,
    'dados': 'treino 2019–2023 (ajuste) e validação 2024 (escolha da época); 2025 e 2026 não usados',
    'gpu': torch.cuda.get_device_name(0),
}
with open(ARQ_REGISTRO, 'w') as f:
  json.dump(registro_vq, f, indent=2, ensure_ascii=False)
CELL4_6_VQGAN = registro_vq

print('\n=== Célula 4.6: VQGAN ajustada à RMF ===')
display(pd.DataFrame([registro_vq]).T.rename(columns={0: 'valor'}))
print(f'Curva por época: {ARQ_CURVA}')
print('Próximo passo: Célula 4.7 (Transformer reajustado sobre os tokens da VQGAN nova).')
