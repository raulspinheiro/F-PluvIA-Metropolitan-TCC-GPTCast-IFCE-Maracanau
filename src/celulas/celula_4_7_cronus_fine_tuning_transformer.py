# @title
# Célula 4.7 (Cronus - Fine-tuning Transformer): F-PluvIA-VQ — o mesmo ajuste fino da Célula 4 (Transformer do GPTCast-8x8 do Zenodo, BF16,
# lr 5e-5, lote efetivo 16, até 15 épocas, parada antecipada com paciência 4, semente 42), agora sobre os tokens da
# VQGAN ajustada na Célula 4.6. Só a VQGAN muda em relação ao F-PluvIA-G (V1): é uma ablação limpa.
import copy
import datetime
import json
import os
import re
import time

import pandas as pd
import torch
import lightning.pytorch as pl_modern
import pytorch_lightning as pl_legacy
from gptcast.models import GPTCast

FORCE_RETRAIN_VQT = False      # True = refaz o ajuste do Transformer mesmo havendo checkpoint registrado

faltam = [n for n in ('md_rmf', 'gpt8_zeroshot', 'vq_rmf', 'CHECKPOINTS_ENV_DIR', 'RESULTS_ENV_DIR', 'ENV_TAG',
                      'BATCH_SIZE_AUTO', 'ACCUM_STEPS_AUTO', 'SEED', 'VAL_CROP', 'TOKENIZACAO_TREINO')
          if n not in globals()]
if faltam:
  raise RuntimeError(f'Rode antes as Células 1 a 4 e a 4.6. Faltam: {", ".join(faltam)}')
if not hasattr(GPTCast, '_orig_encode_to_z'):
  raise RuntimeError('A tokenização empilhada da Célula 4 não está ativa: rode a Célula 4 nesta sessão.')
if not torch.cuda.is_available():
  raise RuntimeError('A Célula 4.7 treina o Transformer: ative a GPU.')
DEVICE = 'cuda'

PASTA_VQT = os.path.join(CHECKPOINTS_ENV_DIR, 'fpluvia_vq')
os.makedirs(PASTA_VQT, exist_ok=True)
REGISTRO_VQT = os.path.join(PASTA_VQT, f'registro_treino_fpluvia_vq_{ENV_TAG}.json')


def _montar_vq():
  """GPTCast-8x8 do Zenodo (mesmo ponto de partida da V1) com a VQGAN da Célula 4.6 no lugar da original."""
  m = copy.deepcopy(gpt8_zeroshot).to(DEVICE)
  m.first_stage_model.load_state_dict(vq_rmf.state_dict())
  for p in m.first_stage_model.parameters():
    p.requires_grad = False
  return m


def _carregar(model, path):
  state = torch.load(path, map_location=DEVICE)['state_dict']
  res = model.load_state_dict(state, strict=False)
  if any(k.startswith('transformer.') for k in res.missing_keys):
    raise RuntimeError(f'{os.path.basename(path)} não tem os pesos do Transformer: abortando.')
  # a VQGAN do checkpoint tem de ser a da Célula 4.6 (o Lightning salva o modelo inteiro)
  for k, v in vq_rmf.state_dict().items():
    if not torch.equal(model.first_stage_model.state_dict()[k].to(v.device), v):
      raise RuntimeError(f'A VQGAN do checkpoint difere da Célula 4.6 em {k}: o checkpoint é de outro ajuste.')
  return model


gpt8_vq = _montar_vq()
rec_vqt = None
if not FORCE_RETRAIN_VQT and os.path.exists(REGISTRO_VQT):
  with open(REGISTRO_VQT) as f:
    rec_vqt = json.load(f)
  ck = os.path.join(PASTA_VQT, rec_vqt['checkpoint'])
  if os.path.isfile(ck):
    gpt8_vq = _carregar(gpt8_vq, ck).eval()
    print(f'[Célula 4.7] Checkpoint do F-PluvIA-VQ carregado: {rec_vqt["checkpoint"]} '
          f'(treino de {rec_vqt["data_treino_utc"]}).')
  else:
    rec_vqt = None

if rec_vqt is None:
  pl_active = pl_modern if issubclass(GPTCast, pl_modern.LightningModule) else pl_legacy
  pl_active.seed_everything(SEED, workers=True)
  precision_mode = 'bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed'
  gpt8_vq.train()
  for p in gpt8_vq.transformer.parameters():
    p.requires_grad = True
  target_lr = 5e-5
  gpt8_vq.base_learning_rate = target_lr / (BATCH_SIZE_AUTO * ACCUM_STEPS_AUTO)
  gpt8_vq.learning_rate = target_lr
  run_tag = datetime.datetime.utcnow().strftime('%Y%m%dT%H%MZ')
  ck_cb = pl_active.callbacks.ModelCheckpoint(
      dirpath=PASTA_VQT, filename=f'fpluvia_vq_rmf_2km_{ENV_TAG}_{run_tag}_best-{{epoch:02d}}-{{val/loss:.4f}}',
      auto_insert_metric_name=False, monitor='val/loss', mode='min', save_top_k=1, save_last=True)
  es_cb = pl_active.callbacks.EarlyStopping(monitor='val/loss', patience=4, mode='min', verbose=True)
  logger = pl_active.loggers.CSVLogger(save_dir=RESULTS_ENV_DIR, name=f'curva_treino_fpluvia_vq_{ENV_TAG}', version=run_tag)
  trainer = pl_active.Trainer(accelerator='gpu', devices=1, max_epochs=15, precision=precision_mode,
                              accumulate_grad_batches=ACCUM_STEPS_AUTO, callbacks=[ck_cb, es_cb], logger=logger,
                              log_every_n_steps=5, deterministic=False)
  print(f'[Célula 4.7] Ajuste do Transformer sobre os tokens da VQGAN nova: lote {BATCH_SIZE_AUTO} x acúmulo '
        f'{ACCUM_STEPS_AUTO}, {precision_mode}, lr {target_lr}, semente {SEED} (mesma receita da Célula 4).')
  torch.cuda.reset_peak_memory_stats(0)
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  trainer.fit(gpt8_vq, datamodule=md_rmf)
  torch.cuda.synchronize()
  t_fit = time.perf_counter() - t0
  best = ck_cb.best_model_path
  if not best or not os.path.exists(best):
    raise RuntimeError('O ajuste terminou sem checkpoint de melhor val/loss.')
  m = re.search(r'best-(\d+)-', os.path.basename(best))
  rec_vqt = {
      'data_treino_utc': run_tag, 'checkpoint': os.path.basename(best),
      'epocas_executadas': int(trainer.current_epoch), 'melhor_epoca': int(m.group(1)) if m else None,
      'melhor_val_loss_2024': round(float(ck_cb.best_model_score.cpu().item()), 4),
      'observacao_val_loss': 'não comparável com a da V1: os tokens (o vocabulário do alvo) são outros',
      'finetuning_tempo_total_min': round(t_fit / 60, 2), 'precisao_numerica': precision_mode,
      'batch_size_treino': BATCH_SIZE_AUTO, 'acumulo_gradiente': ACCUM_STEPS_AUTO, 'learning_rate_efetiva': target_lr,
      'semente_treino': SEED, 'corte_validacao': VAL_CROP, 'tokenizacao_treino': TOKENIZACAO_TREINO,
      'vqgan': (globals().get('CELL4_6_VQGAN') or {}).get('arquivo_melhor'),
      'pico_vram_treino_gb': round(torch.cuda.max_memory_allocated(0) / 1024**3, 2),
      'curva_por_epoca_csv': os.path.join(logger.log_dir, 'metrics.csv'), 'gpu': torch.cuda.get_device_name(0),
  }
  with open(REGISTRO_VQT, 'w') as f:
    json.dump(rec_vqt, f, indent=2, ensure_ascii=False)
  gpt8_vq = _carregar(gpt8_vq, best)

gpt8_vq = gpt8_vq.to(DEVICE).eval()
best_ckpt_path_vq = os.path.join(PASTA_VQT, rec_vqt['checkpoint'])
CELL4_7_FPLUVIA_VQ = rec_vqt
print('\n=== Célula 4.7: F-PluvIA-VQ pronto (gpt8_vq) ===')
display(pd.DataFrame([rec_vqt]).T.rename(columns={0: 'valor'}))
print('O F-PluvIA-G (V1) continua em gpt8, e o Zero-Shot em gpt8_zeroshot. Próximo passo: Célula 4.8.')
