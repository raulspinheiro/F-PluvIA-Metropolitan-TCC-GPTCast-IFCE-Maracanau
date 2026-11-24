# @title
# Célula 0: Cronômetro de Ponta a Ponta (F-PluvIA Metropolitan) — tempo de máquina e unidades de computação
# Cole no início do notebook e execute antes de qualquer outra célula.
#
# Registra, em CSV no Drive, gravado a cada etapa (nada se perde se a sessão cair):
#   (1) toda célula executada, automaticamente (início, fim, duração);
#   (2) etapas nomeadas, com o bloco  `with etapa("nome", categoria="..."):`
#   (3) o saldo de unidades de computação (UC) lido por você no painel do Colab.
# Sem credenciais: nada aqui lê Secrets, tokens ou senhas.
import csv
import os
import socket
import time
import uuid
from contextlib import contextmanager
# Apelidos privados: outras células fazem `import datetime` e `import time`, que trocariam os nomes globais
# usados pelos registros automáticos desta célula (o erro "module 'datetime' has no attribute 'now'").
from datetime import datetime as _dt_cron, timezone as _tz_cron
import time as _time_cron

# ---- Configuração -----------------------------------------------------------
PASTA_CRONOMETRO = os.environ.get(
    "FPLUVIA_CRONOMETRO_DIR", "/content/drive/MyDrive/GPTCast_RMF_Cronus/cronometro")  # Cronus: pasta própria
TAXA_UC_PADRAO = 10.05  # painel "Recursos" do Colab (G4, 22/11/2026)
_taxa_txt = os.environ.get("FPLUVIA_TAXA_UC_H", "").strip().replace(",", ".")
try:
    TAXA_UC_POR_HORA = float(_taxa_txt) if _taxa_txt else TAXA_UC_PADRAO
except ValueError:  # valor inválido na variável de ambiente (ex.: texto de exemplo): usa o padrão e avisa
    print(f"[cronômetro] FPLUVIA_TAXA_UC_H inválida ({_taxa_txt!r}); usando {TAXA_UC_PADRAO} UC/h.")
    os.environ.pop("FPLUVIA_TAXA_UC_H", None)
    TAXA_UC_POR_HORA = TAXA_UC_PADRAO
SESSAO_ID = _dt_cron.now(_tz_cron.utc).strftime("%Y%m%dT%H%MZ") + "_" + uuid.uuid4().hex[:6]

# MODO AMOSTRA: "1" = cada etapa roda só uma amostra (para medir a CPU e extrapolar).
#               "0" = execução completa (use na GPU).
MODO_AMOSTRA = os.environ.get("FPLUVIA_MODO_AMOSTRA", "0") == "1"
# Tamanho da amostra por categoria, na unidade da etapa (passos, janelas...).
AMOSTRA_N = {"3_treino": 20, "4_validacao": 5, "5_teste_FR": 5, "6_teste_LR": 5}

# Monta o Drive AQUI, antes de criar a pasta: criar /content/drive/... sem o Drive montado
# impediria a montagem da Célula 1 ("Mountpoint must not already contain files").
# A Célula 1 usa drive.mount('/content/drive') sem force_remount, então vira só um aviso.
if PASTA_CRONOMETRO.startswith("/content/drive") and not os.path.ismount("/content/drive"):
    try:
        from google.colab import drive
        drive.mount("/content/drive")  # escolha aqui a conta do Drive desta execução
    except ImportError:
        raise RuntimeError("Fora do Colab: defina FPLUVIA_CRONOMETRO_DIR para uma pasta local.")

os.makedirs(PASTA_CRONOMETRO, exist_ok=True)
ARQ_ETAPAS = os.path.join(PASTA_CRONOMETRO, "cronometro_etapas.csv")
ARQ_CELULAS = os.path.join(PASTA_CRONOMETRO, "cronometro_celulas.csv")
ARQ_SALDO = os.path.join(PASTA_CRONOMETRO, "cronometro_saldo_uc.csv")

try:
    import torch
    _CUDA = torch.cuda.is_available()
except Exception:  # torch ausente ou sem GPU
    torch, _CUDA = None, False

GPU_NOME = torch.cuda.get_device_name(0) if _CUDA else f"CPU ({os.cpu_count()} núcleos lógicos)"
MODO = "amostra" if MODO_AMOSTRA else "completo"


def limitar(seq, categoria):
    """No modo amostra, devolve só os primeiros AMOSTRA_N[categoria] itens de `seq`.
    Uso:  for janela in limitar(janelas_teste, "5_teste_FR"): ...
    Fora do modo amostra (ou categoria sem limite), devolve `seq` inteira."""
    n = AMOSTRA_N.get(categoria) if MODO_AMOSTRA else None
    if n is None:
        return seq
    try:
        return seq[:n]
    except TypeError:  # geradores e iteradores
        import itertools
        return list(itertools.islice(seq, n))


def _agora():
    return _dt_cron.now(_tz_cron.utc).isoformat(timespec="seconds")


def _gravar(arquivo, linha):
    novo = not os.path.exists(arquivo)
    with open(arquivo, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linha.keys()))
        if novo:
            w.writeheader()
        w.writerow(linha)


# ---- (2) Etapas nomeadas ----------------------------------------------------
@contextmanager
def etapa(nome, categoria="outros", quantidade=None, unidade=""):
    """Uso:
        with etapa("ETL Final Run 2019-2025", categoria="1_extracao", quantidade=28, unidade="meses"):
            ...código da etapa...
    """
    if _CUDA:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    inicio_iso, t0 = _agora(), _time_cron.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException as e:  # registra também interrupções e erros
        status = f"erro: {type(e).__name__}"
        raise
    finally:
        if _CUDA:
            torch.cuda.synchronize()
        dur = _time_cron.perf_counter() - t0
        pico_gb = torch.cuda.max_memory_allocated() / 1e9 if _CUDA else 0.0
        linha = {
            "sessao": SESSAO_ID, "maquina": socket.gethostname(), "gpu": GPU_NOME,
            "modo": MODO, "taxa_uc_h": TAXA_UC_POR_HORA,
            "categoria": categoria, "etapa": nome, "status": status,
            "inicio_utc": inicio_iso, "fim_utc": _agora(),
            "duracao_s": round(dur, 3), "duracao_min": round(dur / 60, 3),
            "pico_vram_gb": round(pico_gb, 3),
            "quantidade": quantidade if quantidade is not None else "", "unidade": unidade,
            "s_por_unidade": round(dur / quantidade, 4) if quantidade else "",
            "uc_estimadas": round(dur / 3600 * TAXA_UC_POR_HORA, 3),
        }
        _gravar(ARQ_ETAPAS, linha)
        print(f"[cronômetro] {nome}: {dur/60:.1f} min | {linha['uc_estimadas']} UC (est.) | {status}")


# ---- (1) Toda célula, automaticamente ----------------------------------------
_estado_celula = {}


def _antes_celula(info=None):
    _estado_celula["t0"] = _time_cron.perf_counter()
    _estado_celula["inicio"] = _agora()
    src = getattr(info, "raw_cell", "") or ""
    # Primeira linha útil: pula o "# @title" que abre quase todas as células deste notebook
    _estado_celula["titulo"] = next((l.strip()[:90] for l in src.splitlines()
                                     if l.strip() and not l.strip().lower().startswith(("# @title", "#@title"))), "")


# Categoria de cada célula do notebook, pelo título (as etapas do pipeline sem precisar de `with etapa(...)`)
CATEGORIAS_POR_TITULO = [
    ("Célula 2:", "1_extracao"), ("Célula 13.1", "6_teste_LR"),
    ("Célula 3:", "2_dataset"),
    ("Célula 4.6", "3_treino"), ("Célula 4.7", "3_treino"), ("Célula 4.3", "3_treino"), ("Célula 4:", "3_treino"),
    ("Célula 4.1", "4_validacao"), ("Célula 4.2", "4_validacao"), ("Célula 4.4", "4_validacao"),
    ("Célula 4.5", "4_validacao"), ("Célula 4.8", "4_validacao"),
    ("Célula 12:", "5_teste_FR"),
    ("Célula 12.1", "8_metricas"), ("Célula 13.2", "8_metricas"), ("Célula 13.7", "8_metricas"),
    ("Célula 13.3", "8_metricas"), ("Célula 13.4", "8_metricas"),
]


def _categoria(titulo):
    for chave, cat in CATEGORIAS_POR_TITULO:
        if chave in titulo:
            return cat
    return "outros"


def _depois_celula(result=None):
    if "t0" not in _estado_celula:
        return
    dur = _time_cron.perf_counter() - _estado_celula.pop("t0")
    erro = getattr(result, "error_in_exec", None) or getattr(result, "error_before_exec", None)
    _gravar(ARQ_CELULAS, {
        "sessao": SESSAO_ID, "gpu": GPU_NOME, "categoria": _categoria(_estado_celula.get("titulo", "")),
        "inicio_utc": _estado_celula.get("inicio"),
        "fim_utc": _agora(), "duracao_s": round(dur, 3),
        "status": "ok" if erro is None else f"erro: {type(erro).__name__}",
        "primeira_linha": _estado_celula.get("titulo", ""),
        "uc_estimadas": round(dur / 3600 * TAXA_UC_POR_HORA, 4),
    })


try:
    _ip = get_ipython()  # noqa: F821 (existe no Colab/Jupyter)
    for ev, fn in (("pre_run_cell", _antes_celula), ("post_run_cell", _depois_celula)):
        try:
            _ip.events.unregister(ev, fn)
        except ValueError:
            pass
        _ip.events.register(ev, fn)
except NameError:
    pass  # fora do Jupyter: só o `with etapa(...)` funciona


# ---- (3) Saldo real de UC, lido no painel do Colab --------------------------
def registrar_saldo_uc(saldo, observacao=""):
    """Leia o saldo em 'Recursos' no Colab e registre: registrar_saldo_uc(487.3, "antes do ETL")."""
    _gravar(ARQ_SALDO, {"sessao": SESSAO_ID, "momento_utc": _agora(),
                        "saldo_uc": saldo, "observacao": observacao})
    print(f"[cronômetro] saldo registrado: {saldo} UC ({observacao})")


# ---- Resumo -------------------------------------------------------------------
def resumo_cronometro():
    import pandas as pd
    if os.path.exists(ARQ_ETAPAS):
        d = pd.read_csv(ARQ_ETAPAS)
    elif os.path.exists(ARQ_CELULAS):   # sem etapas nomeadas: usa o registro automático de cada célula
        d = pd.read_csv(ARQ_CELULAS)
        d = d.assign(etapa=d["primeira_linha"], pico_vram_gb=0.0)
        if "categoria" not in d:
            d["categoria"] = d["primeira_linha"].fillna("").map(_categoria)
        print("(resumo pelo registro automático das células)")
    else:
        print("Nenhuma etapa registrada ainda.")
        return None
    tab = (d.groupby("categoria")
             .agg(etapas=("etapa", "count"), horas=("duracao_s", lambda s: s.sum() / 3600),
                  uc_estimadas=("uc_estimadas", "sum"), pico_vram_gb=("pico_vram_gb", "max"))
             .round(3))
    tab.loc["TOTAL"] = [tab.etapas.sum(), tab.horas.sum(), tab.uc_estimadas.sum(), tab.pico_vram_gb.max()]
    print(tab.to_string())
    if os.path.exists(ARQ_SALDO):
        s = pd.read_csv(ARQ_SALDO)
        if len(s) >= 2:
            print(f"\nUC realmente debitadas (saldo inicial − atual): "
                  f"{s.saldo_uc.iloc[0] - s.saldo_uc.iloc[-1]:.2f} UC")
    return tab


def tabela_extrapolacao(csv_gpu, csv_cpu, taxa_uc_cpu=None):
    """Compara a execução completa na GPU com a amostra na CPU, etapa a etapa (pelo nome).
    Tempo extrapolado na CPU = (s por unidade na CPU) × (unidades totais da execução na GPU).
    Etapas sem `quantidade` não são extrapoladas (ficam com o tempo medido na GPU)."""
    import pandas as pd
    g = pd.read_csv(csv_gpu)
    c = pd.read_csv(csv_cpu)
    g = g[g.status == "ok"].groupby("etapa", sort=False).last()
    c = c[c.status == "ok"].groupby("etapa", sort=False).last()
    taxa_cpu = taxa_uc_cpu if taxa_uc_cpu is not None else float(c["taxa_uc_h"].iloc[0])
    linhas = []
    for etapa_nome, rg in g.iterrows():
        q = pd.to_numeric(rg.get("quantidade"), errors="coerce")
        s_gpu = rg["duracao_s"]
        if etapa_nome in c.index and pd.notna(q) and q > 0:
            rc = c.loc[etapa_nome]
            spu_cpu = pd.to_numeric(rc.get("s_por_unidade"), errors="coerce")
            s_cpu = spu_cpu * q if pd.notna(spu_cpu) else float("nan")
            origem = "extrapolado da amostra na CPU"
        else:
            s_cpu, origem = s_gpu, "sem amostra: tempo da GPU repetido"
        linhas.append({
            "categoria": rg["categoria"], "etapa": etapa_nome,
            "unidades": q, "h_gpu": s_gpu / 3600, "h_cpu_est": s_cpu / 3600,
            "razao_cpu_gpu": s_cpu / s_gpu if s_gpu else float("nan"),
            "uc_gpu": rg["uc_estimadas"], "uc_cpu_est": s_cpu / 3600 * taxa_cpu,
            "origem": origem,
        })
    t = pd.DataFrame(linhas)
    tot = t[["h_gpu", "h_cpu_est", "uc_gpu", "uc_cpu_est"]].sum()
    print(t.round(3).to_string(index=False))
    print(f"\nTOTAL  GPU: {tot.h_gpu:.1f} h, {tot.uc_gpu:.1f} UC  |  "
          f"CPU (estimado): {tot.h_cpu_est:.1f} h (≈ {tot.h_cpu_est/24:.1f} dias), {tot.uc_cpu_est:.1f} UC")
    return t


print(f"[cronômetro] ativo | sessão {SESSAO_ID} | {GPU_NOME} | taxa {TAXA_UC_POR_HORA} UC/h | modo {MODO}")
print(f"[cronômetro] registros em {PASTA_CRONOMETRO}")

# =============================================================================
# CATEGORIAS SUGERIDAS (use estes nomes para o resumo sair na ordem do pipeline)
#   1_extracao      ETL IMERG Final Run 2019-2025 e Late Run 2026 (download + 2 km)
#   2_dataset       montagem das janelas de 11 quadros e partições
#   3_treino        ajuste fino (e o ajuste alternativo, se rodar)
#   4_validacao     perdas Fine-Tuned × Zero-Shot e cada regra de decodificação
#   5_teste_FR      teste 2025 (Final Run): F-PluvIA-G, C8, Zero-Shot
#   6_teste_LR      teste cego 2026 (Late Run): F-PluvIA-G, C8, Zero-Shot
#   7_baselines     persistência, Lucas-Kanade, LINDA (CPU)
#   8_metricas      bootstrap, F1 municipal, figuras e tabelas
#
# EXEMPLOS
#   registrar_saldo_uc(500.0, "início, Drive vazio")
#   with etapa("ETL Final Run 2019-2025", "1_extracao", quantidade=28, unidade="meses"): ...
#   for regra in regras:
#       with etapa(f"Validação 2024 – {regra}", "4_validacao", quantidade=240, unidade="janelas"): ...
#   with etapa("Teste cego 2026 – F-PluvIA-C8", "6_teste_LR", quantidade=2110, unidade="janelas"): ...
#   resumo_cronometro()
#
# MODO AMOSTRA (ambiente só com CPU)
#   import os; os.environ["FPLUVIA_MODO_AMOSTRA"] = "1"; os.environ["FPLUVIA_TAXA_UC_H"] = "<taxa da CPU>"
#   os.environ["FPLUVIA_CRONOMETRO_DIR"] = "/content/drive/MyDrive/FPluvIA_cronometro_CPU"
#   (defina ANTES de executar esta célula; use os MESMOS nomes de etapa da execução na GPU)
#   with etapa("Teste 2025 – F-PluvIA-G", "5_teste_FR", quantidade=len(limitar(janelas, "5_teste_FR")), unidade="janelas"):
#       for j in limitar(janelas, "5_teste_FR"): ...
#   Na CPU, rode em fp32 (sem autocast bf16).
#
# EXTRAPOLAÇÃO (depois das duas execuções)
#   tabela_extrapolacao("/content/drive/MyDrive/FPluvIA_cronometro/cronometro_etapas.csv",
#                       "/content/drive/MyDrive/FPluvIA_cronometro_CPU/cronometro_etapas.csv")
# =============================================================================
