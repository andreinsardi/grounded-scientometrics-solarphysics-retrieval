from __future__ import annotations

import json
import shutil
import textwrap
from datetime import datetime
from pathlib import Path


NOTEBOOK_PATH = Path(
    r"C:\Users\andre\odrive\Google Drive\Unicamp\artigo bibliometria\grounded-scientometrics-solarphysics-retrieval\notebooks\01_abstract_llm_nucleo_core_holdout.ipynb"
)


def to_source(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(text),
    }


CELL_0 = """
# 01 Abstract LLM - Nucleo - core + holdout

Notebook canonico da Camada 1 para o corpus `Nucleo`.

## Papel metodologico

Este notebook:

1. le os outputs canonicos do `00_consolidacao`;
2. prepara texto semantico para `core` e `holdout`;
3. gera embeddings com `Sentence-Transformers`;
4. ajusta um pipeline `BERTopic + UMAP + KMeans` separadamente por periodo;
5. exporta artefatos rastreaveis para a pasta canonica `01_abstract_llm`.

## Regra importante

- Este notebook **nao** treina o retriever especializado.
- O treino do `SciBERT_SolarPhysics_Search` continua reservado ao notebook canonicamente planejado como `05_scibert_solarphysics_search_rebuild.ipynb`.
- Aqui fazemos apenas a camada de estrutura semantica upstream, em espelhamento `core/holdout`.
"""


CELL_1 = r'''
# ============================================================
# Instalacao de dependencias para Colab GPU
# ============================================================
!pip install -U -q pip setuptools wheel
!pip uninstall -y -q numpy scipy scikit-learn umap-learn hdbscan numba bertopic || true
!pip install -U -q numpy==2.0.2 scipy==1.14.1 pandas==2.2.2 scikit-learn==1.6.1 numba==0.60.0 umap-learn==0.5.12 hdbscan==0.8.42 openpyxl jedi==0.19.2
!pip install -U -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip install -U -q sentence-transformers==2.7.0 transformers==4.45.2 bertopic==0.16.3
import jedi
print("jedi:", jedi.__version__)

print("Dependencias instaladas.")
'''


CELL_2 = r'''
from google.colab import drive
drive.mount("/content/drive")

import json
import math
import os
import re
import time
import unicodedata
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline
import sklearn
import bertopic

warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")

pd.set_option("display.max_columns", 160)
pd.set_option("display.max_colwidth", 220)
sns.set_theme(style="whitegrid")

print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("GPU disponivel:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("scikit-learn:", sklearn.__version__)
print("umap-learn  :", umap.__version__)
print("bertopic    :", bertopic.__version__)

sk_major_minor = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
assert sk_major_minor >= (1, 6), (
    "Versao de scikit-learn abaixo do esperado. "
    "Reinicie o runtime e rerode a celula de instalacao."
)
'''


CELL_3 = r'''
# ============================================================
# Configuracao geral
# ============================================================

DRIVE_ROOT = Path("/content/drive/MyDrive/Unicamp")
PROJECT_ROOT = DRIVE_ROOT / "artigo bibliometria" / "grounded-scientometrics-solarphysics-retrieval"
DATA_ROOT = DRIVE_ROOT / "artigo bibliometria" / "base de dados" / "Artigo_Bibliometria Base Bruta" / "BASES_UNIFICADAS_POR_TEMA"

TARGET_CORPUS = "Nucleo"
READ_STAGE = "00_consolidacao"
WRITE_STAGE = "01_abstract_llm"
PERIODS = ["core", "holdout"]

USE_TITLE_KEYWORD_FALLBACK = False
MIN_TOPIC_TOKENS = 12
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
SUMMARIZER_PRIMARY = "facebook/bart-large-cnn"
SUMMARIZER_FALLBACK = "sshleifer/distilbart-cnn-12-6"
GENERATE_LLM_SUMMARIES = True
TOP_TOPICS_FOR_SUMMARY = 10

assert PROJECT_ROOT.exists(), f"PROJECT_ROOT nao encontrado: {PROJECT_ROOT}"
assert DATA_ROOT.exists(), f"DATA_ROOT nao encontrado: {DATA_ROOT}"

print("PROJECT_ROOT =", PROJECT_ROOT)
print("DATA_ROOT    =", DATA_ROOT)
print("TARGET_CORPUS =", TARGET_CORPUS)
print("READ_STAGE    =", READ_STAGE)
print("WRITE_STAGE   =", WRITE_STAGE)
print("USE_TITLE_KEYWORD_FALLBACK =", USE_TITLE_KEYWORD_FALLBACK)
print("MIN_TOPIC_TOKENS =", MIN_TOPIC_TOKENS)
print("EMBED_MODEL_NAME =", EMBED_MODEL_NAME)
'''


CELL_4 = r'''
# ============================================================
# Saidas, logs e paths
# ============================================================

PIPE_START_TS = time.time()
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_dir() -> Path:
    return DATA_ROOT / TARGET_CORPUS / "04_rebuild_outputs" / READ_STAGE


def write_dir() -> Path:
    return DATA_ROOT / TARGET_CORPUS / "04_rebuild_outputs" / WRITE_STAGE


def period_input_csv(period: str) -> Path:
    return read_dir() / f"{TARGET_CORPUS}_{period}_bibliometrix_clean.csv"


def period_output_dir(period: str) -> Path:
    return ensure_dir(write_dir() / period)


def elapsed_seconds() -> float:
    return time.time() - PIPE_START_TS


def fmt_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


GLOBAL_LOG_DIR = ensure_dir(PROJECT_ROOT / "outputs" / "camada1_logs")
GLOBAL_LOG_FILE = GLOBAL_LOG_DIR / f"01_abstract_llm_{TARGET_CORPUS}_{RUN_TS}.txt"
WRITE_ROOT = ensure_dir(write_dir())
ensure_dir(WRITE_ROOT / "logs")
ensure_dir(WRITE_ROOT / "figures")
ensure_dir(WRITE_ROOT / "tables")


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{now} | +{fmt_seconds(elapsed_seconds())}]"
    line = f"{prefix} {message}"
    print(line, flush=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def stage_banner(title: str) -> None:
    bar = "=" * 96
    log(bar)
    log(title)
    log(bar)


for period in PERIODS:
    out = period_output_dir(period)
    ensure_dir(out / "figures")
    ensure_dir(out / "tables")
    ensure_dir(out / "reports")

print("GLOBAL_LOG_FILE =", GLOBAL_LOG_FILE)
print("WRITE_ROOT      =", WRITE_ROOT)
'''


CELL_5 = r'''
# ============================================================
# Leitura do 00 canonico e preparo de texto
# ============================================================

EXPECTED_COLS = ["TI", "AB", "PY", "TC", "SO", "DI", "AU", "DE", "ID", "C1", "publication_date"]


def clean_text(value):
    if value is None:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return pd.NA
    return text


def normalize_topic_text(value: str) -> str | pd._libs.missing.NAType:
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"http\S+|www\.\S+|\b10\.\d{4,9}/\S+\b", " ", text)
    text = re.sub(r"[^a-z\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return pd.NA
    return text


def tokens_len(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.split().str.len()


def ensure_expected_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EXPECTED_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def prepare_period_frame(period: str):
    csv_path = period_input_csv(period)
    assert csv_path.exists(), f"Input nao encontrado: {csv_path}"

    raw_df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    raw_df = ensure_expected_cols(raw_df)
    raw_df["period"] = period
    raw_df["abstract_clean"] = raw_df["AB"].map(normalize_topic_text)
    raw_df["title_clean"] = raw_df["TI"].map(normalize_topic_text)
    raw_df["keywords_clean"] = (
        raw_df["DE"].fillna("").astype(str) + "; " + raw_df["ID"].fillna("").astype(str)
    ).map(normalize_topic_text)

    raw_df["topic_text"] = raw_df["abstract_clean"]
    raw_df["text_source"] = "abstract"

    if USE_TITLE_KEYWORD_FALLBACK:
        mask_missing = raw_df["topic_text"].isna()
        fallback_text = (
            raw_df["title_clean"].fillna("").astype(str)
            + ". keywords: "
            + raw_df["keywords_clean"].fillna("").astype(str)
        ).str.strip().replace({"": pd.NA, ". keywords:": pd.NA})
        raw_df.loc[mask_missing, "topic_text"] = fallback_text[mask_missing]
        raw_df.loc[mask_missing & raw_df["topic_text"].notna(), "text_source"] = "title_keywords_fallback"

    raw_df["topic_tokens"] = tokens_len(raw_df["topic_text"])
    usable_df = raw_df[raw_df["topic_tokens"] >= MIN_TOPIC_TOKENS].copy().reset_index(drop=True)
    usable_df["PY_num"] = pd.to_numeric(usable_df["PY"], errors="coerce").astype("Int64")
    usable_df["TC_num"] = pd.to_numeric(usable_df["TC"], errors="coerce").fillna(0)

    coverage_row = {
        "corpus": TARGET_CORPUS,
        "period": period,
        "rows_input": int(len(raw_df)),
        "rows_usable": int(len(usable_df)),
        "usable_pct": round((len(usable_df) / len(raw_df) * 100), 2) if len(raw_df) else 0.0,
        "abstract_present_pct": round(raw_df["abstract_clean"].notna().mean() * 100, 2) if len(raw_df) else 0.0,
        "title_present_pct": round(raw_df["title_clean"].notna().mean() * 100, 2) if len(raw_df) else 0.0,
        "keywords_present_pct": round(raw_df["keywords_clean"].notna().mean() * 100, 2) if len(raw_df) else 0.0,
        "mean_tokens_usable": round(float(usable_df["topic_tokens"].mean()), 2) if len(usable_df) else 0.0,
    }

    coverage_df = pd.DataFrame([coverage_row])
    coverage_out = period_output_dir(period) / "tables" / f"{TARGET_CORPUS}_{period}_text_coverage.csv"
    coverage_df.to_csv(coverage_out, index=False)

    log(
        f"[prep] {period} | input={len(raw_df)} | usable={len(usable_df)} | "
        f"usable_pct={coverage_row['usable_pct']}% | abstract_pct={coverage_row['abstract_present_pct']}%"
    )

    return raw_df, usable_df, coverage_row


stage_banner("LEITURA E PREPARO DO NUCLEO")

period_frames = {}
coverage_rows = []
for period in PERIODS:
    raw_df, usable_df, coverage_row = prepare_period_frame(period)
    period_frames[period] = {"raw": raw_df, "usable": usable_df}
    coverage_rows.append(coverage_row)

coverage_df = pd.DataFrame(coverage_rows)
coverage_df.to_csv(WRITE_ROOT / "tables" / f"{TARGET_CORPUS}_period_text_coverage_{RUN_TS}.csv", index=False)
display(coverage_df)
'''


CELL_6 = r'''
# ============================================================
# Helpers de embeddings, BERTopic, resumos e export
# ============================================================

def candidate_cluster_values(n_docs: int) -> list[int]:
    if n_docs < 1200:
        return [12, 16, 20]
    if n_docs < 3000:
        return [20, 30, 40]
    if n_docs < 8000:
        return [30, 40, 50]
    return [40, 60, 80]


def build_vectorizer(n_docs: int):
    min_df = 3 if n_docs < 3000 else 5
    return CountVectorizer(
        stop_words="english",
        min_df=min_df,
        max_df=0.90,
        ngram_range=(1, 2),
    )


def build_topic_model(n_clusters: int, n_docs: int):
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    representation_model = MaximalMarginalRelevance(diversity=0.3)
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=kmeans_model,
        vectorizer_model=build_vectorizer(n_docs),
        representation_model=representation_model,
        calculate_probabilities=False,
        verbose=False,
    )


def representation_is_valid(rep) -> bool:
    return isinstance(rep, list) and any(isinstance(w, str) and w.strip() for w in rep)


def fit_best_topic_model(texts: list[str], embeddings: np.ndarray, period: str):
    candidates = candidate_cluster_values(len(texts))
    tuning_rows = []
    best = None
    best_score = -1.0

    for n_clusters in candidates:
        log(f"[topic] {period} | testando n_clusters={n_clusters}")
        model = build_topic_model(n_clusters, len(texts))
        topics, _ = model.fit_transform(texts, embeddings)
        info = model.get_topic_info().copy()
        valid_ratio = float(info["Representation"].apply(representation_is_valid).mean()) if len(info) else 0.0
        topic_only = info[info["Topic"] >= 0].copy()
        avg_size = float(topic_only["Count"].mean()) if len(topic_only) else 0.0
        ideal_size = max(1.0, len(texts) / max(1, n_clusters))
        size_ratio = min(avg_size / ideal_size, 1.0)
        score = valid_ratio * 0.7 + size_ratio * 0.3

        tuning_rows.append(
            {
                "period": period,
                "n_clusters": n_clusters,
                "n_rows": len(texts),
                "valid_representation_ratio": round(valid_ratio, 4),
                "avg_topic_size": round(avg_size, 2),
                "ideal_topic_size": round(ideal_size, 2),
                "score": round(score, 4),
            }
        )

        if score > best_score:
            best_score = score
            best = (n_clusters, model, topics, info)

    assert best is not None, f"Nenhum modelo ajustado para {period}"
    tuning_df = pd.DataFrame(tuning_rows).sort_values("score", ascending=False).reset_index(drop=True)
    best_clusters, best_model, best_topics, best_info = best
    log(f"[topic] {period} | melhor n_clusters={best_clusters} | score={best_score:.4f}")
    return best_clusters, best_model, best_topics, best_info, tuning_df


def fallback_terms(topic_model, topic_id: int, topn: int = 10) -> list[str]:
    pairs = topic_model.get_topics().get(topic_id, [])
    return [word for word, _ in pairs[:topn]] if pairs else []


def repair_topic_info(topic_model, info: pd.DataFrame) -> pd.DataFrame:
    fixed = info.copy()
    fixed["Representation"] = fixed.apply(
        lambda row: row["Representation"] if representation_is_valid(row["Representation"]) else fallback_terms(topic_model, int(row["Topic"]), topn=10),
        axis=1,
    )
    return fixed


def load_summarizer():
    if not GENERATE_LLM_SUMMARIES:
        return None
    device = 0 if torch.cuda.is_available() else -1
    try:
        log(f"[summary] carregando modelo principal: {SUMMARIZER_PRIMARY}")
        return pipeline("summarization", model=SUMMARIZER_PRIMARY, device=device)
    except Exception as exc:
        log(f"[summary] falha no modelo principal: {exc}")
        log(f"[summary] fallback para: {SUMMARIZER_FALLBACK}")
        return pipeline("summarization", model=SUMMARIZER_FALLBACK, device=device)


def summarize_text(text: str, summarizer, base_max: int = 140) -> str:
    if summarizer is None or not isinstance(text, str) or len(text.split()) < 40:
        return ""
    n_tokens = max(1, len(text.split()))
    max_len = min(base_max, max(32, int(0.6 * n_tokens)))
    min_len = max(24, int(0.45 * max_len))
    try:
        out = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return out[0]["summary_text"]
    except Exception as exc:
        log(f"[summary] falha de resumo: {exc}")
        return " ".join(text.split()[:80])


def build_topic_summaries(df_used: pd.DataFrame, topics: list[int], info: pd.DataFrame, summarizer, period: str):
    df_local = df_used.copy()
    df_local["topic_id"] = topics
    summaries = {}
    top_topic_ids = (
        info[info["Topic"] >= 0]
        .sort_values("Count", ascending=False)["Topic"]
        .head(TOP_TOPICS_FOR_SUMMARY)
        .tolist()
    )

    for topic_id in top_topic_ids:
        docs = df_local[df_local["topic_id"] == topic_id]["topic_text"].dropna().astype(str).tolist()
        combined = " ".join(docs[:60])[:2400]
        summaries[int(topic_id)] = summarize_text(combined, summarizer)
        log(f"[summary] {period} | topic={topic_id} | docs={len(docs)}")

    return summaries


def build_topic_cards(df_used: pd.DataFrame, topics: list[int], info: pd.DataFrame, topic_summaries: dict) -> list[dict]:
    df_local = df_used.copy()
    df_local["topic_id"] = topics
    cards = []

    for topic_id in info[info["Topic"] >= 0]["Topic"].tolist():
        sub = df_local[df_local["topic_id"] == topic_id].copy()
        years = pd.to_numeric(sub["PY"], errors="coerce").dropna().astype(int).tolist()
        citations = pd.to_numeric(sub["TC"], errors="coerce").dropna().tolist()
        top_years = [year for year, _ in Counter(years).most_common(3)]
        keywords = info.loc[info["Topic"] == topic_id, "Representation"].iloc[0]
        keywords = keywords if isinstance(keywords, list) else []
        title = " / ".join(keywords[:3]) if len(keywords) >= 3 else f"Topic {topic_id}"

        cards.append(
            {
                "topic": int(topic_id),
                "title": title,
                "size": int(len(sub)),
                "top_years": top_years,
                "mean_citations": round(float(np.mean(citations)), 2) if citations else 0.0,
                "keywords": keywords[:10],
                "summary": topic_summaries.get(int(topic_id), ""),
            }
        )

    return cards


def build_executive_summary(cards: list[dict], summarizer) -> str:
    if summarizer is None or not cards:
        return ""
    chunks = []
    for card in cards[:TOP_TOPICS_FOR_SUMMARY]:
        title = card.get("title", "")
        summary = card.get("summary", "")
        if title or summary:
            chunks.append((title + ". " + summary).strip())
    combined = " ".join(chunks)[:3800]
    return summarize_text(combined, summarizer, base_max=180)


def save_topic_trend_plot(counts_df: pd.DataFrame, period: str, out_path: Path):
    if counts_df.empty:
        return

    top_topics = (
        counts_df.groupby("topic")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index
        .tolist()
    )
    plot_df = (
        counts_df[counts_df["topic"].isin(top_topics)]
        .pivot(index="year", columns="topic", values="count")
        .fillna(0)
    )

    plt.figure(figsize=(12, 6))
    for topic_id in plot_df.columns:
        plt.plot(plot_df.index, plot_df[topic_id], marker="o", label=f"Topic {topic_id}")
    plt.title(f"Evolucao anual por topico - {TARGET_CORPUS} - {period}")
    plt.xlabel("Ano")
    plt.ylabel("Documentos por ano")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close()


def run_period_pipeline(period: str, df_used: pd.DataFrame, embedder, summarizer):
    stage_banner(f"ABSTRACT LLM - {TARGET_CORPUS} - {period}")
    out_dir = period_output_dir(period)
    texts = df_used["topic_text"].fillna("").astype(str).tolist()

    log(f"[embed] {period} | docs={len(texts)} | model={EMBED_MODEL_NAME}")
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    log(f"[embed] {period} | shape={embeddings.shape}")

    best_k, topic_model, topics, info, tuning_df = fit_best_topic_model(texts, embeddings, period)
    info = repair_topic_info(topic_model, info)

    df_assign = df_used.copy().reset_index(drop=True)
    df_assign["topic_id"] = topics

    counts_df = (
        df_assign[df_assign["topic_id"] >= 0]
        .assign(year=pd.to_numeric(df_assign["PY"], errors="coerce"))
        .dropna(subset=["year"])
        .groupby(["topic_id", "year"])
        .size()
        .reset_index(name="count")
        .rename(columns={"topic_id": "topic"})
    )

    trend_plot_path = out_dir / "figures" / f"{TARGET_CORPUS}_{period}_topic_trends_top10.png"
    save_topic_trend_plot(counts_df, period, trend_plot_path)

    topic_summaries = build_topic_summaries(df_assign, topics, info, summarizer, period)
    cards = build_topic_cards(df_assign, topics, info, topic_summaries)
    executive_summary = build_executive_summary(cards, summarizer)

    info.to_csv(out_dir / "tables" / f"{TARGET_CORPUS}_{period}_topics_info.csv", index=False)
    tuning_df.to_csv(out_dir / "tables" / f"{TARGET_CORPUS}_{period}_topic_model_tuning.csv", index=False)
    counts_df.to_csv(out_dir / "tables" / f"{TARGET_CORPUS}_{period}_topic_year_counts.csv", index=False)
    df_assign.to_csv(out_dir / "tables" / f"{TARGET_CORPUS}_{period}_doc_topic_assignment.csv", index=False)
    pd.DataFrame(cards).to_csv(out_dir / "tables" / f"{TARGET_CORPUS}_{period}_topic_cards.csv", index=False)

    with open(out_dir / "reports" / f"{TARGET_CORPUS}_{period}_topic_cards.json", "w", encoding="utf-8") as fh:
        json.dump(cards, fh, indent=2, ensure_ascii=False)
    with open(out_dir / "reports" / f"{TARGET_CORPUS}_{period}_executive_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(executive_summary or "")

    metadata = {
        "corpus": TARGET_CORPUS,
        "period": period,
        "rows_used": int(len(df_assign)),
        "chosen_n_clusters": int(best_k),
        "n_topics_non_negative": int((info["Topic"] >= 0).sum()),
        "embedding_model": EMBED_MODEL_NAME,
        "summarizer_enabled": bool(summarizer is not None),
        "run_ts": RUN_TS,
    }
    with open(out_dir / f"{TARGET_CORPUS}_{period}_run_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    log(
        f"[export] {period} | chosen_k={best_k} | topics={(info['Topic'] >= 0).sum()} | "
        f"out_dir={out_dir}"
    )

    return {
        "period": period,
        "rows_used": int(len(df_assign)),
        "chosen_n_clusters": int(best_k),
        "n_topics_non_negative": int((info["Topic"] >= 0).sum()),
        "trend_plot": str(trend_plot_path),
        "executive_summary_chars": len(executive_summary or ""),
    }
'''


CELL_7 = r'''
# ============================================================
# Execucao principal: core e holdout
# ============================================================

stage_banner("EXECUCAO PRINCIPAL DO ABSTRACT LLM")

embed_device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=embed_device)
log(f"Embedder carregado em {embed_device}: {EMBED_MODEL_NAME}")

summarizer = load_summarizer()
if summarizer is None:
    log("Resumos LLM desativados ou indisponiveis.")
else:
    log("Summarizer carregado com sucesso.")

run_rows = []
for period in PERIODS:
    df_used = period_frames[period]["usable"]
    if df_used.empty:
        raise RuntimeError(f"Sem documentos usaveis para {TARGET_CORPUS}/{period}.")
    run_rows.append(run_period_pipeline(period, df_used, embedder, summarizer))

run_df = pd.DataFrame(run_rows)
run_df.to_csv(WRITE_ROOT / "tables" / f"{TARGET_CORPUS}_period_run_summary_{RUN_TS}.csv", index=False)
display(run_df)
'''


CELL_8 = r'''
# ============================================================
# Comparacao core vs holdout e manifesto final
# ============================================================

stage_banner("COMPARACAO CORE VS HOLDOUT")

coverage_df = pd.read_csv(WRITE_ROOT / "tables" / f"{TARGET_CORPUS}_period_text_coverage_{RUN_TS}.csv")
run_df = pd.read_csv(WRITE_ROOT / "tables" / f"{TARGET_CORPUS}_period_run_summary_{RUN_TS}.csv")
summary_df = coverage_df.merge(run_df, on="period", how="left")
summary_df.to_csv(WRITE_ROOT / "tables" / f"{TARGET_CORPUS}_core_holdout_overview_{RUN_TS}.csv", index=False)

manifest_rows = []
for period in PERIODS:
    for path in sorted(period_output_dir(period).rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "corpus": TARGET_CORPUS,
                    "period": period,
                    "artifact": str(path),
                    "size_bytes": path.stat().st_size,
                }
            )

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(WRITE_ROOT / "tables" / f"{TARGET_CORPUS}_artifact_manifest_{RUN_TS}.csv", index=False)

display(summary_df)
display(manifest_df.head(30))

print("Arquivos finais salvos em:", WRITE_ROOT)
'''


def main() -> None:
    notebook = {
        "cells": [
            md_cell(CELL_0),
            code_cell(CELL_1),
            code_cell(CELL_2),
            code_cell(CELL_3),
            code_cell(CELL_4),
            code_cell(CELL_5),
            code_cell(CELL_6),
            code_cell(CELL_7),
            code_cell(CELL_8),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    if NOTEBOOK_PATH.exists():
        backup_path = NOTEBOOK_PATH.with_name(
            f"{NOTEBOOK_PATH.stem}.pre_rewrite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
        )
        shutil.copy2(NOTEBOOK_PATH, backup_path)
        print("Backup salvo em:", backup_path)

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Notebook criado com sucesso:", NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
