from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(
    r"C:\Users\andre\odrive\Google Drive\Unicamp\artigo bibliometria\grounded-scientometrics-solarphysics-retrieval"
)
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"


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


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Notebook criado com sucesso:", path)


def render_template(text: str, mapping: dict[str, str]) -> str:
    rendered = text
    for key, value in mapping.items():
        rendered = rendered.replace(f"__{key}__", value)
    return rendered


ABSTRACT_MD = """
# __NOTEBOOK_ID__ Abstract LLM - __CORPUS__ - core + holdout

Notebook canonico da Camada 0 semantica para o corpus `__CORPUS__`.

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


ABSTRACT_INSTALL = r'''
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


ABSTRACT_IMPORTS = r'''
from google.colab import drive
drive.mount("/content/drive")

import json
import math
import os
import re
import shutil
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


ABSTRACT_CONFIG = r'''
# ============================================================
# Configuracao geral
# ============================================================

DRIVE_ROOT = Path("/content/drive/MyDrive/Unicamp")
PROJECT_ROOT = DRIVE_ROOT / "artigo bibliometria" / "grounded-scientometrics-solarphysics-retrieval"
DATA_ROOT = DRIVE_ROOT / "artigo bibliometria" / "base de dados" / "Artigo_Bibliometria Base Bruta" / "BASES_UNIFICADAS_POR_TEMA"

TARGET_CORPUS = "__CORPUS__"
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


ABSTRACT_PATHS = r'''
# ============================================================
# Saidas, logs e paths
# ============================================================

PIPE_START_TS = time.time()
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
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


GLOBAL_LOG_DIR = ensure_dir(PROJECT_ROOT / "outputs" / "camada1_logs" / TARGET_CORPUS)
GLOBAL_LOG_FILE = GLOBAL_LOG_DIR / f"__NOTEBOOK_ID___abstract_llm_{TARGET_CORPUS}_{RUN_TS}.txt"
WRITE_ROOT = ensure_dir(write_dir())
ensure_dir(WRITE_ROOT / "logs")
ensure_dir(WRITE_ROOT / "figures")
ensure_dir(WRITE_ROOT / "tables")


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{now} | +{fmt_seconds(elapsed_seconds())}]"
    line = f"{prefix} {message}"
    print(line, flush=True)
    GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
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


ABSTRACT_PREP = r'''
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


def normalize_topic_text(value: str):
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


stage_banner("LEITURA E PREPARO DO __CORPUS__")

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


ABSTRACT_HELPERS = r'''
# ============================================================
# Helpers de embeddings, BERTopic, resumos e export
# ============================================================

def candidate_cluster_values(n_docs: int) -> list[int]:
    # Preserve comparability with the historical paper: the reviewed
    # Abstract_LLM notebooks for the canonical corpora evaluated 30/40/50
    # and, under score ties, effectively preferred the smallest k.
    if TARGET_CORPUS in {"Nucleo", "PIML", "CombFinal", "ML_Multimodal", "ML"}:
        return [30, 40, 50]
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

        if best is None or score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and n_clusters < best[0]
        ):
            best_score = score
            best = (n_clusters, model, topics, info)

    assert best is not None, f"Nenhum modelo ajustado para {period}"
    tuning_df = (
        pd.DataFrame(tuning_rows)
        .sort_values(["score", "n_clusters"], ascending=[False, True])
        .reset_index(drop=True)
    )
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


ABSTRACT_RUN = r'''
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


ABSTRACT_COMPARE = r'''
# ============================================================
# Comparacao core vs holdout e manifesto final
# ============================================================

stage_banner("COMPARACAO CORE VS HOLDOUT")

coverage_path = WRITE_ROOT / "tables" / f"{TARGET_CORPUS}_period_text_coverage_{RUN_TS}.csv"
run_path = WRITE_ROOT / "tables" / f"{TARGET_CORPUS}_period_run_summary_{RUN_TS}.csv"

if coverage_path.exists():
    coverage_df = pd.read_csv(coverage_path)
else:
    log("[compare] coverage agregada nao encontrada para RUN_TS atual; reconstruindo a partir dos arquivos por periodo.")
    coverage_rows = []
    for period in PERIODS:
        period_cov_path = period_output_dir(period) / "tables" / f"{TARGET_CORPUS}_{period}_text_coverage.csv"
        if not period_cov_path.exists():
            raise FileNotFoundError(f"Arquivo de cobertura ausente: {period_cov_path}")
        coverage_rows.append(pd.read_csv(period_cov_path))
    coverage_df = pd.concat(coverage_rows, ignore_index=True)
    coverage_df.to_csv(coverage_path, index=False)

if run_path.exists():
    run_df = pd.read_csv(run_path)
else:
    log("[compare] run_summary agregado nao encontrado para RUN_TS atual; reconstruindo a partir dos artefatos por periodo.")
    run_rows = []
    for period in PERIODS:
        out_dir = period_output_dir(period)
        meta_path = out_dir / f"{TARGET_CORPUS}_{period}_run_metadata.json"
        doc_assign_path = out_dir / "tables" / f"{TARGET_CORPUS}_{period}_doc_topic_assignment.csv"
        topics_info_path = out_dir / "tables" / f"{TARGET_CORPUS}_{period}_topics_info.csv"
        tuning_path = out_dir / "tables" / f"{TARGET_CORPUS}_{period}_topic_model_tuning.csv"
        executive_summary_path = out_dir / "reports" / f"{TARGET_CORPUS}_{period}_executive_summary.txt"
        executive_summary_chars = 0
        if executive_summary_path.exists():
            executive_summary_chars = len(executive_summary_path.read_text(encoding="utf-8"))

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            rows_used = int(metadata.get("rows_used", 0))
            chosen_n_clusters = int(metadata.get("chosen_n_clusters", 0))
            n_topics_non_negative = int(metadata.get("n_topics_non_negative", 0))
        else:
            if not doc_assign_path.exists():
                raise FileNotFoundError(
                    f"Artefato ausente para reconstruir rows_used: {doc_assign_path}"
                )
            if not topics_info_path.exists():
                raise FileNotFoundError(
                    f"Artefato ausente para reconstruir n_topics_non_negative: {topics_info_path}"
                )
            if not tuning_path.exists():
                raise FileNotFoundError(
                    f"Artefato ausente para reconstruir chosen_n_clusters: {tuning_path}"
                )

            rows_used = int(len(pd.read_csv(doc_assign_path)))
            topics_info_df = pd.read_csv(topics_info_path)
            n_topics_non_negative = int((topics_info_df["Topic"] >= 0).sum())
            tuning_df = pd.read_csv(tuning_path)
            chosen_n_clusters = int(
                tuning_df.sort_values(["score", "n_clusters"], ascending=[False, True]).iloc[0]["n_clusters"]
            )

        run_rows.append(
            {
                "period": period,
                "rows_used": rows_used,
                "chosen_n_clusters": chosen_n_clusters,
                "n_topics_non_negative": n_topics_non_negative,
                "trend_plot": str(out_dir / "figures" / f"{TARGET_CORPUS}_{period}_topic_trends_top10.png"),
                "executive_summary_chars": executive_summary_chars,
            }
        )

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(run_path, index=False)

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


SCIBERT_MD = """
# 05 SciBERT SolarPhysics Search - rebuild

Notebook canonico da Camada 1 para treinar novamente o retriever especializado
`SciBERT_SolarPhysics_Search` usando **apenas**:

- `Nucleo_core`
- `PIML_core`
- `CombFinal_core`

## Papel metodologico

Este notebook:

1. consome os corpora canonicos do `00_consolidacao`;
2. consome os artefatos semanticos do `01_abstract_llm` para auditoria de insumos;
3. monta o corpus textual de dominio e os pares fracos de contraste;
4. executa `DAPT` e depois `contrastive fine-tuning`;
5. calcula as metricas historicas do paper (`perplexity`, `NMI`, `ARI`, `Silhouette`, `MRR`, `Recall@K`, `NearestCentroidAcc`);
6. publica opcionalmente o modelo atualizado no Hugging Face Hub;
7. salva checkpoints, manifestos, relatorios e artefatos no Google Drive.

## Regra metodologica

- `ML_Multimodal` **nao** entra no treino.
- O treino permanece restrito ao `core`.
- As metricas paper-facing ficam neste notebook; o incremento do major review continua em `06`, `07` e `08`.
- O notebook precisa ser acompanhado por prints frequentes e por logs salvos no Drive.
"""


SCIBERT_INSTALL = r'''
# ============================================================
# Instalacao de dependencias para Colab GPU
# ============================================================
!pip install -U -q pip setuptools wheel
!pip install -U -q numpy==2.0.2 scipy==1.14.1 pandas==2.2.2 scikit-learn==1.5.2 numba==0.60.0
!pip install -U -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip install -U -q transformers==4.45.2 datasets==3.0.1 accelerate==0.34.2 sentence-transformers==2.7.0 huggingface_hub==0.25.2
!pip install -U -q pyarrow openpyxl pyreadr faiss-cpu umap-learn==0.5.6 matplotlib==3.9.2 tabulate==0.9.0

print("Dependencias instaladas.")
'''


SCIBERT_IMPORTS = r'''
from google.colab import drive
drive.mount("/content/drive")

import json
import math
import os
import random
import re
import shutil
import time
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 180)
pd.set_option("display.max_colwidth", 240)

print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("GPU disponivel:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
'''


SCIBERT_CONFIG = r'''
# ============================================================
# Configuracao geral
# ============================================================

DRIVE_ROOT = Path("/content/drive/MyDrive/Unicamp")
PROJECT_ROOT = DRIVE_ROOT / "artigo bibliometria" / "grounded-scientometrics-solarphysics-retrieval"
DATA_ROOT = DRIVE_ROOT / "artigo bibliometria" / "base de dados" / "Artigo_Bibliometria Base Bruta" / "BASES_UNIFICADAS_POR_TEMA"

READ_STAGE_CONSOL = "00_consolidacao"
READ_STAGE_ABSTRACT = "01_abstract_llm"
TRAIN_CORPORA = ["Nucleo", "PIML", "CombFinal"]
WRITE_ROOT = DATA_ROOT / "_cross_corpus_rebuild" / "05_scibert_solarphysics_search"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

BASE_MODEL_ID = "allenai/scibert_scivocab_uncased"
MAX_SEQ_LEN = 384
MIN_TOKENS = 16
MAX_DAPT_DOCS = None
MAX_CONTRASTIVE_PAIRS = 120000
DAPT_EVAL_DOCS = 4000
RUN_DAPT = True
RUN_CONTRASTIVE = True
FORCE_RETRAIN = False
DAPT_EPOCHS = 2
CONTRASTIVE_EPOCHS = 2
DAPT_BATCH_SIZE = 16
CONTRASTIVE_BATCH_SIZE = 64
TOPICS_PER_CORPUS_FOR_AUDIT = 12
RUN_HISTORICAL_AB_EVAL = True
MIN_TECHNIQUE_LABEL_FREQ = 1
MAX_AB_EVAL_DOCS = 8000
AB_RECALL_KS = (10, 50, 100)
AB_UMAP_SAMPLE = 5000
HISTORICAL_AB_EVAL_CORPORA = ["Nucleo", "PIML", "CombFinal", "ML_Multimodal"]
PAIR_SAMPLE_CAPS = {
    "Nucleo": 10000,
    "PIML": 12000,
    "CombFinal": 4000,
}
RUN_HF_PUBLISH = False
HF_REPO_ID = "andreinsardi/SciBERT-SolarPhysics-Search"
HF_PRIVATE = False
HF_COMMIT_MESSAGE = "Canonical major-review rebuild update"
HF_TOKEN_ENV = "HF_TOKEN"

assert PROJECT_ROOT.exists(), f"PROJECT_ROOT nao encontrado: {PROJECT_ROOT}"
assert DATA_ROOT.exists(), f"DATA_ROOT nao encontrado: {DATA_ROOT}"

print("WRITE_ROOT =", WRITE_ROOT)
print("TRAIN_CORPORA =", TRAIN_CORPORA)
print("BASE_MODEL_ID =", BASE_MODEL_ID)
print("RUN_DAPT =", RUN_DAPT)
print("RUN_CONTRASTIVE =", RUN_CONTRASTIVE)
print("RUN_HISTORICAL_AB_EVAL =", RUN_HISTORICAL_AB_EVAL)
print("RUN_HF_PUBLISH =", RUN_HF_PUBLISH)
print("HISTORICAL_AB_EVAL_CORPORA =", HISTORICAL_AB_EVAL_CORPORA)
'''


SCIBERT_LOGGING = r'''
# ============================================================
# Saidas, logs e helpers
# ============================================================

PIPE_START_TS = time.time()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


ARTIFACTS_DIR = ensure_dir(WRITE_ROOT / "artifacts")
REPORTS_DIR = ensure_dir(WRITE_ROOT / "reports")
CHECKPOINT_DIR = ensure_dir(WRITE_ROOT / "checkpoints")
DAPT_DIR = ensure_dir(CHECKPOINT_DIR / "scibert_dapt")
FINAL_MODEL_DIR = ensure_dir(CHECKPOINT_DIR / "SciBERT-SolarPhysics-Search")
GLOBAL_LOG_DIR = ensure_dir(PROJECT_ROOT / "outputs" / "camada2_logs")
GLOBAL_LOG_FILE = GLOBAL_LOG_DIR / f"05_scibert_rebuild_{RUN_TS}.txt"


def elapsed_seconds() -> float:
    return time.time() - PIPE_START_TS


def fmt_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{now} | +{fmt_seconds(elapsed_seconds())}]"
    line = f"{prefix} {message}"
    print(line, flush=True)
    GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def stage_banner(title: str) -> None:
    bar = "=" * 96
    log(bar)
    log(title)
    log(bar)


def read_consol_csv(corpus: str) -> Path:
    return DATA_ROOT / corpus / "04_rebuild_outputs" / READ_STAGE_CONSOL / f"{corpus}_core_bibliometrix_clean.csv"


def read_topic_cards_csv(corpus: str) -> Path:
    return DATA_ROOT / corpus / "04_rebuild_outputs" / READ_STAGE_ABSTRACT / "core" / "tables" / f"{corpus}_core_topic_cards.csv"


print("GLOBAL_LOG_FILE =", GLOBAL_LOG_FILE)
print("FINAL_MODEL_DIR =", FINAL_MODEL_DIR)
'''


SCIBERT_LOAD = r'''
# ============================================================
# Leitura dos corpora de treino e auditoria dos insumos
# ============================================================

EXPECTED_COLS = ["TI", "AB", "DE", "ID", "SO", "PY", "TC", "DI"]


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


def light_clean_text(value):
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"https?://\S+|doi:\S+", " ", text, flags=re.I)
    text = text.lower().strip()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return pd.NA
    return text


def ensure_expected_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EXPECTED_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def concat_alias(corpus: str) -> str:
    mapping = {
        "Nucleo": "nucleo",
        "PIML": "piml",
        "CombFinal": "combf",
        "ML_Multimodal": "ml",
    }
    if corpus not in mapping:
        raise KeyError(f"Corpus sem alias concat conhecido: {corpus}")
    return mapping[corpus]


def build_concat_export_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "corpus": df["corpus"],
            "doc_local_id": df["doc_local_id"],
            "TI": df["title_clean"],
            "AB": df["abstract_clean"],
            "DE": df["de_clean"],
            "ID": df["id_clean"],
            "SO": df["SO"],
            "PY": df["PY"],
            "TC": df["TC"],
            "DI": df["DI"],
            "text_full": df["text_full"],
        }
    )
    for col in ["TI", "AB", "DE", "ID", "SO", "PY", "TC", "DI", "text_full"]:
        out[col] = out[col].fillna("").astype(str)
    return out


def prepare_train_frame(corpus: str) -> pd.DataFrame:
    csv_path = read_consol_csv(corpus)
    assert csv_path.exists(), f"Input nao encontrado: {csv_path}"
    raw = pd.read_csv(csv_path, dtype=str, low_memory=False)
    raw = ensure_expected_cols(raw)
    raw["corpus"] = corpus
    for col in ["TI", "AB", "DE", "ID", "SO", "PY", "TC", "DI"]:
        raw[col] = raw[col].map(clean_text)
    raw["title_clean"] = raw["TI"].map(light_clean_text)
    raw["abstract_clean"] = raw["AB"].map(light_clean_text)
    raw["de_clean"] = raw["DE"].map(light_clean_text)
    raw["id_clean"] = raw["ID"].map(light_clean_text)
    raw["keywords_clean"] = (
        raw["de_clean"].fillna("").astype(str) + "; " + raw["id_clean"].fillna("").astype(str)
    ).map(light_clean_text)
    raw["text_full"] = (
        raw["title_clean"].fillna("").astype(str)
        + ". "
        + raw["abstract_clean"].fillna("").astype(str)
        + ". keywords: "
        + raw["keywords_clean"].fillna("").astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    raw["query_side"] = (
        raw["title_clean"].fillna("").astype(str)
        + ". keywords: "
        + raw["keywords_clean"].fillna("").astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    raw["positive_side"] = raw["abstract_clean"].fillna(raw["text_full"])
    raw["text_tokens"] = raw["text_full"].fillna("").astype(str).str.split().str.len()
    prepared = raw.reset_index(drop=True).copy()
    prepared["doc_local_id"] = [f"{corpus}_{i:07d}" for i in range(len(prepared))]
    log(
        f"[load] {corpus} | input={len(raw)} | ready={len(prepared)} | "
        f"abstract_pct={round(raw['abstract_clean'].notna().mean() * 100, 2) if len(raw) else 0.0}%"
    )
    return prepared


stage_banner("LEITURA DOS CORPORA DE TREINO")

domain_frames = []
domain_frame_map = {}
concat_frame_map = {}
audit_rows = []
for corpus in TRAIN_CORPORA:
    df = prepare_train_frame(corpus)
    domain_frames.append(df)
    domain_frame_map[corpus] = df
    concat_frame = build_concat_export_df(df)
    concat_frame_map[corpus] = concat_frame
    concat_frame.to_csv(ARTIFACTS_DIR / f"concat_{concat_alias(corpus)}.csv", index=False)
    audit_rows.append(
        {
            "corpus": corpus,
            "rows_usable": int(len(df)),
            "mean_tokens": round(float(df["text_tokens"].mean()), 2) if len(df) else 0.0,
            "mean_citations": round(float(pd.to_numeric(df["TC"], errors="coerce").fillna(0).mean()), 2) if len(df) else 0.0,
        }
    )
    cards_path = read_topic_cards_csv(corpus)
    if cards_path.exists():
        cards = pd.read_csv(cards_path)
        log(f"[audit] {corpus} | topic_cards_core={len(cards)} | path={cards_path}")
        display(cards.head(TOPICS_PER_CORPUS_FOR_AUDIT))
    else:
        log(f"[audit] {corpus} | topic_cards_core ausente em {cards_path}")

train_df = pd.concat(domain_frames, ignore_index=True)
if MAX_DAPT_DOCS:
    train_df = train_df.head(MAX_DAPT_DOCS).copy()

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(ARTIFACTS_DIR / "domain_core_corpus_audit.csv", index=False)
train_df.to_csv(ARTIFACTS_DIR / "domain_core_training_corpus.csv", index=False)

historical_ab_frames = []
historical_ab_audit_rows = []
historical_ab_concat_frames = []
for corpus in HISTORICAL_AB_EVAL_CORPORA:
    if corpus in domain_frame_map:
        eval_df = domain_frame_map[corpus].copy()
        concat_eval_df = concat_frame_map[corpus].copy()
    else:
        eval_df = prepare_train_frame(corpus)
        concat_eval_df = build_concat_export_df(eval_df)
        concat_eval_df.to_csv(ARTIFACTS_DIR / f"concat_{concat_alias(corpus)}.csv", index=False)
    historical_ab_frames.append(eval_df)
    historical_ab_concat_frames.append(concat_eval_df)
    historical_ab_audit_rows.append(
        {
            "corpus": corpus,
            "rows_usable": int(len(eval_df)),
            "mean_tokens": round(float(eval_df["text_tokens"].mean()), 2) if len(eval_df) else 0.0,
        }
    )

historical_ab_source_df = pd.concat(historical_ab_frames, ignore_index=True)
historical_ab_source_df = historical_ab_source_df.drop_duplicates(subset=["corpus", "doc_local_id"]).reset_index(drop=True)
historical_ab_concat_df = pd.concat(historical_ab_concat_frames, ignore_index=True)
historical_ab_concat_df = historical_ab_concat_df.drop_duplicates(subset=["corpus", "doc_local_id"]).reset_index(drop=True)
pd.DataFrame(historical_ab_audit_rows).to_csv(ARTIFACTS_DIR / "historical_ab_corpus_audit.csv", index=False)
historical_ab_source_df.to_csv(ARTIFACTS_DIR / "historical_ab_source_corpus.csv", index=False)
historical_ab_concat_df.to_csv(ARTIFACTS_DIR / "historical_ab_concat_corpus.csv", index=False)

log(f"[load] train_df total={len(train_df)}")
log(f"[load] historical_ab_source_df total={len(historical_ab_source_df)}")
log(f"[load] historical_ab_concat_df total={len(historical_ab_concat_df)}")
display(audit_df)
display(train_df.head(3))
'''


SCIBERT_PAIRS = r'''
# ============================================================
# Montagem do corpus DAPT e dos pares fracos de contraste
# ============================================================

stage_banner("MONTAGEM DE INSUMOS DE TREINO")

dapt_texts = train_df["text_full"].dropna().astype(str).tolist()


def build_pairs_for_corpus(df: pd.DataFrame, sample_cap: int | None = None) -> pd.DataFrame:
    subset = df.dropna(subset=["abstract_clean", "title_clean"]).copy()
    if sample_cap and len(subset) > sample_cap:
        subset = subset.sample(sample_cap, random_state=42)

    def safe_text(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    rows = []
    for row in subset.itertuples(index=False):
        abstract_text = safe_text(row.abstract_clean)
        title_text = safe_text(row.title_clean)
        de_text = safe_text(row.de_clean)
        id_text = safe_text(row.id_clean)

        if len(title_text) > 10 and len(abstract_text) > 10:
            rows.append(
                {
                    "corpus": row.corpus,
                    "doc_local_id": row.doc_local_id,
                    "query_side": title_text,
                    "positive_side": abstract_text,
                    "query_origin": "title",
                    "PY": row.PY,
                    "DI": row.DI,
                    "SO": row.SO,
                }
            )
        if len(de_text) > 5 and len(abstract_text) > 10:
            rows.append(
                {
                    "corpus": row.corpus,
                    "doc_local_id": row.doc_local_id,
                    "query_side": de_text,
                    "positive_side": abstract_text,
                    "query_origin": "de",
                    "PY": row.PY,
                    "DI": row.DI,
                    "SO": row.SO,
                }
            )
        if len(id_text) > 5 and len(abstract_text) > 10:
            rows.append(
                {
                    "corpus": row.corpus,
                    "doc_local_id": row.doc_local_id,
                    "query_side": id_text,
                    "positive_side": abstract_text,
                    "query_origin": "id",
                    "PY": row.PY,
                    "DI": row.DI,
                    "SO": row.SO,
                }
            )

    return pd.DataFrame(rows)


pair_parts = []
pair_audit_rows = []
for corpus in TRAIN_CORPORA:
    cap = PAIR_SAMPLE_CAPS.get(corpus)
    built = build_pairs_for_corpus(domain_frame_map[corpus], sample_cap=cap)
    pair_parts.append(built)
    pair_audit_rows.append(
        {
            "corpus": corpus,
            "sample_cap": cap,
            "pairs_built_raw": int(len(built)),
        }
    )

pair_df = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
if len(pair_df):
    pair_df = pair_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

if MAX_CONTRASTIVE_PAIRS and len(pair_df) > MAX_CONTRASTIVE_PAIRS:
    pair_df = pair_df.sample(MAX_CONTRASTIVE_PAIRS, random_state=42).reset_index(drop=True)

pair_df["query_tokens"] = pair_df["query_side"].fillna("").astype(str).str.split().str.len()
pair_df["positive_tokens"] = pair_df["positive_side"].fillna("").astype(str).str.split().str.len()

log(f"[pairs] dapt_texts={len(dapt_texts)}")
log(f"[pairs] contrastive_pairs={len(pair_df)}")

pair_df.to_csv(ARTIFACTS_DIR / "contrastive_pairs.csv", index=False)
pd.DataFrame(pair_audit_rows).to_csv(ARTIFACTS_DIR / "contrastive_pair_audit.csv", index=False)
pd.DataFrame(
    [
        {
            "n_dapt_texts": len(dapt_texts),
            "n_pairs": len(pair_df),
            "mean_query_tokens": round(float(pair_df["query_tokens"].mean()), 2) if len(pair_df) else 0.0,
            "mean_positive_tokens": round(float(pair_df["positive_tokens"].mean()), 2) if len(pair_df) else 0.0,
        }
    ]
).to_csv(ARTIFACTS_DIR / "training_input_summary.csv", index=False)

display(pair_df.head(5))
'''


SCIBERT_DAPT = r'''
# ============================================================
# DAPT (Masked Language Modeling) no core de dominio
# ============================================================

stage_banner("DAPT - SCIBERT")

if FINAL_MODEL_DIR.exists() and any(FINAL_MODEL_DIR.iterdir()) and not FORCE_RETRAIN:
    log(f"[dapt] modelo final ja existe em {FINAL_MODEL_DIR}. Reuso sem retreino.")
    dapt_base_path = str(FINAL_MODEL_DIR)
    dapt_report = {
        "status": "reused_existing_final_model",
        "base_path": dapt_base_path,
        "run_ts": RUN_TS,
    }
    with open(REPORTS_DIR / "dapt_eval_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(dapt_report, fh, indent=2, ensure_ascii=False)
else:
    if RUN_DAPT:
        raw_ds = Dataset.from_dict({"text": dapt_texts})
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        mlm_model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_ID)

        def tokenize_batch(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding="max_length",
            )

        eval_size = min(DAPT_EVAL_DOCS, len(dapt_texts))
        tokenized_train = raw_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
        tokenized_eval = tokenized_train.select(range(eval_size))
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

        args = TrainingArguments(
            output_dir=str(DAPT_DIR),
            overwrite_output_dir=True,
            num_train_epochs=DAPT_EPOCHS,
            per_device_train_batch_size=DAPT_BATCH_SIZE,
            per_device_eval_batch_size=DAPT_BATCH_SIZE,
            logging_steps=25,
            save_strategy="epoch",
            save_total_limit=2,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=mlm_model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=collator,
        )

        log(
            f"[dapt] iniciando treino MLM | docs={len(dapt_texts)} | "
            f"train={len(tokenized_train)} | eval={len(tokenized_eval)} | "
            f"eval_mode=paper_compat_subset | epochs={DAPT_EPOCHS}"
        )
        trainer.train()
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss")
        ppl = math.exp(eval_loss) if eval_loss is not None and eval_loss < 20 else None
        trainer.save_model(str(DAPT_DIR))
        tokenizer.save_pretrained(str(DAPT_DIR))
        with open(REPORTS_DIR / "dapt_training_args.json", "w", encoding="utf-8") as fh:
            json.dump(args.to_dict(), fh, indent=2, ensure_ascii=False)
        with open(REPORTS_DIR / "dapt_eval_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "eval_loss": eval_loss,
                    "perplexity": ppl,
                    "train_docs": len(tokenized_train),
                    "eval_docs": len(tokenized_eval),
                    "run_ts": RUN_TS,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )
        pd.DataFrame(
            [
                {
                    "eval_loss": eval_loss,
                    "perplexity": ppl,
                    "train_docs": len(tokenized_train),
                    "eval_docs": len(tokenized_eval),
                }
            ]
        ).to_csv(REPORTS_DIR / "dapt_eval_summary.csv", index=False)
        dapt_base_path = str(DAPT_DIR)
        log(
            f"[dapt] concluido | checkpoint={DAPT_DIR} | "
            f"eval_loss={round(eval_loss, 4) if eval_loss is not None else 'NA'} | "
            f"perplexity={round(ppl, 4) if ppl is not None else 'NA'}"
        )
    else:
        dapt_base_path = BASE_MODEL_ID
        log("[dapt] desativado por configuracao. Seguindo direto para o contrastivo.")
        with open(REPORTS_DIR / "dapt_eval_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "status": "skipped_by_configuration",
                    "base_path": dapt_base_path,
                    "run_ts": RUN_TS,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

print("Base para o contrastivo =", dapt_base_path)
'''


SCIBERT_CONTRASTIVE = r'''
# ============================================================
# Fine-tuning contrastivo (Sentence-Transformers)
# ============================================================

stage_banner("FINE-TUNING CONTRASTIVO")

if FINAL_MODEL_DIR.exists() and any(FINAL_MODEL_DIR.iterdir()) and not FORCE_RETRAIN:
    log(f"[contrastive] modelo final ja existe em {FINAL_MODEL_DIR}. Reuso sem retreino.")
else:
    if not RUN_CONTRASTIVE:
        raise RuntimeError("RUN_CONTRASTIVE=False e nao existe modelo final pronto.")

    base_for_sentence = dapt_base_path if isinstance(dapt_base_path, str) else BASE_MODEL_ID
    word = models.Transformer(base_for_sentence)
    pool = models.Pooling(
        word.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    norm = models.Normalize()
    st_model = SentenceTransformer(modules=[word, pool, norm], device="cuda" if torch.cuda.is_available() else "cpu")
    st_model.max_seq_length = MAX_SEQ_LEN

    pair_examples = [
        InputExample(texts=[row.query_side, row.positive_side])
        for row in pair_df.itertuples(index=False)
    ]
    train_loader = DataLoader(pair_examples, shuffle=True, batch_size=CONTRASTIVE_BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(st_model)
    warmup_steps = max(10, int(len(train_loader) * CONTRASTIVE_EPOCHS * 0.1))

    log(
        f"[contrastive] examples={len(pair_examples)} | batches={len(train_loader)} | "
        f"epochs={CONTRASTIVE_EPOCHS} | warmup={warmup_steps}"
    )

    st_model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=CONTRASTIVE_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=str(FINAL_MODEL_DIR),
        show_progress_bar=True,
    )

    log(f"[contrastive] concluido | final_model={FINAL_MODEL_DIR}")

assert FINAL_MODEL_DIR.exists(), f"Modelo final nao encontrado: {FINAL_MODEL_DIR}"
'''


SCIBERT_HISTORICAL_LABELS = r'''
# ============================================================
# Rotulacao fraca multi-eixo para compatibilidade com o paper
# ============================================================

stage_banner("ROTULACAO FRACA MULTI-EIXO")

BOILERPLATE = [
    r"\belsevier\b.*?\bright(s)?\s+reserved\b",
    r"\ball\s+rights\s+reserved\b",
    r"\u00A9\s*\d{4,}\b",
    r"\b(?:copyright)\b|\u00A9",
]

INLINE_TAGS = [
    r"\binf\s*/\s*inf\b",
    r"\bsup\s*/\s*sup\b",
    r"\bsub\s*/\s*sub\b",
    r"\b(et|al)\.\b",
]


def strong_clean(value):
    text = clean_text(value)
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"https?://\S+|doi:\S+", " ", text, flags=re.I)
    for pattern in BOILERPLATE + INLINE_TAGS:
        text = re.sub(pattern, " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text

fenomeno_patterns = {
    "solar_flare": r"\bsolar\s+flare(s)?\b|\b(goes[-\s]?(x|m|c)\s*class)\b|\bflare\s+forecast(ing)?\b",
    "solar_wind": r"\bsolar\s+wind\b|\binterplanetary\s+magnetic\s+field\b|\bimf\b",
    "magnetic_field_flux": r"\bmagnetic\s+field(s)?\b|\bmagnetic\s+flux\b|\bmagnetogram(s)?\b",
    "magnetic_reconnection": r"\bmagnetic\s+reconnection\b",
    "sunspot_active_region": r"\bsunspot(s)?\b|\bactive\s+region(s)?\b|\bar[\s\-]?\d{3,5}\b",
    "coronal_hole": r"\bcoronal\s+hole(s)?\b",
    "cme": r"\bcoronal\s+mass\s+ejection(s)?\b|\bcme(s)?\b|\bshock\s+(front|arrival)\b",
    "corona_heliosphere": r"\bcorona(l)?\b|\bheliosphere\b|\bmagnetosphere\b|\bheliospheric\s+current\s+sheet\b",
    "mhd_plasma_turbulence": r"\bmhd\b|\bmagnetohydrodynamic(s)?\b|\bplasma\s+turbulence\b|\balfven(\s+wave(s)?)?\b",
    "plasma_diffusion_transport": r"\b(plasma\s+)?diffusion\b|\btransport\s+process(es)?\b|\badvection[-\s]?diffusion\b",
    "coronal_loop": r"\bcoronal\s+loop(s)?\b|\bloops?\s+oscillation(s)?\b",
    "chromosphere_photosphere": r"\bchromosphere\b|\bphotosphere\b|\bquiet\s+sun\b",
    "solar_energetic_particles": r"\bsolar\s+energetic\s+particle(s)?\b|\bsep(s)?\b",
    "magnetic_helicity_instability": r"\bmagnetic\s+helicity\b|\b(plasma|magnetic)\s+instabilit(y|ies)\b",
    "shock_acceleration": r"\bshock\s+(wave|acceleration|front)\b",
    "coronal_heating": r"\bcoronal\s+heating\b",
    "plasma_wave_dynamics": r"\bplasma\s+wave(s)?\b|\bwave\s+dissipation\b",
    "parker_spiral_probe": r"\bparker\s+(spiral|solar\s+probe)\b|\bpsp\b",
}

tarefa_patterns = {
    "forecasting": r"\bforecast(ing)?\b|\bnowcast(ing)?\b|\bprediction\b",
    "classification": r"\bclassification\b|\bclassifier(s)?\b",
    "detection": r"\bdetection\b|\bdetect(ion|or|ing)?\b",
    "segmentation": r"\bsegmentation\b",
    "regression": r"\bregression\b|\binverse\s+problem(s)?\b",
}

metodo_patterns = {
    "piml_pinn": r"physics[-\s]?informed|physics[-\s]?guided|theory[-\s]?guided|\bpinn(s)?\b|pde[-\s]?constrained|physics[-\s]?constrained",
    "governing_equations": r"\b(maxwell|navier[-\s]?stokes|poisson|helmholtz|mhd|advection[-\s]?diffusion)\b\s*(equation|eqs|equations)?",
    "hybrid_physics_ml": r"\bhybrid\s+(physics|model|approach)\b|\bdata[-\s]?driven\s+and\s+physics[-\s]?based\b",
    "statistical_bayesian": r"\bbayesian\b|\bstatistical\b",
}

tecnica_patterns = {
    "transformer": r"\btransformer(s)?\b|transformer[-\s]*(encoder|decoder)\b|\btransformer[-\s]?based\b",
    "attention_variants": r"\b(self|cross|multi[-\s]?head)\s*attention\b|\battention\s+mechanism\b|\btemporal\s+fusion\s+transformer\b|\btft\b",
    "vit_vision_transformer": r"\bvision\s+transformer\b|\bvit\b|\bswin[-\s]?transformer\b",
    "multimodal_fusion": r"\bmultimodal\b|\bmulti[-\s]?modal\b|\b(data|feature)\s+fusion\b|\bearly\s+fusion\b|\blate\s+fusion\b|\bcross[-\s]?modal\b",
    "cnn": r"\bcnn(s)?\b|\bconvolutional\s+neural\s+network(s)?\b|\bresnet\b|\befficientnet\b",
    "lstm_gru_rnn": r"\blstm\b|\bgru\b|\brnn(s)?\b|\brecurrent\s+neural\s+network\b",
    "gnn_graph": r"\bgnn(s)?\b|\bgraph\s+neural\s+network(s)?\b|\bgraph\s+convolution\b",
    "svm_rf_xgb": r"\bsvm\b|\bsupport\s+vector\s+machine\b|\brandom\s+forest\b|\bxgboost\b|\bxgb\b",
    "autoencoder_vae": r"\bautoencoder(s)?\b|\bvariational\s+autoencoder\b|\bvae\b",
    "ann_dnn_ffn": r"\bartificial\s+neural\s+network(s)?\b|\bdeep\s+neural\s+network(s)?\b|\bfeed[-\s]?forward\s+network(s)?\b",
    "ensemble_learning": r"\bensemble\s+learning\b|\bmodel\s+ensemble(s)?\b",
    "multi_task_learning": r"\bmulti[-\s]?task\s+learning\b|\bmulti[-\s]?objective\s+learning\b",
    "spatio_temporal_transformer": r"\bspatio[-\s]?temporal\s+(transformer|network)\b|\btemporal\s+fusion\s+transformer\b",
    "dual_encoder": r"\bdual[-\s]?encoder\b|\bbiencoder\b",
    "gat_graph_attention": r"\bgraph\s+attention\s+network(s)?\b|\bgat\b",
    "physics_constrained_loss": r"\bphysics[-\s]?constrained\s+loss\b|\bphysical\s+constraint(s)?\b",
}

dados_patterns = {
    "magnetogram_hmi": r"\bmagnetogram(s)?\b|\bhmi\b|\bsdo/hmi\b",
    "aia_imagery": r"\baia\b|\bsdo/aia\b",
    "goes": r"\bgoes(-\w+)?\b",
    "rhessi": r"\brhessi\b",
    "soho_lasco": r"\bsoho\b|\blasco\b",
    "stereo": r"\bstereo\b",
    "hinode": r"\bhinode\b",
    "parker_solar_probe": r"\bparker\s+solar\s+probe\b|\bpsp\b",
}


def tag_with_patterns(text: str, patterns_dict: dict[str, str]) -> str:
    text = str(text or "")
    hits = []
    for label, pattern in patterns_dict.items():
        if re.search(pattern, text, flags=re.I) and label not in hits:
            hits.append(label)
    return ";".join(hits)


label_base_df = pd.concat(
    [concat_frame_map[corpus].copy() for corpus in TRAIN_CORPORA],
    ignore_index=True,
)
for col in ["TI", "AB", "DE", "ID"]:
    label_base_df[col] = label_base_df[col].map(strong_clean)
label_base_df["label_text_full"] = (
    label_base_df["TI"].fillna("").astype(str).str.strip()
    + ". "
    + label_base_df["AB"].fillna("").astype(str).str.strip()
    + " keywords: "
    + label_base_df["DE"].fillna("").astype(str)
    + "; "
    + label_base_df["ID"].fillna("").astype(str)
).str.replace(r"\s+", " ", regex=True).str.strip()

label_df = label_base_df[["doc_local_id", "corpus", "PY"]].copy()
label_df["DOI"] = label_base_df["DI"]
label_df["Fenomeno"] = label_base_df["label_text_full"].map(lambda text: tag_with_patterns(text, fenomeno_patterns))
label_df["Tarefa"] = label_base_df["label_text_full"].map(lambda text: tag_with_patterns(text, tarefa_patterns))
label_df["Metodo"] = label_base_df["label_text_full"].map(lambda text: tag_with_patterns(text, metodo_patterns))
label_df["Tecnica"] = label_base_df["label_text_full"].map(lambda text: tag_with_patterns(text, tecnica_patterns))
label_df["Dados"] = label_base_df["label_text_full"].map(lambda text: tag_with_patterns(text, dados_patterns))
label_df["primary_tecnica"] = label_df["Tecnica"].fillna("").astype(str).str.split(";").str[0].str.strip()
label_df.to_csv(ARTIFACTS_DIR / "labels_multi_axis.csv", index=False)

coverage_rows = []
for axis in ["Fenomeno", "Tarefa", "Metodo", "Tecnica", "Dados", "primary_tecnica"]:
    non_empty = label_df[axis].fillna("").astype(str).str.len().gt(0)
    coverage_rows.append(
        {
            "axis": axis,
            "coverage_pct": round(float(non_empty.mean() * 100), 2),
            "non_empty_docs": int(non_empty.sum()),
            "total_docs": int(len(label_df)),
        }
    )

coverage_df = pd.DataFrame(coverage_rows)
coverage_df.to_csv(ARTIFACTS_DIR / "labels_coverage.csv", index=False)

primary_tecnica_counts = (
    label_df["primary_tecnica"]
    .fillna("")
    .loc[lambda series: series.str.len().gt(0)]
    .value_counts()
    .rename_axis("primary_tecnica")
    .reset_index(name="count")
)
primary_tecnica_counts.to_csv(ARTIFACTS_DIR / "primary_tecnica_counts.csv", index=False)


def explode_axis(value: str) -> list[str]:
    parts = [item.strip() for item in str(value or "").split(";")]
    return [item for item in parts if item]


pair_rows = []
for row in label_df[["Fenomeno", "Tecnica"]].itertuples(index=False):
    phenomena = explode_axis(row.Fenomeno)
    techniques = explode_axis(row.Tecnica)
    for phenomenon in phenomena:
        for technique in techniques:
            pair_rows.append((phenomenon, technique))

if pair_rows:
    pair_df = pd.DataFrame(pair_rows, columns=["Fenomeno", "Tecnica"])
    fen_tec_df = (
        pair_df.value_counts()
        .rename("freq")
        .reset_index()
        .sort_values(["freq", "Fenomeno", "Tecnica"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
else:
    fen_tec_df = pd.DataFrame(columns=["Fenomeno", "Tecnica", "freq"])

fen_tec_df.to_csv(ARTIFACTS_DIR / "fenomeno_x_tecnica.csv", index=False)

log(
    f"[labels] docs={len(label_df)} | tecnica_coverage="
    f"{round(float(coverage_df.loc[coverage_df['axis'] == 'Tecnica', 'coverage_pct'].iloc[0]), 2) if len(coverage_df) else 0.0}%"
)
display(coverage_df)
display(primary_tecnica_counts.head(20))
display(fen_tec_df.head(20))
'''


SCIBERT_HISTORICAL_AB = r'''
# ============================================================
# Metricas historicas do paper: baseline vs fine-tuned
# ============================================================

stage_banner("AVALIACAO HISTORICA PAPER-FACING")

if not RUN_HISTORICAL_AB_EVAL:
    log("[ab] avaliacao historica desativada por configuracao.")
    with open(REPORTS_DIR / "historical_ab_eval_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"status": "skipped_by_configuration", "run_ts": RUN_TS}, fh, indent=2, ensure_ascii=False)
else:
    labels_df = pd.read_csv(ARTIFACTS_DIR / "labels_multi_axis.csv")
    labels_lookup = labels_df.drop(columns=["DI"], errors="ignore")
    labels_lookup = labels_lookup.rename(columns={"DOI": "DI"}).drop_duplicates(subset=["DI"], keep="first")
    eval_source_df = historical_ab_concat_df.copy()
    eval_df = eval_source_df.merge(labels_lookup[["DI", "Tecnica"]], on="DI", how="left")
    eval_df["Tecnica_clean"] = eval_df["Tecnica"].fillna("").astype(str).str.split(";").str[0].str.strip()
    txt_col = "AB" if "AB" in eval_df.columns else ("text_full" if "text_full" in eval_df.columns else None)
    if txt_col is None:
        eval_df["text_full"] = eval_df.astype(str).agg(" ".join, axis=1)
        txt_col = "text_full"
    eval_df["eval_text"] = eval_df[txt_col].fillna("").astype(str).str.strip()
    eval_df = eval_df[eval_df["Tecnica_clean"].str.len().gt(0)].copy()

    if MIN_TECHNIQUE_LABEL_FREQ and int(MIN_TECHNIQUE_LABEL_FREQ) > 1:
        label_counts = eval_df["Tecnica_clean"].value_counts()
        keep_labels = label_counts[label_counts >= MIN_TECHNIQUE_LABEL_FREQ].index.tolist()
        eval_df = eval_df[eval_df["Tecnica_clean"].isin(keep_labels)].copy()

    if MAX_AB_EVAL_DOCS:
        sample_n = min(MAX_AB_EVAL_DOCS, len(eval_df))
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(eval_df.index.to_numpy(), size=sample_n, replace=False)
        eval_df = eval_df.loc[sample_idx].reset_index(drop=True)

    if eval_df["Tecnica_clean"].nunique() < 2:
        raise RuntimeError(
            "[ab] menos de duas classes elegiveis de Tecnica_clean apos o filtro; "
            "reduza MIN_TECHNIQUE_LABEL_FREQ ou revise a rotulacao fraca."
        )

    eval_df.to_csv(ARTIFACTS_DIR / "ab_eval_sample.csv", index=False)
    pd.DataFrame(
        [{"Tecnica_clean": label, "count": int(count)} for label, count in eval_df["Tecnica_clean"].value_counts().items()]
    ).to_csv(REPORTS_DIR / "ab_eval_primary_tecnica_counts.csv", index=False)

    texts = eval_df["eval_text"].tolist()
    y_tecnica = eval_df["Tecnica_clean"].tolist()
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    encode_batch_size = 128 if torch.cuda.is_available() else 32

    base_word = models.Transformer(BASE_MODEL_ID)
    base_pool = models.Pooling(
        base_word.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    base_norm = models.Normalize()
    model_baseline = SentenceTransformer(modules=[base_word, base_pool, base_norm], device=device_name)
    model_ft = SentenceTransformer(str(FINAL_MODEL_DIR), device=device_name)

    log(
        f"[ab] docs={len(eval_df)} | classes={eval_df['Tecnica_clean'].nunique()} | "
        f"batch_size={encode_batch_size}"
    )
    emb_base = model_baseline.encode(
        texts,
        normalize_embeddings=True,
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb_ft = model_ft.encode(
        texts,
        normalize_embeddings=True,
        batch_size=encode_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import umap

    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(y_tecnica)
    k_eval = len(label_encoder.classes_)


    def eval_clustering_metrics(embs: np.ndarray, y_true_labels: np.ndarray, model_name: str) -> dict:
        km = KMeans(n_clusters=k_eval, n_init=10, random_state=42)
        y_pred = km.fit_predict(embs)
        sil = float("nan")
        sample_size = min(5000, len(embs))
        if sample_size >= 10 and len(np.unique(y_pred[:sample_size])) > 1:
            sil = float(silhouette_score(embs[:sample_size], y_pred[:sample_size], metric="cosine"))
        return {
            "model": model_name,
            "k": int(k_eval),
            "NMI": float(normalized_mutual_info_score(y_true_labels, y_pred)),
            "ARI": float(adjusted_rand_score(y_true_labels, y_pred)),
            "Silhouette": sil,
        }


    def build_cosine_index(embs: np.ndarray):
        matrix = np.asarray(embs, dtype="float32").copy()
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return index


    def eval_retrieval_metrics(embs: np.ndarray, labels: list[str], ks: tuple[int, ...]) -> dict:
        matrix = np.asarray(embs, dtype="float32")
        index = build_cosine_index(matrix)
        _, neighbors = index.search(matrix, max(ks) + 1)
        neighbors = neighbors[:, 1:]
        label_arr = np.asarray(labels)
        rr_values = []
        recalls = {k: [] for k in ks}
        for idx in range(len(label_arr)):
            target = label_arr[idx]
            hits = label_arr[neighbors[idx]] == target
            hit_positions = np.where(hits)[0]
            rr_values.append(float(1.0 / (hit_positions[0] + 1)) if len(hit_positions) else 0.0)
            for k in ks:
                recalls[k].append(float(hits[:k].any()))
        out = {"MRR": float(np.mean(rr_values))}
        for k in ks:
            out[f"Recall@{k}"] = float(np.mean(recalls[k]))
        return out


    def nearest_centroid_accuracy(embs: np.ndarray, labels: list[str]) -> float:
        labels_arr = np.asarray(labels)
        classes = np.unique(labels_arr)
        centroids = np.vstack([embs[labels_arr == cls].mean(axis=0) for cls in classes]).astype("float32")
        embs_norm = np.asarray(embs, dtype="float32")
        embs_norm = embs_norm / np.linalg.norm(embs_norm, axis=1, keepdims=True)
        cent_norm = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        sims = embs_norm @ cent_norm.T
        pred = classes[np.argmax(sims, axis=1)]
        return float(accuracy_score(labels_arr, pred))


    df_clu = pd.DataFrame(
        [
            eval_clustering_metrics(emb_base, y_true, "SciBERT-baseline"),
            eval_clustering_metrics(emb_ft, y_true, "SciBERT-SolarPhysics-Search"),
        ]
    )
    df_clu.to_csv(REPORTS_DIR / "eval_clustering_tecnica.csv", index=False)

    df_ret = pd.DataFrame(
        [
            {"model": "SciBERT-baseline", **eval_retrieval_metrics(emb_base, y_tecnica, AB_RECALL_KS)},
            {"model": "SciBERT-SolarPhysics-Search", **eval_retrieval_metrics(emb_ft, y_tecnica, AB_RECALL_KS)},
        ]
    )
    df_ret.to_csv(REPORTS_DIR / "eval_retrieval_tecnica.csv", index=False)

    df_nc = pd.DataFrame(
        [
            {"model": "SciBERT-baseline", "NearestCentroidAcc": nearest_centroid_accuracy(emb_base, y_tecnica)},
            {"model": "SciBERT-SolarPhysics-Search", "NearestCentroidAcc": nearest_centroid_accuracy(emb_ft, y_tecnica)},
        ]
    )
    df_nc.to_csv(REPORTS_DIR / "eval_nearest_centroid.csv", index=False)

    fig_path = REPORTS_DIR / "umap_baseline_vs_ft.png"
    umap_take = min(AB_UMAP_SAMPLE, len(eval_df))
    if umap_take >= 10:
        rng = np.random.default_rng(42)
        plot_idx = np.sort(rng.choice(len(eval_df), size=umap_take, replace=False))
        umap_base = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42).fit_transform(emb_base[plot_idx])
        umap_ft = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42).fit_transform(emb_ft[plot_idx])

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        axes[0].scatter(umap_base[:, 0], umap_base[:, 1], s=5, alpha=0.65)
        axes[0].set_title("SciBERT baseline - UMAP")
        axes[0].set_xlabel("UMAP-1")
        axes[0].set_ylabel("UMAP-2")

        axes[1].scatter(umap_ft[:, 0], umap_ft[:, 1], s=5, alpha=0.65, color="tab:blue")
        axes[1].set_title("SciBERT-SolarPhysics-Search - UMAP")
        axes[1].set_xlabel("UMAP-1")
        axes[1].set_ylabel("UMAP-2")

        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close(fig)
        log(f"[ab] figura UMAP salva em {fig_path}")
    else:
        log("[ab] amostra insuficiente para UMAP; figura nao gerada.")

    report_path = REPORTS_DIR / "AB_experimento_baseline_vs_finetuned.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("# Experimento A/B - SciBERT (baseline) vs SciBERT-SolarPhysics-Search\n\n")
        fh.write(f"- docs avaliados: {len(eval_df)}\n")
        fh.write(f"- classes de tecnica: {eval_df['Tecnica_clean'].nunique()}\n")
        fh.write(f"- corpus de treino: {', '.join(TRAIN_CORPORA)}\n\n")
        fh.write("## Clusterizacao (Tecnica)\n")
        fh.write(df_clu.to_markdown(index=False))
        fh.write("\n\n## Retrieval por classe (Tecnica)\n")
        fh.write(df_ret.to_markdown(index=False))
        fh.write("\n\n## Acuracia por centroide\n")
        fh.write(df_nc.to_markdown(index=False))
        fh.write("\n")
        if fig_path.exists():
            fh.write("\n## Figura UMAP\n")
            fh.write(f"![UMAP]({fig_path.name})\n")

    ab_summary = {
        "status": "completed",
        "eval_docs": int(len(eval_df)),
        "eval_classes": int(eval_df["Tecnica_clean"].nunique()),
        "min_technique_label_freq": int(MIN_TECHNIQUE_LABEL_FREQ),
        "max_ab_eval_docs": int(MAX_AB_EVAL_DOCS) if MAX_AB_EVAL_DOCS else None,
        "umap_generated": bool(fig_path.exists()),
        "run_ts": RUN_TS,
    }
    with open(REPORTS_DIR / "historical_ab_eval_summary.json", "w", encoding="utf-8") as fh:
        json.dump(ab_summary, fh, indent=2, ensure_ascii=False)

    log(
        f"[ab] concluido | docs={len(eval_df)} | classes={eval_df['Tecnica_clean'].nunique()} | "
        f"report={report_path}"
    )
    display(df_clu)
    display(df_ret)
    display(df_nc)
'''


SCIBERT_PUBLISH = r'''
# ============================================================
# Publicacao opcional do modelo no Hugging Face Hub
# ============================================================

stage_banner("PUBLICACAO OPCIONAL NO HUGGING FACE")

hf_report = {
    "run_hf_publish": bool(RUN_HF_PUBLISH),
    "hf_repo_id": HF_REPO_ID,
    "hf_private": bool(HF_PRIVATE),
    "status": "not_started",
    "run_ts": RUN_TS,
}

if not RUN_HF_PUBLISH:
    hf_report["status"] = "skipped_by_configuration"
    log("[hf] publicacao desativada por configuracao.")
else:
    from huggingface_hub import login

    token = None
    if str(HF_TOKEN_ENV).startswith("hf_"):
        token = str(HF_TOKEN_ENV).strip()
        hf_report["auth_mode"] = "direct_token_config"
        log("[hf] token direto detectado em HF_TOKEN_ENV. Recomendado migrar para os.environ['HF_TOKEN'].")
    else:
        token = os.environ.get(HF_TOKEN_ENV, "").strip() or None
    if not token:
        hf_report["status"] = "skipped_missing_token"
        hf_report["error"] = (
            f"Variavel {HF_TOKEN_ENV} ausente. Defina um token Hugging Face com permissao de escrita "
            "antes de rerodar esta celula."
        )
        log(f"[hf] {HF_TOKEN_ENV} ausente. Pulando a publicacao sem derrubar o notebook.")
    else:
        try:
            login(token=token, add_to_git_credential=False)
            hf_report["auth_mode"] = "env_token"
            log(f"[hf] autenticado via variavel de ambiente {HF_TOKEN_ENV}.")

            retrieval_path = REPORTS_DIR / "eval_retrieval_tecnica.csv"
            clustering_path = REPORTS_DIR / "eval_clustering_tecnica.csv"
            nearest_centroid_path = REPORTS_DIR / "eval_nearest_centroid.csv"
            dapt_metrics_path = REPORTS_DIR / "dapt_eval_metrics.json"

            model_card_lines = [
                "---",
                "license: apache-2.0",
                "library_name: sentence-transformers",
                f"base_model: {BASE_MODEL_ID}",
                "tags:",
                "- solar-physics",
                "- scientific-retrieval",
                "- sentence-transformers",
                "- scibert",
                "- major-review-rebuild",
                "---",
                "",
                "# SciBERT-SolarPhysics-Search",
                "",
                "Modelo canonico reconstruido para o major review, treinado somente em `Nucleo_core`, `PIML_core` e `CombFinal_core`.",
                "",
                "## Metodo",
                "",
                "- `DAPT` (MLM) sobre o corpus de dominio do `core`.",
                "- Fine-tuning contrastivo com `query_side -> positive_side`.",
                "- Sem uso de `ML_Multimodal` no treino.",
                "",
                "## Compatibilidade com o paper",
                "",
                "- O notebook 05 calcula `perplexity`, `NMI`, `ARI`, `Silhouette`, `MRR`, `Recall@K` e `NearestCentroidAcc`.",
                "- Os notebooks 06 a 08 calculam o incremento aprovado do major review (`SciBERT generic`, `BM25`, core vs holdout, bootstrap CIs e auditoria final).",
                "",
            ]

            if dapt_metrics_path.exists():
                dapt_metrics = json.loads(dapt_metrics_path.read_text(encoding="utf-8"))
                model_card_lines.extend(
                    [
                        "## DAPT",
                        "",
                        f"- eval_loss: {dapt_metrics.get('eval_loss')}",
                        f"- perplexity: {dapt_metrics.get('perplexity')}",
                        "",
                    ]
                )

            if retrieval_path.exists():
                model_card_lines.extend(
                    [
                        "## Retrieval por classe (Tecnica)",
                        "",
                        pd.read_csv(retrieval_path).to_markdown(index=False),
                        "",
                    ]
                )

            if clustering_path.exists():
                model_card_lines.extend(
                    [
                        "## Clusterizacao (Tecnica)",
                        "",
                        pd.read_csv(clustering_path).to_markdown(index=False),
                        "",
                    ]
                )

            if nearest_centroid_path.exists():
                model_card_lines.extend(
                    [
                        "## Separabilidade por centroide",
                        "",
                        pd.read_csv(nearest_centroid_path).to_markdown(index=False),
                        "",
                    ]
                )

            (FINAL_MODEL_DIR / "README.md").write_text("\n".join(model_card_lines), encoding="utf-8")

            publish_model = SentenceTransformer(str(FINAL_MODEL_DIR), device="cuda" if torch.cuda.is_available() else "cpu")
            push_result = publish_model.push_to_hub(
                HF_REPO_ID,
                token=token,
                private=HF_PRIVATE,
                commit_message=HF_COMMIT_MESSAGE,
                exist_ok=True,
                replace_model_card=True,
            )
            hf_report["status"] = "published"
            hf_report["push_result"] = push_result
            log(f"[hf] publicacao concluida em {HF_REPO_ID}")
        except Exception as exc:
            hf_report["status"] = "failed"
            hf_report["error"] = repr(exc)
            log(f"[hf] falha na publicacao: {exc}")

with open(REPORTS_DIR / "hf_publish_report.json", "w", encoding="utf-8") as fh:
    json.dump(hf_report, fh, indent=2, ensure_ascii=False)

display(pd.DataFrame([hf_report]))
'''


SCIBERT_EXPORT = r'''
# ============================================================
# Auditoria final, artefatos e manifesto
# ============================================================

stage_banner("AUDITORIA FINAL DO RETRIEVER")

final_model = SentenceTransformer(str(FINAL_MODEL_DIR), device="cuda" if torch.cuda.is_available() else "cpu")
sample_df = train_df.groupby("corpus", group_keys=False).head(400).copy()
sample_texts = sample_df["text_full"].fillna("").astype(str).tolist()
sample_emb = final_model.encode(sample_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype("float32")
faiss.normalize_L2(sample_emb)

weak_query_rows = []
for corpus in TRAIN_CORPORA:
    cards_path = read_topic_cards_csv(corpus)
    if not cards_path.exists():
        continue
    cards = pd.read_csv(cards_path).head(TOPICS_PER_CORPUS_FOR_AUDIT)
    for row in cards.itertuples(index=False):
        query_text = f"{getattr(row, 'title', '')}. {getattr(row, 'keywords', '')}".strip()
        if len(query_text.split()) < 3:
            continue
        q_emb = final_model.encode([query_text], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        scores = np.matmul(sample_emb, q_emb[0])
        top_idx = np.argsort(scores)[::-1][:5]
        for rank, idx in enumerate(top_idx, start=1):
            hit = sample_df.iloc[int(idx)]
            weak_query_rows.append(
                {
                    "source_corpus": corpus,
                    "query_text": query_text,
                    "rank": rank,
                    "score": float(scores[idx]),
                    "hit_corpus": hit["corpus"],
                    "hit_title": hit["TI"],
                    "hit_year": hit["PY"],
                    "hit_doi": hit["DI"],
                }
            )

weak_query_df = pd.DataFrame(weak_query_rows)
weak_query_df.to_csv(ARTIFACTS_DIR / "weak_query_neighbors.csv", index=False)

manifest_rows = []
for path in sorted(WRITE_ROOT.rglob("*")):
    if path.is_file():
        manifest_rows.append(
            {
                "artifact": str(path),
                "size_bytes": path.stat().st_size,
            }
        )

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(REPORTS_DIR / "artifact_manifest.csv", index=False)

training_manifest = {
    "train_corpora": TRAIN_CORPORA,
    "n_train_docs": int(len(train_df)),
    "n_dapt_texts": int(len(dapt_texts)),
    "n_contrastive_pairs": int(len(pair_df)),
    "base_model_id": BASE_MODEL_ID,
    "run_dapt": RUN_DAPT,
    "run_contrastive": RUN_CONTRASTIVE,
    "run_historical_ab_eval": RUN_HISTORICAL_AB_EVAL,
    "run_hf_publish": RUN_HF_PUBLISH,
    "hf_repo_id": HF_REPO_ID,
    "run_ts": RUN_TS,
}
with open(REPORTS_DIR / "training_manifest.json", "w", encoding="utf-8") as fh:
    json.dump(training_manifest, fh, indent=2, ensure_ascii=False)

display(pd.read_csv(ARTIFACTS_DIR / "training_input_summary.csv"))
display(weak_query_df.head(20))
display(manifest_df.head(30))

print("Artefatos finais salvos em:", WRITE_ROOT)
'''


PIPE2_MD = """
# 06 Pipe 2 - retriever analytics rebuild

Notebook canonico da Camada 2 para reconstruir a ponte entre:

- corpora canonicos do `00_consolidacao`;
- retriever especializado do `05_scibert_solarphysics_search`;
- corpus de aplicacao `ML_Multimodal`;
- artefatos de consulta derivados do `core`.

## Papel metodologico

Este notebook:

1. monta `docs_master` e `index_map` para `ML_core` e `ML_holdout`;
2. gera embeddings e indices FAISS para `SciBERT` generico e `SciBERT-SolarPhysics-Search`;
3. constroi um banco de queries derivado **apenas do core**;
4. executa analytics de recuperacao e produz candidatos iniciais de white space;
5. salva logs, auditorias e artefatos no Google Drive.
"""


PIPE2_INSTALL = r'''
# ============================================================
# Instalacao de dependencias para Colab
# ============================================================
!pip install -U -q pip setuptools wheel
!pip uninstall -y -q numpy scipy scikit-learn sentence-transformers transformers faiss-cpu numba umap-learn hdbscan || true
!pip install -U -q numpy==2.0.2 scipy==1.14.1 pandas==2.2.2 scikit-learn==1.6.1 numba==0.60.0 openpyxl pyarrow jedi==0.19.2
!pip install -U -q faiss-cpu sentence-transformers==2.7.0 transformers==4.45.2

import numpy, scipy, sklearn, sentence_transformers, transformers
print("numpy            :", numpy.__version__)
print("scipy            :", scipy.__version__)
print("scikit-learn     :", sklearn.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("transformers     :", transformers.__version__)

print("Dependencias instaladas.")
'''


PIPE2_IMPORTS = r'''
from google.colab import drive
drive.mount("/content/drive")

import ast
import hashlib
import json
import math
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 240)

print("Torch:", torch.__version__)
print("GPU disponivel:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
'''


PIPE2_CONFIG = r'''
# ============================================================
# Configuracao geral
# ============================================================

DRIVE_ROOT = Path("/content/drive/MyDrive/Unicamp")
PROJECT_ROOT = DRIVE_ROOT / "artigo bibliometria" / "grounded-scientometrics-solarphysics-retrieval"
DATA_ROOT = DRIVE_ROOT / "artigo bibliometria" / "base de dados" / "Artigo_Bibliometria Base Bruta" / "BASES_UNIFICADAS_POR_TEMA"

TARGET_CORPUS = "ML_Multimodal"
READ_STAGE_CONSOL = "00_consolidacao"
READ_STAGE_ABSTRACT = "01_abstract_llm"
WRITE_STAGE = "06_pipe2_retriever_analytics"
MODEL_ROOT = DATA_ROOT / "_cross_corpus_rebuild" / "05_scibert_solarphysics_search" / "checkpoints" / "SciBERT-SolarPhysics-Search"

PERIODS = ["core", "holdout"]
QUERY_CORPORA = ["Nucleo", "PIML", "CombFinal"]
GENERIC_MODEL_ID = "allenai/scibert_scivocab_uncased"
TOPICS_PER_CORPUS = 20
TOPK = 20
EMBED_BATCH_SIZE = 64

assert PROJECT_ROOT.exists(), f"PROJECT_ROOT nao encontrado: {PROJECT_ROOT}"
assert DATA_ROOT.exists(), f"DATA_ROOT nao encontrado: {DATA_ROOT}"
assert MODEL_ROOT.exists(), f"Modelo especializado nao encontrado: {MODEL_ROOT}"

print("MODEL_ROOT =", MODEL_ROOT)
print("TOPK =", TOPK)
'''


PIPE2_LOGGING = r'''
# ============================================================
# Saidas, logs e helpers
# ============================================================

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
PIPE_START_TS = time.time()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


WRITE_ROOT = ensure_dir(DATA_ROOT / TARGET_CORPUS / "04_rebuild_outputs" / WRITE_STAGE)
GLOBAL_LOG_DIR = ensure_dir(PROJECT_ROOT / "outputs" / "camada2_logs")
GLOBAL_LOG_FILE = GLOBAL_LOG_DIR / f"06_pipe2_{RUN_TS}.txt"


def period_root(period: str) -> Path:
    return ensure_dir(WRITE_ROOT / period)


for period in PERIODS:
    ensure_dir(period_root(period) / "indices")
    ensure_dir(period_root(period) / "tables")
    ensure_dir(period_root(period) / "reports")

ensure_dir(WRITE_ROOT / "tables")
ensure_dir(WRITE_ROOT / "reports")


def elapsed_seconds() -> float:
    return time.time() - PIPE_START_TS


def fmt_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{now} | +{fmt_seconds(elapsed_seconds())}]"
    line = f"{prefix} {message}"
    print(line, flush=True)
    GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def stage_banner(title: str) -> None:
    bar = "=" * 96
    log(bar)
    log(title)
    log(bar)


def period_input_csv(period: str) -> Path:
    return DATA_ROOT / TARGET_CORPUS / "04_rebuild_outputs" / READ_STAGE_CONSOL / f"{TARGET_CORPUS}_{period}_bibliometrix_clean.csv"


print("WRITE_ROOT =", WRITE_ROOT)
print("GLOBAL_LOG_FILE =", GLOBAL_LOG_FILE)
'''


PIPE2_DOCS = r'''
# ============================================================
# Reconstrucao de docs_master e index_map por periodo
# ============================================================

EXPECTED_COLS = ["TI", "AB", "DE", "ID", "SO", "PY", "TC", "DI", "publication_date"]


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


def normalize_text(value):
    text = clean_text(value)
    if pd.isna(text):
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def stable_doc_id(row: pd.Series, period: str) -> str:
    seed = "|".join(
        [
            period,
            normalize_text(row.get("TI")),
            normalize_text(row.get("DI")),
            normalize_text(row.get("PY")),
            normalize_text(row.get("SO")),
        ]
    )
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:16]


def ensure_expected_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EXPECTED_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def build_docs_master(period: str) -> pd.DataFrame:
    csv_path = period_input_csv(period)
    assert csv_path.exists(), f"Input nao encontrado: {csv_path}"
    raw = pd.read_csv(csv_path, dtype=str, low_memory=False)
    raw = ensure_expected_cols(raw)
    raw["period"] = period
    raw["title"] = raw["TI"].map(normalize_text)
    raw["abstract"] = raw["AB"].map(normalize_text)
    raw["keywords"] = (raw["DE"].fillna("").astype(str) + "; " + raw["ID"].fillna("").astype(str)).map(normalize_text)
    raw["source"] = raw["SO"].map(normalize_text)
    raw["year"] = pd.to_numeric(raw["PY"], errors="coerce").astype("Int64")
    raw["citations"] = pd.to_numeric(raw["TC"], errors="coerce").fillna(0)
    raw["doi"] = raw["DI"].map(normalize_text)
    raw["text_full"] = (
        raw["title"].fillna("").astype(str)
        + " [SEP] "
        + raw["abstract"].fillna("").astype(str)
        + " [SEP] "
        + raw["keywords"].fillna("").astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    raw["doc_id"] = raw.apply(lambda row: stable_doc_id(row, period), axis=1)
    raw["faiss_pos"] = np.arange(len(raw))

    docs_master = raw[
        ["doc_id", "faiss_pos", "period", "title", "abstract", "keywords", "source", "year", "citations", "doi", "publication_date", "text_full"]
    ].copy()
    docs_master = docs_master.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    docs_master["faiss_pos"] = np.arange(len(docs_master))

    log(
        f"[docs] {period} | rows={len(docs_master)} | "
        f"abstract_pct={round(docs_master['abstract'].astype(str).str.len().gt(0).mean() * 100, 2) if len(docs_master) else 0.0}%"
    )
    return docs_master


stage_banner("RECONSTRUCAO DO CORPUS DE APLICACAO")

period_docs = {}
coverage_rows = []
for period in PERIODS:
    docs = build_docs_master(period)
    docs.to_csv(period_root(period) / "tables" / f"{TARGET_CORPUS}_{period}_docs_master.csv", index=False)
    docs[["faiss_pos", "doc_id", "period"]].to_csv(period_root(period) / "tables" / f"{TARGET_CORPUS}_{period}_index_map.csv", index=False)
    period_docs[period] = docs
    coverage_rows.append(
        {
            "period": period,
            "rows": int(len(docs)),
            "abstract_present_pct": round(docs["abstract"].astype(str).str.len().gt(0).mean() * 100, 2) if len(docs) else 0.0,
            "doi_present_pct": round(docs["doi"].astype(str).str.len().gt(0).mean() * 100, 2) if len(docs) else 0.0,
            "mean_year": round(float(docs["year"].dropna().astype(float).mean()), 2) if docs["year"].notna().any() else None,
        }
    )

coverage_df = pd.DataFrame(coverage_rows)
coverage_df.to_csv(WRITE_ROOT / "tables" / "ml_docs_master_coverage.csv", index=False)
display(coverage_df)
'''


PIPE2_EMBED = r'''
# ============================================================
# Embeddings e indices FAISS por periodo
# ============================================================

stage_banner("EMBEDDINGS E INDICES")

device = "cuda" if torch.cuda.is_available() else "cpu"
generic_model = SentenceTransformer(GENERIC_MODEL_ID, device=device)
specialized_model = SentenceTransformer(str(MODEL_ROOT), device=device)
log(f"[model] generic={GENERIC_MODEL_ID} | specialized={MODEL_ROOT} | device={device}")


def encode_texts(model, texts: list[str], batch_size: int, period: str, label: str) -> np.ndarray:
    all_rows = []
    total_batches = math.ceil(len(texts) / batch_size) if texts else 0
    for batch_idx, start in enumerate(range(0, len(texts), batch_size), start=1):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False).astype("float32")
        all_rows.append(emb)
        if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == total_batches:
            log(f"[embed] {period} | {label} | batch {batch_idx}/{total_batches} | rows {end}/{len(texts)}")
    return np.vstack(all_rows) if all_rows else np.zeros((0, 768), dtype="float32")


embedding_manifest_rows = []
for period in PERIODS:
    docs = period_docs[period]
    texts = docs["text_full"].fillna("").astype(str).tolist()
    for label, model in [("generic", generic_model), ("specialized", specialized_model)]:
        emb = encode_texts(model, texts, EMBED_BATCH_SIZE, period, label)
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        np.save(period_root(period) / "indices" / f"{TARGET_CORPUS}_{period}_{label}_embeddings.npy", emb)
        faiss.write_index(index, str(period_root(period) / "indices" / f"{TARGET_CORPUS}_{period}_{label}.faiss"))

        embedding_manifest_rows.append(
            {
                "period": period,
                "retriever": label,
                "rows": int(len(docs)),
                "embedding_dim": int(emb.shape[1]) if emb.size else 0,
                "index_ntotal": int(index.ntotal),
            }
        )

embedding_manifest_df = pd.DataFrame(embedding_manifest_rows)
embedding_manifest_df.to_csv(WRITE_ROOT / "tables" / "embedding_index_manifest.csv", index=False)
display(embedding_manifest_df)
'''


PIPE2_QUERIES = r'''
# ============================================================
# Banco de queries derivado apenas do core
# ============================================================

stage_banner("BANCO DE QUERIES DO CORE")


def parse_keywords(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except Exception:
        pass
    text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return [part.strip() for part in text.split(",") if part.strip()]


query_rows = []
for corpus in QUERY_CORPORA:
    cards_path = DATA_ROOT / corpus / "04_rebuild_outputs" / READ_STAGE_ABSTRACT / "core" / "tables" / f"{corpus}_core_topic_cards.csv"
    assert cards_path.exists(), f"Topic cards core nao encontrados: {cards_path}"
    cards = pd.read_csv(cards_path).head(TOPICS_PER_CORPUS).copy()
    log(f"[query-bank] {corpus} | rows={len(cards)} | path={cards_path}")
    for row in cards.itertuples(index=False):
        keywords = parse_keywords(getattr(row, "keywords", ""))
        title = str(getattr(row, "title", "")).strip()
        query_text = (title + ". " + " ".join(keywords[:6])).strip()
        query_text = re.sub(r"\s+", " ", query_text)
        if len(query_text.split()) < 3:
            continue
        query_rows.append(
            {
                "query_id": f"{corpus}_topic_{getattr(row, 'topic', 'na')}",
                "source_corpus": corpus,
                "topic": getattr(row, "topic", None),
                "query_text": query_text,
                "topic_size": getattr(row, "size", None),
                "keywords": "; ".join(keywords[:10]),
            }
        )

query_bank = pd.DataFrame(query_rows).drop_duplicates(subset=["query_id", "query_text"]).reset_index(drop=True)
query_bank.to_csv(WRITE_ROOT / "tables" / "core_gap_query_bank.csv", index=False)

log(f"[query-bank] total queries={len(query_bank)}")
display(query_bank.head(20))
'''


PIPE2_ANALYTICS = r'''
# ============================================================
# Analytics de recuperacao e candidatos de white space
# ============================================================

stage_banner("ANALYTICS DE RECUPERACAO")

assets = {}
for period in PERIODS:
    docs = pd.read_csv(period_root(period) / "tables" / f"{TARGET_CORPUS}_{period}_docs_master.csv")
    idx_map = pd.read_csv(period_root(period) / "tables" / f"{TARGET_CORPUS}_{period}_index_map.csv")
    assets[period] = {
        "docs": docs,
        "idx_map": idx_map,
        "generic": faiss.read_index(str(period_root(period) / "indices" / f"{TARGET_CORPUS}_{period}_generic.faiss")),
        "specialized": faiss.read_index(str(period_root(period) / "indices" / f"{TARGET_CORPUS}_{period}_specialized.faiss")),
    }


def semantic_search(model, index, docs: pd.DataFrame, query: str, topk: int, period: str, label: str) -> pd.DataFrame:
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, topk)
    hits = docs.iloc[idxs[0]].copy().reset_index(drop=True)
    hits["rank"] = np.arange(1, len(hits) + 1)
    hits["score"] = scores[0]
    hits["period"] = period
    hits["retriever"] = label
    return hits


semantic_hit_rows = []
metric_rows = []

for period in PERIODS:
    log(f"[analytics] iniciando periodo={period}")
    docs = assets[period]["docs"]
    total_queries = len(query_bank)
    for q_idx, q_row in enumerate(query_bank.itertuples(index=False), start=1):
        gen_hits = semantic_search(generic_model, assets[period]["generic"], docs, q_row.query_text, TOPK, period, "generic")
        spec_hits = semantic_search(specialized_model, assets[period]["specialized"], docs, q_row.query_text, TOPK, period, "specialized")

        gen_ids = set(gen_hits["doc_id"].head(10).tolist())
        spec_ids = set(spec_hits["doc_id"].head(10).tolist())
        overlap10 = len(gen_ids & spec_ids) / max(1, len(gen_ids | spec_ids))
        metric_rows.append(
            {
                "period": period,
                "query_id": q_row.query_id,
                "source_corpus": q_row.source_corpus,
                "query_text": q_row.query_text,
                "generic_top1_score": float(gen_hits["score"].iloc[0]),
                "specialized_top1_score": float(spec_hits["score"].iloc[0]),
                "score_lift_top1": float(spec_hits["score"].iloc[0] - gen_hits["score"].iloc[0]),
                "overlap_at_10": round(float(overlap10), 4),
                "specialized_mean_citations_top10": round(float(pd.to_numeric(spec_hits["citations"], errors="coerce").fillna(0).head(10).mean()), 2),
                "specialized_recent_share_top10": round(float(pd.to_numeric(spec_hits["year"], errors="coerce").fillna(0).head(10).ge(2023).mean()), 4),
            }
        )

        for hit_df in [gen_hits, spec_hits]:
            hit_df = hit_df.copy()
            hit_df["query_id"] = q_row.query_id
            hit_df["query_text"] = q_row.query_text
            hit_df["source_corpus"] = q_row.source_corpus
            semantic_hit_rows.append(hit_df)

        if q_idx == 1 or q_idx % 10 == 0 or q_idx == total_queries:
            log(f"[analytics] {period} | query {q_idx}/{total_queries}")

semantic_hits = pd.concat(semantic_hit_rows, ignore_index=True)
semantic_metrics = pd.DataFrame(metric_rows)
semantic_hits.to_csv(WRITE_ROOT / "tables" / "semantic_query_hits.csv", index=False)
semantic_metrics.to_csv(WRITE_ROOT / "tables" / "semantic_query_metrics.csv", index=False)

white_space = semantic_metrics.copy()
white_space["candidate_score"] = white_space["score_lift_top1"] + (1 - white_space["overlap_at_10"]) * 0.25
white_space = white_space.sort_values(["period", "candidate_score"], ascending=[True, False]).reset_index(drop=True)
white_space.to_csv(WRITE_ROOT / "tables" / "white_space_candidates.csv", index=False)

display(semantic_metrics.head(20))
display(white_space.head(20))
'''


PIPE2_MANIFEST = r'''
# ============================================================
# Manifesto final da camada 2
# ============================================================

stage_banner("MANIFESTO FINAL DA CAMADA 2")

manifest_rows = []
for path in sorted(WRITE_ROOT.rglob("*")):
    if path.is_file():
        manifest_rows.append(
            {
                "artifact": str(path),
                "size_bytes": path.stat().st_size,
            }
        )

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(WRITE_ROOT / "reports" / "artifact_manifest.csv", index=False)

display(pd.read_csv(WRITE_ROOT / "tables" / "ml_docs_master_coverage.csv"))
display(pd.read_csv(WRITE_ROOT / "tables" / "embedding_index_manifest.csv"))
display(manifest_df.head(30))

print("Arquivos finais salvos em:", WRITE_ROOT)
'''


PIPE3_MD = """
# 07 Pipe 3 - agent scientometrics rebuild

Notebook canonico da Camada 3 para:

1. carregar os artefatos de `06_pipe2_retriever_analytics`;
2. adicionar o baseline `BM25`;
3. montar bundles de evidencia por query e por retriever;
4. produzir saídas experimentais replicáveis para `core` e `holdout`;
5. opcionalmente gerar resumos grounded por LLM sem quebrar a rastreabilidade.

## Observacao metodologica

- O banco de queries continua derivado do `core`.
- O `holdout` espelha a mesma familia de aplicacoes.
- Sem chave de API, o notebook ainda produz o experimento retrieval-only completo.
"""


PIPE3_INSTALL = r'''
# ============================================================
# Instalacao de dependencias para Colab
# ============================================================
!pip install -U -q pip setuptools wheel
!pip uninstall -y -q numpy scipy scikit-learn sentence-transformers transformers faiss-cpu numba umap-learn hdbscan || true
!pip install -U -q numpy==2.0.2 scipy==1.14.1 pandas==2.2.2 scikit-learn==1.6.1 numba==0.60.0 openpyxl pyarrow jedi==0.19.2
!pip install -U -q faiss-cpu sentence-transformers==2.7.0 transformers==4.45.2 rank-bm25==0.2.2
!pip install -U -q openai==1.* pydantic

import numpy, scipy, sklearn, sentence_transformers, transformers
print("numpy            :", numpy.__version__)
print("scipy            :", scipy.__version__)
print("scikit-learn     :", sklearn.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("transformers     :", transformers.__version__)

print("Dependencias instaladas.")
'''


PIPE3_IMPORTS = r'''
from google.colab import drive
drive.mount("/content/drive")

import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from rank_bm25 import BM25Okapi

pd.set_option("display.max_columns", 220)
pd.set_option("display.max_colwidth", 260)

print("OPENAI_API_KEY presente:", bool(os.getenv("OPENAI_API_KEY")))
'''


PIPE3_CONFIG = r'''
# ============================================================
# Configuracao geral
# ============================================================

DRIVE_ROOT = Path("/content/drive/MyDrive/Unicamp")
PROJECT_ROOT = DRIVE_ROOT / "artigo bibliometria" / "grounded-scientometrics-solarphysics-retrieval"
DATA_ROOT = DRIVE_ROOT / "artigo bibliometria" / "base de dados" / "Artigo_Bibliometria Base Bruta" / "BASES_UNIFICADAS_POR_TEMA"

TARGET_CORPUS = "ML_Multimodal"
READ_STAGE = "06_pipe2_retriever_analytics"
WRITE_STAGE = "07_pipe3_agent_scientometrics"
PERIODS = ["core", "holdout"]
TOPK = 20
MAX_LLM_QUERIES = 30
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_OPENAI_IF_AVAILABLE = True

PIPE2_ROOT = DATA_ROOT / TARGET_CORPUS / "04_rebuild_outputs" / READ_STAGE
WRITE_ROOT = DATA_ROOT / TARGET_CORPUS / "04_rebuild_outputs" / WRITE_STAGE

assert PIPE2_ROOT.exists(), f"PIPE2_ROOT nao encontrado: {PIPE2_ROOT}"
print("PIPE2_ROOT =", PIPE2_ROOT)
print("WRITE_ROOT =", WRITE_ROOT)
'''


PIPE3_LOGGING = r'''
# ============================================================
# Saidas, logs e helpers
# ============================================================

RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
PIPE_START_TS = time.time()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


ensure_dir(WRITE_ROOT)
ensure_dir(WRITE_ROOT / "tables")
ensure_dir(WRITE_ROOT / "reports")
for period in PERIODS:
    ensure_dir(WRITE_ROOT / period / "tables")
    ensure_dir(WRITE_ROOT / period / "reports")

GLOBAL_LOG_DIR = ensure_dir(PROJECT_ROOT / "outputs" / "camada3_logs")
GLOBAL_LOG_FILE = GLOBAL_LOG_DIR / f"07_pipe3_{RUN_TS}.txt"


def elapsed_seconds() -> float:
    return time.time() - PIPE_START_TS


def fmt_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{now} | +{fmt_seconds(elapsed_seconds())}]"
    line = f"{prefix} {message}"
    print(line, flush=True)
    GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def stage_banner(title: str) -> None:
    bar = "=" * 96
    log(bar)
    log(title)
    log(bar)
'''


PIPE3_LOAD = r'''
# ============================================================
# Leitura dos artefatos do Pipe 2
# ============================================================

stage_banner("LEITURA DOS ARTEFATOS DO PIPE 2")

query_bank = pd.read_csv(PIPE2_ROOT / "tables" / "core_gap_query_bank.csv")
semantic_hits = pd.read_csv(PIPE2_ROOT / "tables" / "semantic_query_hits.csv")
semantic_metrics = pd.read_csv(PIPE2_ROOT / "tables" / "semantic_query_metrics.csv")
white_space = pd.read_csv(PIPE2_ROOT / "tables" / "white_space_candidates.csv")

period_docs = {
    period: pd.read_csv(PIPE2_ROOT / period / "tables" / f"{TARGET_CORPUS}_{period}_docs_master.csv")
    for period in PERIODS
}

log(f"[load] queries={len(query_bank)} | semantic_hits={len(semantic_hits)} | white_space={len(white_space)}")
display(query_bank.head(10))
display(white_space.head(10))
'''


PIPE3_BM25 = r'''
# ============================================================
# Baseline BM25
# ============================================================

stage_banner("BM25")


def tokenize_for_bm25(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return [tok for tok in text.split() if tok]


bm25_models = {}
bm25_tokens = {}
for period in PERIODS:
    docs = period_docs[period].copy()
    docs["bm25_text"] = (docs["title"].fillna("") + " " + docs["abstract"].fillna("") + " " + docs["keywords"].fillna("")).astype(str)
    tokens = [tokenize_for_bm25(text) for text in docs["bm25_text"].tolist()]
    bm25_models[period] = BM25Okapi(tokens)
    bm25_tokens[period] = tokens
    log(f"[bm25] {period} | docs={len(docs)}")


bm25_rows = []
for period in PERIODS:
    docs = period_docs[period]
    total_queries = len(query_bank)
    for q_idx, q_row in enumerate(query_bank.itertuples(index=False), start=1):
        scores = bm25_models[period].get_scores(tokenize_for_bm25(q_row.query_text))
        top_idx = np.argsort(scores)[::-1][:TOPK]
        hits = docs.iloc[top_idx].copy().reset_index(drop=True)
        hits["rank"] = np.arange(1, len(hits) + 1)
        hits["score"] = [float(scores[i]) for i in top_idx]
        hits["period"] = period
        hits["retriever"] = "bm25"
        hits["query_id"] = q_row.query_id
        hits["query_text"] = q_row.query_text
        hits["source_corpus"] = q_row.source_corpus
        bm25_rows.append(hits)
        if q_idx == 1 or q_idx % 10 == 0 or q_idx == total_queries:
            log(f"[bm25] {period} | query {q_idx}/{total_queries}")

bm25_hits = pd.concat(bm25_rows, ignore_index=True)
bm25_hits.to_csv(WRITE_ROOT / "tables" / "bm25_query_hits.csv", index=False)
display(bm25_hits.head(20))
'''


PIPE3_BUNDLES = r'''
# ============================================================
# Bundles de evidencia e metricas por retriever
# ============================================================

stage_banner("BUNDLES DE EVIDENCIA")

semantic_hits = semantic_hits.rename(columns={"retriever": "retriever"})
all_hits = pd.concat([semantic_hits, bm25_hits], ignore_index=True)
all_hits.to_csv(WRITE_ROOT / "tables" / "retrieval_hits_all.csv", index=False)

metric_rows = []
bundle_rows = []

for period in PERIODS:
    ws_period = white_space[white_space["period"] == period].copy()
    ws_lookup = ws_period.set_index("query_id") if len(ws_period) else None
    for retriever in ["generic", "specialized", "bm25"]:
        sub = all_hits[(all_hits["period"] == period) & (all_hits["retriever"] == retriever)].copy()
        for query_id, q_sub in sub.groupby("query_id", sort=False):
            q_sub = q_sub.sort_values("rank").copy()
            metric_rows.append(
                {
                    "period": period,
                    "retriever": retriever,
                    "query_id": query_id,
                    "query_text": q_sub["query_text"].iloc[0],
                    "source_corpus": q_sub["source_corpus"].iloc[0],
                    "top1_score": float(q_sub["score"].iloc[0]),
                    "mean_topk_score": round(float(q_sub["score"].head(TOPK).mean()), 4),
                    "mean_topk_citations": round(float(pd.to_numeric(q_sub["citations"], errors="coerce").fillna(0).head(TOPK).mean()), 2),
                    "recent_share_topk": round(float(pd.to_numeric(q_sub["year"], errors="coerce").fillna(0).head(TOPK).ge(2023).mean()), 4),
                    "distinct_sources_topk": int(q_sub["source"].fillna("").head(TOPK).nunique()),
                    "white_space_candidate_score": float(ws_lookup.loc[query_id, "candidate_score"]) if ws_lookup is not None and query_id in ws_lookup.index else np.nan,
                }
            )

            bundle_rows.append(
                {
                    "period": period,
                    "retriever": retriever,
                    "query_id": query_id,
                    "query_text": q_sub["query_text"].iloc[0],
                    "source_corpus": q_sub["source_corpus"].iloc[0],
                    "bundle_json": json.dumps(
                        q_sub.head(10)[["doc_id", "title", "year", "source", "score", "doi"]].to_dict(orient="records"),
                        ensure_ascii=False,
                    ),
                }
            )

metrics_df = pd.DataFrame(metric_rows)
bundles_df = pd.DataFrame(bundle_rows)
metrics_df.to_csv(WRITE_ROOT / "tables" / "retriever_query_metrics.csv", index=False)
bundles_df.to_csv(WRITE_ROOT / "tables" / "evidence_bundles.csv", index=False)

display(metrics_df.head(20))
display(bundles_df.head(10))
'''


PIPE3_AGENT = r'''
# ============================================================
# Saida agent-like grounded (OpenAI opcional)
# ============================================================

stage_banner("SAIDA AGENT-LIKE")

candidate_queries = (
    white_space.sort_values(["period", "candidate_score"], ascending=[True, False])
    .groupby("period", group_keys=False)
    .head(MAX_LLM_QUERIES)
    .copy()
)

client = OpenAI() if (USE_OPENAI_IF_AVAILABLE and os.getenv("OPENAI_API_KEY")) else None
agent_rows = []

for idx, row in enumerate(candidate_queries.itertuples(index=False), start=1):
    specialized_bundle = bundles_df[
        (bundles_df["period"] == row.period)
        & (bundles_df["retriever"] == "specialized")
        & (bundles_df["query_id"] == row.query_id)
    ]
    evidence_json = specialized_bundle["bundle_json"].iloc[0] if len(specialized_bundle) else "[]"

    if client is None:
        summary = f"Retrieval-only bundle for {row.query_id}. Query: {row.query_text}"
        status = "heuristic_only"
    else:
        prompt = f"""
Voce esta auditando gaps cientometricos.
Use somente as evidencias listadas abaixo.
Retorne um paragrafo curto com:
1. gap-title curto
2. justificativa grounded
3. doc_ids usados

QUERY: {row.query_text}
EVIDENCIAS: {evidence_json}
"""
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": "Seja objetivo e grounded nas evidencias."},
                    {"role": "user", "content": prompt},
                ],
            )
            summary = response.choices[0].message.content
            status = "openai_ok"
        except Exception as exc:
            summary = f"Falha na chamada OpenAI: {exc}"
            status = "openai_error"

    agent_rows.append(
        {
            "period": row.period,
            "query_id": row.query_id,
            "query_text": row.query_text,
            "source_corpus": row.source_corpus,
            "candidate_score": row.candidate_score,
            "agent_status": status,
            "agent_output": summary,
        }
    )

    if idx == 1 or idx % 5 == 0 or idx == len(candidate_queries):
        log(f"[agent] processed {idx}/{len(candidate_queries)}")

agent_df = pd.DataFrame(agent_rows)
agent_df.to_csv(WRITE_ROOT / "tables" / "agent_outputs.csv", index=False)
with open(WRITE_ROOT / "reports" / "agent_outputs.jsonl", "w", encoding="utf-8") as fh:
    for row in agent_rows:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

display(agent_df.head(20))
'''


PIPE3_EXPORT = r'''
# ============================================================
# Tabelas do artigo e manifesto final
# ============================================================

stage_banner("EXPORT FINAL DO PIPE 3")

paper_summary = (
    metrics_df.groupby(["period", "retriever"], as_index=False)
    .agg(
        n_queries=("query_id", "nunique"),
        mean_top1_score=("top1_score", "mean"),
        mean_mean_topk_score=("mean_topk_score", "mean"),
        mean_topk_citations=("mean_topk_citations", "mean"),
        mean_recent_share=("recent_share_topk", "mean"),
        mean_distinct_sources=("distinct_sources_topk", "mean"),
    )
)
paper_summary.to_csv(WRITE_ROOT / "tables" / "paper_ready_retriever_summary.csv", index=False)

with pd.ExcelWriter(WRITE_ROOT / "reports" / "pipe3_paper_tables.xlsx", engine="openpyxl") as writer:
    paper_summary.to_excel(writer, sheet_name="retriever_summary", index=False)
    metrics_df.to_excel(writer, sheet_name="query_metrics", index=False)
    agent_df.to_excel(writer, sheet_name="agent_outputs", index=False)

manifest_rows = []
for path in sorted(WRITE_ROOT.rglob("*")):
    if path.is_file():
        manifest_rows.append({"artifact": str(path), "size_bytes": path.stat().st_size})

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(WRITE_ROOT / "reports" / "artifact_manifest.csv", index=False)

display(paper_summary)
display(manifest_df.head(30))
print("Arquivos finais salvos em:", WRITE_ROOT)
'''


PIPE4_MD = """
# 08 Pipe 4 - statistical validation rebuild

Notebook canonico da Camada 4 para:

1. consumir os outputs experimentais do `07_pipe3_agent_scientometrics`;
2. calcular intervalos de confianca e deltas principais;
3. separar replicacao `core` de validacao temporal `holdout`;
4. gerar tabelas finais e uma amostra de auditoria manual pequena;
5. salvar o pacote estatistico final no Google Drive.
"""


PIPE4_INSTALL = r'''
# ============================================================
# Instalacao de dependencias para Colab
# ============================================================
!pip install -U -q pip setuptools wheel
!pip uninstall -y -q numpy scipy scikit-learn sentence-transformers transformers faiss-cpu numba umap-learn hdbscan || true
!pip install -U -q numpy==2.0.2 scipy==1.14.1 pandas==2.2.2 scikit-learn==1.6.1 openpyxl pyarrow jedi==0.19.2

import numpy, scipy, pandas, sklearn
print("numpy            :", numpy.__version__)
print("scipy            :", scipy.__version__)
print("pandas           :", pandas.__version__)
print("scikit-learn     :", sklearn.__version__)

print("Dependencias instaladas.")
'''


PIPE4_IMPORTS = r'''
from google.colab import drive
drive.mount("/content/drive")

import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

pd.set_option("display.max_columns", 220)
pd.set_option("display.max_colwidth", 260)
'''


PIPE4_CONFIG = r'''
# ============================================================
# Configuracao geral
# ============================================================

DRIVE_ROOT = Path("/content/drive/MyDrive/Unicamp")
PROJECT_ROOT = DRIVE_ROOT / "artigo bibliometria" / "grounded-scientometrics-solarphysics-retrieval"
DATA_ROOT = DRIVE_ROOT / "artigo bibliometria" / "base de dados" / "Artigo_Bibliometria Base Bruta" / "BASES_UNIFICADAS_POR_TEMA"

TARGET_CORPUS = "ML_Multimodal"
READ_STAGE = "07_pipe3_agent_scientometrics"
WRITE_STAGE = "08_pipe4_statistical_validation"
PIPE3_ROOT = DATA_ROOT / TARGET_CORPUS / "04_rebuild_outputs" / READ_STAGE
WRITE_ROOT = DATA_ROOT / TARGET_CORPUS / "04_rebuild_outputs" / WRITE_STAGE
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
BOOTSTRAP_ROUNDS = 2000
MANUAL_AUDIT_N = 24

assert PIPE3_ROOT.exists(), f"PIPE3_ROOT nao encontrado: {PIPE3_ROOT}"
print("PIPE3_ROOT =", PIPE3_ROOT)
print("WRITE_ROOT =", WRITE_ROOT)
'''


PIPE4_LOGGING = r'''
# ============================================================
# Saidas, logs e helpers
# ============================================================

PIPE_START_TS = time.time()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


ensure_dir(WRITE_ROOT)
ensure_dir(WRITE_ROOT / "tables")
ensure_dir(WRITE_ROOT / "reports")
GLOBAL_LOG_DIR = ensure_dir(PROJECT_ROOT / "outputs" / "camada4_logs")
GLOBAL_LOG_FILE = GLOBAL_LOG_DIR / f"08_pipe4_{RUN_TS}.txt"


def elapsed_seconds() -> float:
    return time.time() - PIPE_START_TS


def fmt_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{now} | +{fmt_seconds(elapsed_seconds())}]"
    line = f"{prefix} {message}"
    print(line, flush=True)
    GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def stage_banner(title: str) -> None:
    bar = "=" * 96
    log(bar)
    log(title)
    log(bar)
'''


PIPE4_LOAD = r'''
# ============================================================
# Leitura dos outputs do Pipe 3
# ============================================================

stage_banner("LEITURA DO PIPE 3")

metrics_df = pd.read_csv(PIPE3_ROOT / "tables" / "retriever_query_metrics.csv")
agent_df = pd.read_csv(PIPE3_ROOT / "tables" / "agent_outputs.csv")
hits_df = pd.read_csv(PIPE3_ROOT / "tables" / "retrieval_hits_all.csv")

log(f"[load] metrics={len(metrics_df)} | agent_outputs={len(agent_df)} | hits={len(hits_df)}")
display(metrics_df.head(10))
display(agent_df.head(10))
'''


PIPE4_STATS = r'''
# ============================================================
# Estatisticas e intervalos de confianca
# ============================================================

stage_banner("ESTATISTICAS")


def wilson_interval(successes: int, total: int, z: float = 1.96):
    if total == 0:
        return (np.nan, np.nan, np.nan)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2)))
    return (p, max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_mean_ci(values: np.ndarray, rounds: int = BOOTSTRAP_ROUNDS, seed: int = 42):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(rounds, len(values)), replace=True)
    means = samples.mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


summary_rows = []
for (period, retriever), sub in metrics_df.groupby(["period", "retriever"], sort=False):
    mean_top1, top1_lo, top1_hi = bootstrap_mean_ci(sub["top1_score"].to_numpy())
    mean_cit, cit_lo, cit_hi = bootstrap_mean_ci(sub["mean_topk_citations"].to_numpy())
    mean_recent, recent_lo, recent_hi = bootstrap_mean_ci(sub["recent_share_topk"].to_numpy())
    success_rate, succ_lo, succ_hi = wilson_interval(
        int(sub["distinct_sources_topk"].fillna(0).ge(3).sum()),
        int(len(sub)),
    )
    summary_rows.append(
        {
            "period": period,
            "retriever": retriever,
            "n_queries": int(len(sub)),
            "mean_top1_score": round(mean_top1, 4),
            "mean_top1_score_ci_low": round(top1_lo, 4),
            "mean_top1_score_ci_high": round(top1_hi, 4),
            "mean_topk_citations": round(mean_cit, 4),
            "mean_topk_citations_ci_low": round(cit_lo, 4),
            "mean_topk_citations_ci_high": round(cit_hi, 4),
            "mean_recent_share": round(mean_recent, 4),
            "mean_recent_share_ci_low": round(recent_lo, 4),
            "mean_recent_share_ci_high": round(recent_hi, 4),
            "share_queries_ge_3_sources": round(success_rate, 4),
            "share_queries_ge_3_sources_ci_low": round(succ_lo, 4),
            "share_queries_ge_3_sources_ci_high": round(succ_hi, 4),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(WRITE_ROOT / "tables" / "statistical_summary.csv", index=False)
display(summary_df)
'''


PIPE4_DELTAS = r'''
# ============================================================
# Deltas principais: especializado vs generico e core vs holdout
# ============================================================

stage_banner("DELTAS PRINCIPAIS")

pivot_top1 = metrics_df.pivot_table(
    index=["period", "query_id", "query_text", "source_corpus"],
    columns="retriever",
    values="top1_score",
    aggfunc="first",
).reset_index()

if "specialized" in pivot_top1.columns and "generic" in pivot_top1.columns:
    pivot_top1["delta_specialized_vs_generic"] = pivot_top1["specialized"] - pivot_top1["generic"]
else:
    pivot_top1["delta_specialized_vs_generic"] = np.nan

delta_rows = []
for period, sub in pivot_top1.groupby("period", sort=False):
    delta_mean, delta_lo, delta_hi = bootstrap_mean_ci(sub["delta_specialized_vs_generic"].to_numpy())
    delta_rows.append(
        {
            "period": period,
            "metric": "delta_specialized_vs_generic_top1",
            "mean": round(delta_mean, 4),
            "ci_low": round(delta_lo, 4),
            "ci_high": round(delta_hi, 4),
            "n_queries": int(len(sub)),
        }
    )

core_holdout_merge = summary_df.pivot(index="retriever", columns="period", values="mean_top1_score").reset_index()
if {"core", "holdout"}.issubset(core_holdout_merge.columns):
    core_holdout_merge["holdout_minus_core_mean_top1"] = core_holdout_merge["holdout"] - core_holdout_merge["core"]

delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(WRITE_ROOT / "tables" / "paired_deltas.csv", index=False)
core_holdout_merge.to_csv(WRITE_ROOT / "tables" / "core_holdout_mean_top1_delta.csv", index=False)

display(delta_df)
display(core_holdout_merge)
'''


PIPE4_AUDIT = r'''
# ============================================================
# Amostra para auditoria manual pequena
# ============================================================

stage_banner("AUDITORIA MANUAL PEQUENA")

top_hits = hits_df[hits_df["rank"] == 1].copy()
top_hits["audit_bucket"] = top_hits["period"] + "__" + top_hits["retriever"]

audit_rows = []
for bucket, sub in top_hits.groupby("audit_bucket", sort=False):
    take_n = min(max(1, MANUAL_AUDIT_N // max(1, top_hits["audit_bucket"].nunique())), len(sub))
    audit_rows.append(sub.sample(take_n, random_state=42))

audit_df = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()
audit_df.to_csv(WRITE_ROOT / "tables" / "manual_audit_sample.csv", index=False)

with pd.ExcelWriter(WRITE_ROOT / "reports" / "paper_tables_final.xlsx", engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="statistical_summary", index=False)
    delta_df.to_excel(writer, sheet_name="paired_deltas", index=False)
    core_holdout_merge.to_excel(writer, sheet_name="core_holdout_top1", index=False)
    audit_df.to_excel(writer, sheet_name="manual_audit_sample", index=False)

display(audit_df.head(20))
'''


PIPE4_MANIFEST = r'''
# ============================================================
# Manifesto final da validacao
# ============================================================

stage_banner("MANIFESTO FINAL")

manifest_rows = []
for path in sorted(WRITE_ROOT.rglob("*")):
    if path.is_file():
        manifest_rows.append({"artifact": str(path), "size_bytes": path.stat().st_size})

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(WRITE_ROOT / "reports" / "artifact_manifest.csv", index=False)

with open(WRITE_ROOT / "reports" / "validation_manifest.json", "w", encoding="utf-8") as fh:
    json.dump(
        {
            "read_stage": READ_STAGE,
            "write_stage": WRITE_STAGE,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            "manual_audit_n": MANUAL_AUDIT_N,
            "run_ts": RUN_TS,
        },
        fh,
        indent=2,
        ensure_ascii=False,
    )

display(manifest_df.head(30))
print("Arquivos finais salvos em:", WRITE_ROOT)
'''


PIPE9_MD = """
# 09 Pipe 9 - methodological storyline pack

Notebook canonico de sintese para:

1. reunir os artefatos historicos do paper original;
2. reunir os outputs canonicos do rebuild `00-08`;
3. construir um crosswalk velho->novo por camada;
4. consolidar comparacoes quantitativas e evolucoes metodologicas;
5. exportar um pacote pronto para a reescrita do manuscrito do `major review`.

## Papel metodologico

Este notebook **nao** cria uma nova camada experimental. Ele organiza a evidencia necessaria para:

- contar a historia real do pipeline original;
- mostrar o que foi preservado no regime `core`;
- mostrar o que foi acrescentado para o `major review`;
- sustentar a comparacao entre paper original e rebuild canonico;
- apoiar a reescrita do artigo e da carta-resposta.
"""


PIPE9_INSTALL = r'''
# ============================================================
# Instalacao de dependencias para Colab
# ============================================================
!pip install -U -q pip setuptools wheel
!pip uninstall -y -q numpy scipy scikit-learn sentence-transformers transformers faiss-cpu numba umap-learn hdbscan || true
!pip install -U -q numpy==2.0.2 scipy==1.14.1 pandas==2.2.2 openpyxl pyarrow jedi==0.19.2

import numpy, scipy, pandas
print("numpy            :", numpy.__version__)
print("scipy            :", scipy.__version__)
print("pandas           :", pandas.__version__)

print("Dependencias instaladas.")
'''


PIPE9_IMPORTS = r'''
from google.colab import drive
drive.mount("/content/drive")

import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 240)
pd.set_option("display.max_colwidth", 280)
'''


PIPE9_CONFIG = r'''
# ============================================================
# Configuracao geral
# ============================================================

DRIVE_ROOT = Path("/content/drive/MyDrive/Unicamp")
PROJECT_ROOT = DRIVE_ROOT / "artigo bibliometria" / "grounded-scientometrics-solarphysics-retrieval"
DATA_ROOT = DRIVE_ROOT / "artigo bibliometria" / "base de dados" / "Artigo_Bibliometria Base Bruta" / "BASES_UNIFICADAS_POR_TEMA"
HISTORICAL_ROOT = DRIVE_ROOT / "artigo bibliometria" / "base de dados" / "artefatos artigo resvisado"
HISTORICAL_FINAL_ROOT = DRIVE_ROOT / "artigo bibliometria" / "artefatatos finais"
OVERLEAF_ROOT = DRIVE_ROOT / "artigo bibliometria" / "overleaf-sync"

WRITE_ROOT = DATA_ROOT / "_cross_corpus_rebuild" / "09_methodological_storyline_pack"
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

CORPORA = ["Nucleo", "PIML", "CombFinal", "ML_Multimodal"]
TRAIN_CORPORA = ["Nucleo", "PIML", "CombFinal"]
PERIODS = ["core", "holdout"]

CURRENT_05_ROOT = DATA_ROOT / "_cross_corpus_rebuild" / "05_scibert_solarphysics_search"
CURRENT_06_ROOT = DATA_ROOT / "ML_Multimodal" / "04_rebuild_outputs" / "06_pipe2_retriever_analytics"
CURRENT_07_ROOT = DATA_ROOT / "ML_Multimodal" / "04_rebuild_outputs" / "07_pipe3_agent_scientometrics"
CURRENT_08_ROOT = DATA_ROOT / "ML_Multimodal" / "04_rebuild_outputs" / "08_pipe4_statistical_validation"

HIST_ABSTRACT_NOTEBOOKS = {
    "Nucleo": HISTORICAL_ROOT / "Abstract_LLM_gpu_Nucleo.ipynb",
    "PIML": HISTORICAL_ROOT / "Abstract_LLM_gpu_piml.ipynb",
    "CombFinal": HISTORICAL_ROOT / "Abstract_LLM_gpu_CombFinal.ipynb",
    "ML_Multimodal": HISTORICAL_ROOT / "Abstract_LLM_gpu_ml.ipynb",
}
HIST_SCIBERT_NOTEBOOK = HISTORICAL_ROOT / "SciBERT_SolarPhysics_Search.ipynb"
HIST_PIPE2_ROOT = HISTORICAL_ROOT / "pipe 2"
HIST_PIPE2_NOTEBOOK = HISTORICAL_ROOT / "Pipeline_Analytcs_(SciBERT_SolarPhysics_Search) (3).ipynb"
HIST_PIPE3_NOTEBOOK = HISTORICAL_FINAL_ROOT / "Piepe_3_Scientometrics.ipynb"
HIST_PIPE4_NOTEBOOK = HISTORICAL_FINAL_ROOT / "PIPE_4_Validacao_Estatistica.ipynb"
MANUSCRIPT_TEX = OVERLEAF_ROOT / "sn-article.tex"
PLAN_MD = HISTORICAL_ROOT / "PLANO_OPERACIONAL_MAJOR_REVIEW_REBUILD.md"

assert PROJECT_ROOT.exists(), f"PROJECT_ROOT nao encontrado: {PROJECT_ROOT}"
assert DATA_ROOT.exists(), f"DATA_ROOT nao encontrado: {DATA_ROOT}"
assert HISTORICAL_ROOT.exists(), f"HISTORICAL_ROOT nao encontrado: {HISTORICAL_ROOT}"
assert HISTORICAL_FINAL_ROOT.exists(), f"HISTORICAL_FINAL_ROOT nao encontrado: {HISTORICAL_FINAL_ROOT}"
assert MANUSCRIPT_TEX.exists(), f"MANUSCRIPT_TEX nao encontrado: {MANUSCRIPT_TEX}"
assert PLAN_MD.exists(), f"PLAN_MD nao encontrado: {PLAN_MD}"

print("WRITE_ROOT =", WRITE_ROOT)
print("HISTORICAL_ROOT =", HISTORICAL_ROOT)
print("HISTORICAL_FINAL_ROOT =", HISTORICAL_FINAL_ROOT)
print("PLAN_MD =", PLAN_MD)
'''


PIPE9_LOGGING = r'''
# ============================================================
# Helpers e logging
# ============================================================

PIPE_START_TS = time.time()
TABLES_DIR = WRITE_ROOT / "tables"
REPORTS_DIR = WRITE_ROOT / "reports"
ARTIFACTS_DIR = WRITE_ROOT / "artifacts"
for path in [WRITE_ROOT, TABLES_DIR, REPORTS_DIR, ARTIFACTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

GLOBAL_LOG_FILE = PROJECT_ROOT / "outputs" / "camada5_logs" / f"09_storyline_{RUN_TS}.txt"
GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    elapsed = int(time.time() - PIPE_START_TS)
    hh = elapsed // 3600
    mm = (elapsed % 3600) // 60
    ss = elapsed % 60
    prefix = f"[{datetime.now().strftime('%H:%M:%S')} | +{hh:02d}:{mm:02d}:{ss:02d}]"
    line = f"{prefix} {message}"
    print(line, flush=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def stage_banner(title: str) -> None:
    bar = "=" * 96
    log(bar)
    log(title)
    log(bar)


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_csv_if_exists(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def safe_pct_non_empty(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return round(float(series.fillna("").astype(str).str.len().gt(0).mean() * 100), 2)


def normalize_number(value):
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return value


def comparison_status(historical_value, rebuild_value) -> str:
    try:
        h = float(historical_value)
        r = float(rebuild_value)
    except Exception:
        return "non_numeric"
    if not np.isfinite(h) or not np.isfinite(r):
        return "missing"
    if abs(h - r) < 1e-9:
        return "exact_match"
    rel = abs(r - h) / max(abs(h), 1e-9)
    if rel <= 0.05:
        return "near_match"
    if rel <= 0.20:
        return "moderate_shift"
    return "major_shift"


def fmt_num(value, digits: int = 4) -> str:
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(x):
        return "n/a"
    if abs(x) >= 100:
        return f"{x:,.1f}"
    if abs(x) >= 10:
        return f"{x:,.2f}"
    return f"{x:.{digits}f}"
'''


PIPE9_HISTORICAL = r'''
# ============================================================
# Referencias historicas auditadas + crosswalk
# ============================================================

stage_banner("REFERENCIAS HISTORICAS E CROSSWALK")

historical_rows = []


def add_hist(layer: str, metric_group: str, metric_key: str, value, source_path: Path, note: str) -> None:
    historical_rows.append(
        {
            "layer": layer,
            "metric_group": metric_group,
            "metric_key": metric_key,
            "value": value,
            "source_path": str(source_path),
            "note": note,
        }
    )


# Camada 0 / 1 - referencias auditadas durante o rebuild
add_hist("00", "Nucleo", "Nucleo_total_docs", 7349, MANUSCRIPT_TEX, "total historico do corpus Nucleo reportado no manuscrito original")
add_hist("00", "PIML", "PIML_total_docs", 12866, MANUSCRIPT_TEX, "total historico do corpus PIML reportado no manuscrito original")
add_hist("00", "CombFinal", "CombFinal_total_docs", 982, MANUSCRIPT_TEX, "total historico do corpus CombFinal reportado no manuscrito original")

add_hist("01", "Nucleo", "Nucleo_core_rows_used", 5036, HIST_ABSTRACT_NOTEBOOKS["Nucleo"], "abstracts limpos usados no Abstract_LLM historico")
add_hist("01", "Nucleo", "Nucleo_core_k", 30, HIST_ABSTRACT_NOTEBOOKS["Nucleo"], "numero de clusters historico do Nucleo")
add_hist("01", "PIML", "PIML_core_rows_used", 8955, HIST_ABSTRACT_NOTEBOOKS["PIML"], "abstracts limpos usados no Abstract_LLM historico")
add_hist("01", "PIML", "PIML_core_k", 30, HIST_ABSTRACT_NOTEBOOKS["PIML"], "numero de clusters historico do PIML")
add_hist("01", "CombFinal", "CombFinal_core_rows_used", 780, HIST_ABSTRACT_NOTEBOOKS["CombFinal"], "abstracts limpos usados no Abstract_LLM historico")
add_hist("01", "CombFinal", "CombFinal_core_k", 30, HIST_ABSTRACT_NOTEBOOKS["CombFinal"], "numero de clusters historico do CombFinal")
add_hist("01", "ML_Multimodal", "ML_Multimodal_core_rows_used", 83771, HIST_ABSTRACT_NOTEBOOKS["ML_Multimodal"], "abstracts limpos usados no Abstract_LLM historico do ML")
add_hist("01", "ML_Multimodal", "ML_Multimodal_core_k", 30, HIST_ABSTRACT_NOTEBOOKS["ML_Multimodal"], "numero de clusters historico do ML")

add_hist("05", "SciBERT", "scibert_dapt_docs", 21197, HIST_SCIBERT_NOTEBOOK, "volume historico do DAPT")
add_hist("05", "SciBERT", "scibert_dapt_perplexity", 3.12, HIST_SCIBERT_NOTEBOOK, "perplexity historica auditada do DAPT")
add_hist("05", "SciBERT", "scibert_contrastive_pairs", 36424, HIST_SCIBERT_NOTEBOOK, "pares contrastivos historicos")
add_hist("05", "SciBERT", "scibert_ab_docs", 4555, HIST_SCIBERT_NOTEBOOK, "tamanho da amostra A/B historica")
add_hist("05", "SciBERT", "scibert_ab_classes", 15, HIST_SCIBERT_NOTEBOOK, "numero de classes de tecnica na A/B historica")
add_hist("05", "SciBERT", "scibert_mrr_baseline", 0.668291, HIST_SCIBERT_NOTEBOOK, "MRR historico do baseline")
add_hist("05", "SciBERT", "scibert_mrr_specialized", 0.703821, HIST_SCIBERT_NOTEBOOK, "MRR historico do modelo especializado")
add_hist("05", "SciBERT", "scibert_recall10_baseline", 0.880132, HIST_SCIBERT_NOTEBOOK, "Recall@10 historico do baseline")
add_hist("05", "SciBERT", "scibert_recall10_specialized", 0.899671, HIST_SCIBERT_NOTEBOOK, "Recall@10 historico do especializado")
add_hist("05", "SciBERT", "scibert_recall50_baseline", 0.979802, HIST_SCIBERT_NOTEBOOK, "Recall@50 historico do baseline")
add_hist("05", "SciBERT", "scibert_recall50_specialized", 0.986828, HIST_SCIBERT_NOTEBOOK, "Recall@50 historico do especializado")
add_hist("05", "SciBERT", "scibert_recall100_baseline", 0.991877, HIST_SCIBERT_NOTEBOOK, "Recall@100 historico do baseline")
add_hist("05", "SciBERT", "scibert_recall100_specialized", 0.995170, HIST_SCIBERT_NOTEBOOK, "Recall@100 historico do especializado")
add_hist("05", "SciBERT", "scibert_nmi_baseline", 0.093411, HIST_SCIBERT_NOTEBOOK, "NMI historico do baseline")
add_hist("05", "SciBERT", "scibert_nmi_specialized", 0.108490, HIST_SCIBERT_NOTEBOOK, "NMI historico do especializado")
add_hist("05", "SciBERT", "scibert_ari_baseline", 0.027942, HIST_SCIBERT_NOTEBOOK, "ARI historico do baseline")
add_hist("05", "SciBERT", "scibert_ari_specialized", 0.036066, HIST_SCIBERT_NOTEBOOK, "ARI historico do especializado")
add_hist("05", "SciBERT", "scibert_silhouette_baseline", 0.120245, HIST_SCIBERT_NOTEBOOK, "Silhouette historico do baseline")
add_hist("05", "SciBERT", "scibert_silhouette_specialized", 0.140223, HIST_SCIBERT_NOTEBOOK, "Silhouette historico do especializado")
add_hist("05", "SciBERT", "scibert_nearest_centroid_baseline", 0.315000, HIST_SCIBERT_NOTEBOOK, "Nearest centroid accuracy historico do baseline")
add_hist("05", "SciBERT", "scibert_nearest_centroid_specialized", 0.441000, HIST_SCIBERT_NOTEBOOK, "Nearest centroid accuracy historico do especializado")


# Camada 2 - historico estruturado em CSV
old_pipe2_docs = read_csv_if_exists(HIST_PIPE2_ROOT / "docs_master.csv")
old_pipe2_index = read_csv_if_exists(HIST_PIPE2_ROOT / "index_map.csv")
if old_pipe2_docs is not None:
    add_hist("06", "Pipe2", "pipe2_application_docs_rows", int(len(old_pipe2_docs)), HIST_PIPE2_ROOT / "docs_master.csv", "docs_master historico de aplicacao")
if old_pipe2_index is not None:
    add_hist("06", "Pipe2", "pipe2_index_map_rows", int(len(old_pipe2_index)), HIST_PIPE2_ROOT / "index_map.csv", "index_map historico de aplicacao")


# Camada 3 - historico estruturado em CSV
old_pipe3_agent = read_csv_if_exists(HISTORICAL_FINAL_ROOT / "pipe3_agent_metrics.csv")
if old_pipe3_agent is not None:
    for row in old_pipe3_agent.itertuples(index=False):
        retr_label = "generic" if row.retriever == "R0" else "specialized"
        mode_label = str(row.mode)
        add_hist("07", "Pipe3", f"pipe3_{retr_label}_{mode_label}_gaps", int(row.gaps), HISTORICAL_FINAL_ROOT / "pipe3_agent_metrics.csv", "numero de gaps no package final historico")
        add_hist("07", "Pipe3", f"pipe3_{retr_label}_{mode_label}_domain_hit_abstract_retrieved_topk", float(row.domain_hit_rate_abstract_retrieved_topk), HISTORICAL_FINAL_ROOT / "pipe3_agent_metrics.csv", "domain hit rate historico no top-k reconstruido")
        if not pd.isna(row.domain_hit_rate_abstract_cited):
            add_hist("07", "Pipe3", f"pipe3_{retr_label}_{mode_label}_domain_hit_abstract_cited", float(row.domain_hit_rate_abstract_cited), HISTORICAL_FINAL_ROOT / "pipe3_agent_metrics.csv", "domain hit rate historico nas evidencias citadas")

old_pipe3_h1 = read_csv_if_exists(HISTORICAL_FINAL_ROOT / "pipe3_H1_deltas_v2.csv")
if old_pipe3_h1 is not None:
    delta_col = "delta_domain_hit_rate_abstract_retrieved_topk (R1-R0)"
    if delta_col in old_pipe3_h1.columns:
        for _, row in old_pipe3_h1.iterrows():
            add_hist("07", "Pipe3", f"pipe3_h1_{row['mode']}_delta_domain_hit_retrieved", float(row[delta_col]), HISTORICAL_FINAL_ROOT / "pipe3_H1_deltas_v2.csv", "delta historico R1-R0 no top-k")


# Camada 4 - historico sem pacote tabular estruturado equivalente
add_hist("08", "Pipe4", "pipe4_historical_notebook_present", 1, HIST_PIPE4_NOTEBOOK, "o notebook historico de validacao estatistica existe, mas nao ha pacote tabular equivalente ao rebuild")

historical_df = pd.DataFrame(historical_rows)
historical_df.to_csv(TABLES_DIR / "historical_reference_metrics.csv", index=False)

layer_crosswalk = pd.DataFrame(
    [
        {
            "historical_step": "R consolidation scripts",
            "historical_artifact": "consolida_base_bruta_*.R / *.txt",
            "historical_role": "engenharia das bases",
            "rebuild_pipe": "00_consolidacao",
            "rebuild_role": "corpora canonicos core/holdout",
            "major_review_increment": "split temporal core/holdout com trilha auditavel",
        },
        {
            "historical_step": "Abstract_LLM_gpu_*",
            "historical_artifact": "Abstract_LLM_gpu_*.ipynb",
            "historical_role": "estrutura semantica upstream",
            "rebuild_pipe": "01-04_abstract_llm_*",
            "rebuild_role": "topicos core/holdout por corpus",
            "major_review_increment": "run_metadata, manifests, holdout espelhado",
        },
        {
            "historical_step": "SciBERT_SolarPhysics_Search",
            "historical_artifact": "SciBERT_SolarPhysics_Search.ipynb",
            "historical_role": "treino do retriever especializado",
            "rebuild_pipe": "05_scibert_solarphysics_search",
            "rebuild_role": "DAPT + contrastive + A/B paper-facing",
            "major_review_increment": "HF publish, labels audit, reproducibilidade core-only",
        },
        {
            "historical_step": "Pipeline_Analytcs",
            "historical_artifact": "Pipeline_Analytcs_(SciBERT_SolarPhysics_Search)",
            "historical_role": "ponte entre retriever e corpus de aplicacao",
            "rebuild_pipe": "06_pipe2_retriever_analytics",
            "rebuild_role": "docs_master/index_map e FAISS por periodo",
            "major_review_increment": "indices generic/specialized em core e holdout",
        },
        {
            "historical_step": "Piepe_3_Scientometrics",
            "historical_artifact": "Piepe_3_Scientometrics.ipynb + pipe3_*.csv",
            "historical_role": "experimento final do paper",
            "rebuild_pipe": "07_pipe3_agent_scientometrics",
            "rebuild_role": "bundles grounded + BM25 + tabelas paper-ready",
            "major_review_increment": "baseline lexical e outputs separados por periodo",
        },
        {
            "historical_step": "PIPE_4_Validacao_Estatistica",
            "historical_artifact": "PIPE_4_Validacao_Estatistica.ipynb",
            "historical_role": "validacao final",
            "rebuild_pipe": "08_pipe4_statistical_validation",
            "rebuild_role": "bootstrap CI, deltas, auditoria manual",
            "major_review_increment": "camada estatistica mais rastreavel e temporal",
        },
        {
            "historical_step": "Narrativa dispersa",
            "historical_artifact": "manuscrito + notebooks + exports",
            "historical_role": "historia metodologica implicita",
            "rebuild_pipe": "09_methodological_storyline_pack",
            "rebuild_role": "crosswalk + comparacoes + storyline",
            "major_review_increment": "pacote explicito para reescrita e resposta ao major review",
        },
    ]
)
layer_crosswalk.to_csv(TABLES_DIR / "layer_crosswalk.csv", index=False)

evolution_df = pd.DataFrame(
    [
        {
            "dimension": "temporal_split",
            "historical_state": "corpus unico sem split explicito por periodo",
            "major_review_state": "regimes core e holdout espelhados",
            "manuscript_use": "separar replicacao principal de validacao temporal",
        },
        {
            "dimension": "retriever_baselines",
            "historical_state": "SciBERT generico vs especializado no Phase I",
            "major_review_state": "SciBERT generico + especializado + BM25 no downstream",
            "manuscript_use": "justificar baseline lexical sem mudar a tese central",
        },
        {
            "dimension": "traceability",
            "historical_state": "artefatos distribuidos e parcialmente implicitos",
            "major_review_state": "logs, manifests, docs_master, index_map e pacotes por camada",
            "manuscript_use": "reforcar auditabilidade e reprodutibilidade",
        },
        {
            "dimension": "publication",
            "historical_state": "modelo especializado treinado no Colab historico",
            "major_review_state": "modelo republlicado no Hugging Face",
            "manuscript_use": "mostrar disponibilizacao do retriever especializado",
        },
        {
            "dimension": "statistical_validation",
            "historical_state": "validacao menos estruturada",
            "major_review_state": "bootstrap CI, deltas por periodo e amostra de auditoria manual",
            "manuscript_use": "apoiar a secao de robustez do major review",
        },
    ]
)
evolution_df.to_csv(TABLES_DIR / "major_review_evolution_points.csv", index=False)

display(historical_df.head(40))
display(layer_crosswalk)
display(evolution_df)
'''


PIPE9_REBUILD = r'''
# ============================================================
# Snapshot do rebuild canonico 00-08
# ============================================================

stage_banner("SNAPSHOT DO REBUILD 00-08")

rebuild_rows = []


def add_rebuild(layer: str, metric_group: str, metric_key: str, value, source_path: Path, note: str) -> None:
    rebuild_rows.append(
        {
            "layer": layer,
            "metric_group": metric_group,
            "metric_key": metric_key,
            "value": value,
            "source_path": str(source_path),
            "note": note,
        }
    )


# 00 - consolidacao
for corpus in CORPORA:
    total_rows = 0
    for period in PERIODS:
        csv_path = DATA_ROOT / corpus / "04_rebuild_outputs" / "00_consolidacao" / f"{corpus}_{period}_bibliometrix_clean.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        total_rows += int(len(df))
        add_rebuild("00", corpus, f"{corpus}_{period}_rows", int(len(df)), csv_path, "rows no consolidado canonico")
        if "AB" in df.columns:
            add_rebuild("00", corpus, f"{corpus}_{period}_abstract_pct", safe_pct_non_empty(df["AB"]), csv_path, "percentual com abstract no consolidado")
    if total_rows > 0:
        add_rebuild("00", corpus, f"{corpus}_total_docs", total_rows, DATA_ROOT / corpus / "04_rebuild_outputs" / "00_consolidacao", "soma core+holdout do consolidado canonico")
        log(f"[00] {corpus} | total_docs={total_rows}")


# 01-04 - abstract llm
for corpus in CORPORA:
    for period in PERIODS:
        period_root = DATA_ROOT / corpus / "04_rebuild_outputs" / "01_abstract_llm" / period
        meta_path = period_root / f"{corpus}_{period}_run_metadata.json"
        meta = read_json_if_exists(meta_path)
        rows_used = None
        chosen_k = None
        if meta:
            rows_used = meta.get("rows_used")
            chosen_k = meta.get("chosen_n_clusters")
        else:
            doc_assign_path = period_root / "tables" / f"{corpus}_{period}_doc_topic_assignment.csv"
            tuning_path = period_root / "tables" / f"{corpus}_{period}_topic_model_tuning.csv"
            doc_assign_df = read_csv_if_exists(doc_assign_path)
            tuning_df = read_csv_if_exists(tuning_path)
            if doc_assign_df is not None:
                rows_used = int(len(doc_assign_df))
            if tuning_df is not None and len(tuning_df):
                chosen_k = int(tuning_df.sort_values(["score", "n_clusters"], ascending=[False, True]).iloc[0]["n_clusters"])
        if rows_used is not None:
            add_rebuild("01", corpus, f"{corpus}_{period}_rows_used", int(rows_used), meta_path if meta_path.exists() else period_root, "rows efetivamente usados no Abstract_LLM canonico")
        if chosen_k is not None:
            add_rebuild("01", corpus, f"{corpus}_{period}_k", int(chosen_k), meta_path if meta_path.exists() else period_root, "k final do Abstract_LLM canonico")
    log(f"[01] {corpus} | snapshot coletado")


# 05 - SciBERT / Phase I
train_summary = read_csv_if_exists(CURRENT_05_ROOT / "artifacts" / "training_input_summary.csv")
if train_summary is not None and len(train_summary):
    add_rebuild("05", "SciBERT", "scibert_dapt_docs", int(train_summary.iloc[0]["n_dapt_texts"]), CURRENT_05_ROOT / "artifacts" / "training_input_summary.csv", "numero de textos usados no DAPT")
    add_rebuild("05", "SciBERT", "scibert_contrastive_pairs", int(train_summary.iloc[0]["n_pairs"]), CURRENT_05_ROOT / "artifacts" / "training_input_summary.csv", "numero de pares contrastivos")

dapt_metrics = read_json_if_exists(CURRENT_05_ROOT / "reports" / "dapt_eval_metrics.json")
if dapt_metrics:
    add_rebuild("05", "SciBERT", "scibert_dapt_perplexity", dapt_metrics.get("perplexity"), CURRENT_05_ROOT / "reports" / "dapt_eval_metrics.json", "perplexity do DAPT canonico")
    add_rebuild("05", "SciBERT", "scibert_dapt_eval_loss", dapt_metrics.get("eval_loss"), CURRENT_05_ROOT / "reports" / "dapt_eval_metrics.json", "eval loss do DAPT canonico")

labels_cov = read_csv_if_exists(CURRENT_05_ROOT / "artifacts" / "labels_coverage.csv")
if labels_cov is not None and len(labels_cov):
    tecnica_row = labels_cov.loc[labels_cov["axis"] == "Tecnica"]
    if len(tecnica_row):
        add_rebuild("05", "SciBERT", "scibert_tecnica_coverage_pct", float(tecnica_row.iloc[0]["coverage_pct"]), CURRENT_05_ROOT / "artifacts" / "labels_coverage.csv", "cobertura da rotulacao fraca na tecnica")

ab_summary = read_json_if_exists(CURRENT_05_ROOT / "reports" / "historical_ab_eval_summary.json")
if ab_summary:
    add_rebuild("05", "SciBERT", "scibert_ab_docs", ab_summary.get("eval_docs"), CURRENT_05_ROOT / "reports" / "historical_ab_eval_summary.json", "docs na avaliacao A/B paper-facing")
    add_rebuild("05", "SciBERT", "scibert_ab_classes", ab_summary.get("eval_classes"), CURRENT_05_ROOT / "reports" / "historical_ab_eval_summary.json", "classes na avaliacao A/B paper-facing")

ret_df = read_csv_if_exists(CURRENT_05_ROOT / "reports" / "eval_retrieval_tecnica.csv")
if ret_df is not None:
    model_map = {
        "SciBERT-baseline": "baseline",
        "SciBERT-SolarPhysics-Search": "specialized",
    }
    for _, row in ret_df.iterrows():
        label = model_map.get(row["model"])
        if not label:
            continue
        add_rebuild("05", "SciBERT", f"scibert_mrr_{label}", float(row["MRR"]), CURRENT_05_ROOT / "reports" / "eval_retrieval_tecnica.csv", "MRR do Phase I canonico")
        add_rebuild("05", "SciBERT", f"scibert_recall10_{label}", float(row["Recall@10"]), CURRENT_05_ROOT / "reports" / "eval_retrieval_tecnica.csv", "Recall@10 do Phase I canonico")
        add_rebuild("05", "SciBERT", f"scibert_recall50_{label}", float(row["Recall@50"]), CURRENT_05_ROOT / "reports" / "eval_retrieval_tecnica.csv", "Recall@50 do Phase I canonico")
        add_rebuild("05", "SciBERT", f"scibert_recall100_{label}", float(row["Recall@100"]), CURRENT_05_ROOT / "reports" / "eval_retrieval_tecnica.csv", "Recall@100 do Phase I canonico")

clu_df = read_csv_if_exists(CURRENT_05_ROOT / "reports" / "eval_clustering_tecnica.csv")
if clu_df is not None:
    model_map = {
        "SciBERT-baseline": "baseline",
        "SciBERT-SolarPhysics-Search": "specialized",
    }
    for _, row in clu_df.iterrows():
        label = model_map.get(row["model"])
        if not label:
            continue
        add_rebuild("05", "SciBERT", f"scibert_nmi_{label}", float(row["NMI"]), CURRENT_05_ROOT / "reports" / "eval_clustering_tecnica.csv", "NMI do Phase I canonico")
        add_rebuild("05", "SciBERT", f"scibert_ari_{label}", float(row["ARI"]), CURRENT_05_ROOT / "reports" / "eval_clustering_tecnica.csv", "ARI do Phase I canonico")
        add_rebuild("05", "SciBERT", f"scibert_silhouette_{label}", float(row["Silhouette"]), CURRENT_05_ROOT / "reports" / "eval_clustering_tecnica.csv", "Silhouette do Phase I canonico")

nc_df = read_csv_if_exists(CURRENT_05_ROOT / "reports" / "eval_nearest_centroid.csv")
if nc_df is not None:
    model_map = {
        "SciBERT-baseline": "baseline",
        "SciBERT-SolarPhysics-Search": "specialized",
    }
    for _, row in nc_df.iterrows():
        label = model_map.get(row["model"])
        if not label:
            continue
        add_rebuild("05", "SciBERT", f"scibert_nearest_centroid_{label}", float(row["NearestCentroidAcc"]), CURRENT_05_ROOT / "reports" / "eval_nearest_centroid.csv", "nearest centroid accuracy do Phase I canonico")

hf_report = read_json_if_exists(CURRENT_05_ROOT / "reports" / "hf_publish_report.json")
if hf_report:
    add_rebuild("05", "SciBERT", "scibert_hf_publish_success", int(bool(hf_report.get("status") == "published" or hf_report.get("published") or hf_report.get("push_commit"))), CURRENT_05_ROOT / "reports" / "hf_publish_report.json", "indicador de publicacao no HF")

log("[05] snapshot coletado")


# 06 - Pipe 2
pipe2_cov = read_csv_if_exists(CURRENT_06_ROOT / "tables" / "ml_docs_master_coverage.csv")
if pipe2_cov is not None:
    for row in pipe2_cov.itertuples(index=False):
        add_rebuild("06", "Pipe2", f"pipe2_{row.period}_docs_rows", int(row.rows), CURRENT_06_ROOT / "tables" / "ml_docs_master_coverage.csv", "docs_master por periodo")
        if row.period == "core":
            add_rebuild("06", "Pipe2", "pipe2_application_docs_rows", int(row.rows), CURRENT_06_ROOT / "tables" / "ml_docs_master_coverage.csv", "docs_master do regime de replicacao principal")

core_index = read_csv_if_exists(CURRENT_06_ROOT / "core" / "tables" / "ML_Multimodal_core_index_map.csv")
if core_index is not None:
    add_rebuild("06", "Pipe2", "pipe2_index_map_rows", int(len(core_index)), CURRENT_06_ROOT / "core" / "tables" / "ML_Multimodal_core_index_map.csv", "index_map do regime de replicacao principal")

query_bank = read_csv_if_exists(CURRENT_06_ROOT / "tables" / "core_gap_query_bank.csv")
if query_bank is not None:
    add_rebuild("06", "Pipe2", "pipe2_query_bank_size", int(len(query_bank)), CURRENT_06_ROOT / "tables" / "core_gap_query_bank.csv", "banco de queries derivado do core")

white_space = read_csv_if_exists(CURRENT_06_ROOT / "tables" / "white_space_candidates.csv")
if white_space is not None:
    add_rebuild("06", "Pipe2", "pipe2_white_space_candidates", int(len(white_space)), CURRENT_06_ROOT / "tables" / "white_space_candidates.csv", "candidatos iniciais de white space")

log("[06] snapshot coletado")


# 07 - Pipe 3
pipe3_summary = read_csv_if_exists(CURRENT_07_ROOT / "tables" / "paper_ready_retriever_summary.csv")
if pipe3_summary is not None:
    for row in pipe3_summary.itertuples(index=False):
        add_rebuild("07", "Pipe3", f"pipe3_{row.period}_{row.retriever}_n_queries", int(row.n_queries), CURRENT_07_ROOT / "tables" / "paper_ready_retriever_summary.csv", "queries agregadas por retriever no Pipe 3")
        add_rebuild("07", "Pipe3", f"pipe3_{row.period}_{row.retriever}_mean_top1_score", float(row.mean_top1_score), CURRENT_07_ROOT / "tables" / "paper_ready_retriever_summary.csv", "score medio top1 agregado no Pipe 3")

agent_outputs = read_csv_if_exists(CURRENT_07_ROOT / "tables" / "agent_outputs.csv")
if agent_outputs is not None:
    add_rebuild("07", "Pipe3", "pipe3_agent_outputs_count", int(len(agent_outputs)), CURRENT_07_ROOT / "tables" / "agent_outputs.csv", "quantidade de saidas grounded produzidas")
    add_rebuild("07", "Pipe3", "pipe3_agent_openai_ok_count", int((agent_outputs["agent_status"] == "openai_ok").sum()), CURRENT_07_ROOT / "tables" / "agent_outputs.csv", "quantidade de saidas com sucesso de OpenAI")

bm25_hits = read_csv_if_exists(CURRENT_07_ROOT / "tables" / "bm25_query_hits.csv")
if bm25_hits is not None:
    add_rebuild("07", "Pipe3", "pipe3_bm25_hits_rows", int(len(bm25_hits)), CURRENT_07_ROOT / "tables" / "bm25_query_hits.csv", "rows do baseline BM25 no Pipe 3")

log("[07] snapshot coletado")


# 08 - Pipe 4
pipe4_summary = read_csv_if_exists(CURRENT_08_ROOT / "tables" / "statistical_summary.csv")
if pipe4_summary is not None:
    for row in pipe4_summary.itertuples(index=False):
        add_rebuild("08", "Pipe4", f"pipe4_{row.period}_{row.retriever}_mean_top1_score", float(row.mean_top1_score), CURRENT_08_ROOT / "tables" / "statistical_summary.csv", "media bootstrap de top1 no Pipe 4")

pipe4_delta = read_csv_if_exists(CURRENT_08_ROOT / "tables" / "paired_deltas.csv")
if pipe4_delta is not None and len(pipe4_delta):
    for row in pipe4_delta.itertuples(index=False):
        add_rebuild("08", "Pipe4", f"pipe4_{row.period}_{row.metric}", float(row.mean), CURRENT_08_ROOT / "tables" / "paired_deltas.csv", "delta pareado bootstrap no Pipe 4")

pipe4_audit = read_csv_if_exists(CURRENT_08_ROOT / "tables" / "manual_audit_sample.csv")
if pipe4_audit is not None:
    add_rebuild("08", "Pipe4", "pipe4_manual_audit_rows", int(len(pipe4_audit)), CURRENT_08_ROOT / "tables" / "manual_audit_sample.csv", "tamanho da amostra de auditoria manual")

validation_manifest = read_json_if_exists(CURRENT_08_ROOT / "reports" / "validation_manifest.json")
if validation_manifest:
    add_rebuild("08", "Pipe4", "pipe4_bootstrap_rounds", int(validation_manifest.get("bootstrap_rounds", 0)), CURRENT_08_ROOT / "reports" / "validation_manifest.json", "numero de rodadas bootstrap do Pipe 4")

log("[08] snapshot coletado")

rebuild_df = pd.DataFrame(rebuild_rows)
rebuild_df.to_csv(TABLES_DIR / "rebuild_snapshot_metrics.csv", index=False)
display(rebuild_df.head(60))
'''


PIPE9_COMPARE = r'''
# ============================================================
# Comparacoes paper original vs rebuild + claims para manuscrito
# ============================================================

stage_banner("COMPARACOES E CLAIMS")

historical_df = pd.read_csv(TABLES_DIR / "historical_reference_metrics.csv")
rebuild_df = pd.read_csv(TABLES_DIR / "rebuild_snapshot_metrics.csv")

direct_keys = [
    "Nucleo_core_rows_used",
    "Nucleo_core_k",
    "PIML_core_rows_used",
    "PIML_core_k",
    "CombFinal_core_rows_used",
    "CombFinal_core_k",
    "ML_Multimodal_core_rows_used",
    "ML_Multimodal_core_k",
    "scibert_dapt_docs",
    "scibert_dapt_perplexity",
    "scibert_contrastive_pairs",
    "scibert_ab_docs",
    "scibert_ab_classes",
    "scibert_mrr_baseline",
    "scibert_mrr_specialized",
    "scibert_recall10_baseline",
    "scibert_recall10_specialized",
    "scibert_recall50_baseline",
    "scibert_recall50_specialized",
    "scibert_recall100_baseline",
    "scibert_recall100_specialized",
    "scibert_nmi_baseline",
    "scibert_nmi_specialized",
    "scibert_ari_baseline",
    "scibert_ari_specialized",
    "scibert_silhouette_baseline",
    "scibert_silhouette_specialized",
    "scibert_nearest_centroid_baseline",
    "scibert_nearest_centroid_specialized",
    "pipe2_application_docs_rows",
    "pipe2_index_map_rows",
]

expansion_keys = [
    "Nucleo_total_docs",
    "PIML_total_docs",
    "CombFinal_total_docs",
]


def build_comparison_df(metric_keys: list[str]) -> pd.DataFrame:
    hist = historical_df[historical_df["metric_key"].isin(metric_keys)].copy()
    reb = rebuild_df[rebuild_df["metric_key"].isin(metric_keys)].copy()
    merged = hist.merge(
        reb,
        on="metric_key",
        how="outer",
        suffixes=("_historical", "_rebuild"),
    )
    merged["historical_value_num"] = merged["value_historical"].map(normalize_number)
    merged["rebuild_value_num"] = merged["value_rebuild"].map(normalize_number)
    merged["abs_delta"] = pd.to_numeric(merged["rebuild_value_num"], errors="coerce") - pd.to_numeric(merged["historical_value_num"], errors="coerce")
    merged["pct_delta"] = np.where(
        pd.to_numeric(merged["historical_value_num"], errors="coerce").abs().gt(1e-9),
        merged["abs_delta"] / pd.to_numeric(merged["historical_value_num"], errors="coerce"),
        np.nan,
    )
    merged["status"] = [
        comparison_status(h, r)
        for h, r in zip(merged["historical_value_num"], merged["rebuild_value_num"])
    ]
    return merged


direct_df = build_comparison_df(direct_keys)
expansion_df = build_comparison_df(expansion_keys)
historical_only_df = historical_df.loc[~historical_df["metric_key"].isin(rebuild_df["metric_key"])].copy()
rebuild_only_df = rebuild_df.loc[~rebuild_df["metric_key"].isin(historical_df["metric_key"])].copy()

direct_df.to_csv(TABLES_DIR / "direct_comparisons.csv", index=False)
expansion_df.to_csv(TABLES_DIR / "expansion_comparisons.csv", index=False)
historical_only_df.to_csv(TABLES_DIR / "historical_only_metrics.csv", index=False)
rebuild_only_df.to_csv(TABLES_DIR / "rebuild_only_metrics.csv", index=False)

hist_lookup = historical_df.set_index("metric_key")["value"].to_dict()
rebuild_lookup = rebuild_df.set_index("metric_key")["value"].to_dict()

claims_rows = [
    {
        "claim_id": "C1",
        "claim": "O rebuild preserva a familia de metrica do Phase I e o retriever especializado continua superando o baseline no benchmark paper-facing.",
        "historical_anchor": f"MRR FT={fmt_num(hist_lookup.get('scibert_mrr_specialized'))}; Recall@10 FT={fmt_num(hist_lookup.get('scibert_recall10_specialized'))}",
        "rebuild_anchor": f"MRR FT={fmt_num(rebuild_lookup.get('scibert_mrr_specialized'))}; Recall@10 FT={fmt_num(rebuild_lookup.get('scibert_recall10_specialized'))}",
        "recommended_use": "usar na secao metodologica/experimental como continuidade controlada do paper original",
    },
    {
        "claim_id": "C2",
        "claim": "O major review adiciona um espelhamento temporal explicito via holdout sem contaminar o treino do retriever especializado.",
        "historical_anchor": "pipeline historico sem split temporal auditavel",
        "rebuild_anchor": f"Nucleo total={fmt_num(rebuild_lookup.get('Nucleo_total_docs'))}; PIML total={fmt_num(rebuild_lookup.get('PIML_total_docs'))}; CombFinal total={fmt_num(rebuild_lookup.get('CombFinal_total_docs'))}",
        "recommended_use": "usar na resposta ao major review e na secao de desenho do estudo",
    },
    {
        "claim_id": "C3",
        "claim": "A ponte entre retriever e experimento final agora e rastreavel por docs_master, index_map, manifests e pacotes por camada.",
        "historical_anchor": f"pipe2 docs_master historico={fmt_num(hist_lookup.get('pipe2_application_docs_rows'))}",
        "rebuild_anchor": f"pipe2 core docs_master={fmt_num(rebuild_lookup.get('pipe2_application_docs_rows'))}; pipe4 bootstrap={fmt_num(rebuild_lookup.get('pipe4_bootstrap_rounds'))}",
        "recommended_use": "usar na narrativa de reproducibilidade e auditabilidade",
    },
    {
        "claim_id": "C4",
        "claim": "O pacote final do major review separa a evidencia principal do retriever (Pipe 05) da robustez temporal e estatistica (Pipes 07 e 08).",
        "historical_anchor": "paper original condensava a historia real do pipeline em uma narrativa unica",
        "rebuild_anchor": "Pipes 05, 07, 08 e 09 deixam a historia metodologica explicita e modular",
        "recommended_use": "usar na organizacao da secao Results + Discussion + Appendix",
    },
]

claims_df = pd.DataFrame(claims_rows)
claims_df.to_csv(TABLES_DIR / "manuscript_claims.csv", index=False)

display(direct_df.head(40))
display(expansion_df)
display(claims_df)
'''


PIPE9_EXPORT = r'''
# ============================================================
# Storyline, pacote Excel e inventario
# ============================================================

stage_banner("EXPORT FINAL DO PIPE 9")

historical_df = pd.read_csv(TABLES_DIR / "historical_reference_metrics.csv")
rebuild_df = pd.read_csv(TABLES_DIR / "rebuild_snapshot_metrics.csv")
direct_df = pd.read_csv(TABLES_DIR / "direct_comparisons.csv")
expansion_df = pd.read_csv(TABLES_DIR / "expansion_comparisons.csv")
layer_crosswalk = pd.read_csv(TABLES_DIR / "layer_crosswalk.csv")
evolution_df = pd.read_csv(TABLES_DIR / "major_review_evolution_points.csv")
claims_df = pd.read_csv(TABLES_DIR / "manuscript_claims.csv")

hist_lookup = historical_df.set_index("metric_key")["value"].to_dict()
rebuild_lookup = rebuild_df.set_index("metric_key")["value"].to_dict()

story_lines = [
    "# Methodological Storyline Pack",
    "",
    "## 1. O que o paper original fez de fato",
    "",
    "1. Consolidou bases via scripts R.",
    "2. Derivou estrutura semantica upstream com `Abstract_LLM_*`.",
    "3. Treinou o retriever especializado `SciBERT-SolarPhysics-Search`.",
    "4. Aplicou o retriever ao corpus `ML` para analytics e gaps.",
    "5. Gerou o experimento final em `Piepe_3_Scientometrics`.",
    "6. Fez uma validacao estatistica final menos estruturada.",
    "",
    "## 2. O que o rebuild canonico preservou",
    "",
    f"- Nucleo core rows used: historico={fmt_num(hist_lookup.get('Nucleo_core_rows_used'))} vs rebuild={fmt_num(rebuild_lookup.get('Nucleo_core_rows_used'))}.",
    f"- PIML core rows used: historico={fmt_num(hist_lookup.get('PIML_core_rows_used'))} vs rebuild={fmt_num(rebuild_lookup.get('PIML_core_rows_used'))}.",
    f"- CombFinal core rows used: historico={fmt_num(hist_lookup.get('CombFinal_core_rows_used'))} vs rebuild={fmt_num(rebuild_lookup.get('CombFinal_core_rows_used'))}.",
    f"- ML core rows used: historico={fmt_num(hist_lookup.get('ML_Multimodal_core_rows_used'))} vs rebuild={fmt_num(rebuild_lookup.get('ML_Multimodal_core_rows_used'))}.",
    f"- k historico dos corpora tematicos = 30; rebuild: Nucleo={fmt_num(rebuild_lookup.get('Nucleo_core_k'))}, PIML={fmt_num(rebuild_lookup.get('PIML_core_k'))}, CombFinal={fmt_num(rebuild_lookup.get('CombFinal_core_k'))}, ML={fmt_num(rebuild_lookup.get('ML_Multimodal_core_k'))}.",
    "",
    "## 3. O que o Phase I mostrou",
    "",
    f"- DAPT docs: historico={fmt_num(hist_lookup.get('scibert_dapt_docs'))} vs rebuild={fmt_num(rebuild_lookup.get('scibert_dapt_docs'))}.",
    f"- DAPT perplexity: historico={fmt_num(hist_lookup.get('scibert_dapt_perplexity'))} vs rebuild={fmt_num(rebuild_lookup.get('scibert_dapt_perplexity'))}.",
    f"- Contrastive pairs: historico={fmt_num(hist_lookup.get('scibert_contrastive_pairs'))} vs rebuild={fmt_num(rebuild_lookup.get('scibert_contrastive_pairs'))}.",
    f"- A/B docs/classes: historico={fmt_num(hist_lookup.get('scibert_ab_docs'))}/{fmt_num(hist_lookup.get('scibert_ab_classes'))} vs rebuild={fmt_num(rebuild_lookup.get('scibert_ab_docs'))}/{fmt_num(rebuild_lookup.get('scibert_ab_classes'))}.",
    f"- MRR baseline->specialized no paper original: {fmt_num(hist_lookup.get('scibert_mrr_baseline'))} -> {fmt_num(hist_lookup.get('scibert_mrr_specialized'))}.",
    f"- MRR baseline->specialized no rebuild: {fmt_num(rebuild_lookup.get('scibert_mrr_baseline'))} -> {fmt_num(rebuild_lookup.get('scibert_mrr_specialized'))}.",
    f"- Recall@10 baseline->specialized no rebuild: {fmt_num(rebuild_lookup.get('scibert_recall10_baseline'))} -> {fmt_num(rebuild_lookup.get('scibert_recall10_specialized'))}.",
    "",
    "## 4. O que mudou para responder ao major review",
    "",
    "- Introducao explicita do split `core/holdout` em todas as camadas de aplicacao.",
    "- Treino do retriever mantido apenas em `Nucleo_core + PIML_core + CombFinal_core`.",
    "- Inclusao do baseline `BM25` no downstream.",
    "- Reconstrucao de `docs_master`, `index_map`, indices FAISS e bundles por periodo.",
    "- Pacote estatistico final com bootstrap CI, deltas e amostra de auditoria manual.",
    "- Publicacao do modelo especializado no Hugging Face.",
    "",
    "## 5. Como contar a historia no manuscrito revisado",
    "",
    "- Use o `core` como regime principal de replicacao do paper original.",
    "- Use o `holdout` como espelhamento temporal da mesma familia de aplicacoes.",
    "- Use o `05` como evidencia principal de que o retriever especializado preserva o comportamento central do paper e supera o baseline em metricas rank-based e de clustering.",
    "- Use o `06` como camada de rastreabilidade do corpus de aplicacao.",
    "- Use o `07` como camada de bundles grounded e exemplos de gap.",
    "- Use o `08` como robustez estatistica e validacao temporal; evite vender `raw score delta` entre retrievers como headline principal.",
    "",
    "## 6. Claims recomendados",
    "",
]

for row in claims_df.itertuples(index=False):
    story_lines.append(f"- {row.claim_id}: {row.claim}")
    story_lines.append(f"  Historico: {row.historical_anchor}")
    story_lines.append(f"  Rebuild: {row.rebuild_anchor}")
    story_lines.append(f"  Uso: {row.recommended_use}")

story_md = "\n".join(story_lines) + "\n"
(REPORTS_DIR / "methodological_storyline.md").write_text(story_md, encoding="utf-8")

manuscript_notes = pd.DataFrame(
    [
        {
            "section": "Methods - Study Design",
            "what_to_say": "explicar o split core/holdout e o treino core-only do retriever especializado",
            "primary_evidence": str(TABLES_DIR / "layer_crosswalk.csv"),
        },
        {
            "section": "Methods - Retriever Training",
            "what_to_say": "comparar o Phase I historico com o rebuild do Pipe 05",
            "primary_evidence": str(TABLES_DIR / "direct_comparisons.csv"),
        },
        {
            "section": "Results - Gap Retrieval",
            "what_to_say": "usar os bundles grounded do Pipe 07 e os candidatos de white space do Pipe 06",
            "primary_evidence": str(CURRENT_07_ROOT / "tables" / "agent_outputs.csv"),
        },
        {
            "section": "Results - Robustness",
            "what_to_say": "usar CIs e deltas do Pipe 08 como robustez temporal e estatistica",
            "primary_evidence": str(CURRENT_08_ROOT / "tables" / "statistical_summary.csv"),
        },
    ]
)
manuscript_notes.to_csv(TABLES_DIR / "manuscript_section_notes.csv", index=False)


def inventory_root(root: Path, scope_label: str) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    allowed_suffixes = {".csv", ".json", ".jsonl", ".md", ".xlsx", ".ipynb", ".pdf", ".png", ".faiss", ".npy", ".txt"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            rows.append(
                {
                    "scope": scope_label,
                    "artifact": str(path),
                    "size_bytes": path.stat().st_size,
                    "suffix": path.suffix.lower(),
                }
            )
    return rows


inventory_rows = []
for corpus in CORPORA:
    inventory_rows.extend(inventory_root(DATA_ROOT / corpus / "04_rebuild_outputs" / "00_consolidacao", f"current_00_{corpus}"))
    inventory_rows.extend(inventory_root(DATA_ROOT / corpus / "04_rebuild_outputs" / "01_abstract_llm", f"current_01_{corpus}"))

inventory_rows.extend(inventory_root(CURRENT_05_ROOT, "current_05"))
inventory_rows.extend(inventory_root(CURRENT_06_ROOT, "current_06"))
inventory_rows.extend(inventory_root(CURRENT_07_ROOT, "current_07"))
inventory_rows.extend(inventory_root(CURRENT_08_ROOT, "current_08"))
inventory_rows.extend(inventory_root(HISTORICAL_ROOT, "historical_root"))
inventory_rows.extend(inventory_root(HISTORICAL_FINAL_ROOT, "historical_final_root"))

inventory_df = pd.DataFrame(inventory_rows)
inventory_df.to_csv(TABLES_DIR / "artifact_inventory.csv", index=False)

with pd.ExcelWriter(REPORTS_DIR / "pipe09_method_story_pack.xlsx", engine="openpyxl") as writer:
    historical_df.to_excel(writer, sheet_name="historical_refs", index=False)
    rebuild_df.to_excel(writer, sheet_name="rebuild_snapshot", index=False)
    direct_df.to_excel(writer, sheet_name="direct_comparisons", index=False)
    expansion_df.to_excel(writer, sheet_name="expansion", index=False)
    layer_crosswalk.to_excel(writer, sheet_name="layer_crosswalk", index=False)
    evolution_df.to_excel(writer, sheet_name="major_review_delta", index=False)
    claims_df.to_excel(writer, sheet_name="manuscript_claims", index=False)
    manuscript_notes.to_excel(writer, sheet_name="section_notes", index=False)

with open(REPORTS_DIR / "story_pack_manifest.json", "w", encoding="utf-8") as fh:
    json.dump(
        {
            "run_ts": RUN_TS,
            "write_root": str(WRITE_ROOT),
            "historical_root": str(HISTORICAL_ROOT),
            "historical_final_root": str(HISTORICAL_FINAL_ROOT),
            "current_roots": {
                "05": str(CURRENT_05_ROOT),
                "06": str(CURRENT_06_ROOT),
                "07": str(CURRENT_07_ROOT),
                "08": str(CURRENT_08_ROOT),
            },
            "tables": [
                "historical_reference_metrics.csv",
                "rebuild_snapshot_metrics.csv",
                "direct_comparisons.csv",
                "expansion_comparisons.csv",
                "layer_crosswalk.csv",
                "major_review_evolution_points.csv",
                "manuscript_claims.csv",
                "manuscript_section_notes.csv",
                "artifact_inventory.csv",
            ],
            "reports": [
                "methodological_storyline.md",
                "pipe09_method_story_pack.xlsx",
            ],
        },
        fh,
        indent=2,
        ensure_ascii=False,
    )

display(direct_df.head(30))
display(claims_df)
display(manuscript_notes)
print("Arquivos finais salvos em:", WRITE_ROOT)
'''


def build_abstract_notebook(corpus: str, notebook_id: str, filename: str) -> tuple[Path, list[dict]]:
    mapping = {
        "CORPUS": corpus,
        "NOTEBOOK_ID": notebook_id,
    }
    cells = [
        md_cell(render_template(ABSTRACT_MD, mapping)),
        code_cell(render_template(ABSTRACT_INSTALL, mapping)),
        code_cell(render_template(ABSTRACT_IMPORTS, mapping)),
        code_cell(render_template(ABSTRACT_CONFIG, mapping)),
        code_cell(render_template(ABSTRACT_PATHS, mapping)),
        code_cell(render_template(ABSTRACT_PREP, mapping)),
        code_cell(render_template(ABSTRACT_HELPERS, mapping)),
        code_cell(render_template(ABSTRACT_RUN, mapping)),
        code_cell(render_template(ABSTRACT_COMPARE, mapping)),
    ]
    return NOTEBOOK_ROOT / filename, cells


def build_scibert_notebook() -> tuple[Path, list[dict]]:
    cells = [
        md_cell(SCIBERT_MD),
        code_cell(SCIBERT_INSTALL),
        code_cell(SCIBERT_IMPORTS),
        code_cell(SCIBERT_CONFIG),
        code_cell(SCIBERT_LOGGING),
        code_cell(SCIBERT_LOAD),
        code_cell(SCIBERT_PAIRS),
        code_cell(SCIBERT_DAPT),
        code_cell(SCIBERT_CONTRASTIVE),
        code_cell(SCIBERT_HISTORICAL_LABELS),
        code_cell(SCIBERT_HISTORICAL_AB),
        code_cell(SCIBERT_PUBLISH),
        code_cell(SCIBERT_EXPORT),
    ]
    return NOTEBOOK_ROOT / "05_scibert_solarphysics_search_rebuild.ipynb", cells


def build_pipe2_notebook() -> tuple[Path, list[dict]]:
    cells = [
        md_cell(PIPE2_MD),
        code_cell(PIPE2_INSTALL),
        code_cell(PIPE2_IMPORTS),
        code_cell(PIPE2_CONFIG),
        code_cell(PIPE2_LOGGING),
        code_cell(PIPE2_DOCS),
        code_cell(PIPE2_EMBED),
        code_cell(PIPE2_QUERIES),
        code_cell(PIPE2_ANALYTICS),
        code_cell(PIPE2_MANIFEST),
    ]
    return NOTEBOOK_ROOT / "06_pipe2_retriever_analytics_rebuild.ipynb", cells


def build_pipe3_notebook() -> tuple[Path, list[dict]]:
    cells = [
        md_cell(PIPE3_MD),
        code_cell(PIPE3_INSTALL),
        code_cell(PIPE3_IMPORTS),
        code_cell(PIPE3_CONFIG),
        code_cell(PIPE3_LOGGING),
        code_cell(PIPE3_LOAD),
        code_cell(PIPE3_BM25),
        code_cell(PIPE3_BUNDLES),
        code_cell(PIPE3_AGENT),
        code_cell(PIPE3_EXPORT),
    ]
    return NOTEBOOK_ROOT / "07_pipe3_agent_scientometrics_rebuild.ipynb", cells


def build_pipe4_notebook() -> tuple[Path, list[dict]]:
    cells = [
        md_cell(PIPE4_MD),
        code_cell(PIPE4_INSTALL),
        code_cell(PIPE4_IMPORTS),
        code_cell(PIPE4_CONFIG),
        code_cell(PIPE4_LOGGING),
        code_cell(PIPE4_LOAD),
        code_cell(PIPE4_STATS),
        code_cell(PIPE4_DELTAS),
        code_cell(PIPE4_AUDIT),
        code_cell(PIPE4_MANIFEST),
    ]
    return NOTEBOOK_ROOT / "08_pipe4_statistical_validation_rebuild.ipynb", cells


def build_pipe9_notebook() -> tuple[Path, list[dict]]:
    cells = [
        md_cell(PIPE9_MD),
        code_cell(PIPE9_INSTALL),
        code_cell(PIPE9_IMPORTS),
        code_cell(PIPE9_CONFIG),
        code_cell(PIPE9_LOGGING),
        code_cell(PIPE9_HISTORICAL),
        code_cell(PIPE9_REBUILD),
        code_cell(PIPE9_COMPARE),
        code_cell(PIPE9_EXPORT),
    ]
    return NOTEBOOK_ROOT / "09_methodological_storyline_pack.ipynb", cells


def main() -> None:
    notebook_specs = [
        build_abstract_notebook("Nucleo", "01", "01_abstract_llm_nucleo_core_holdout.ipynb"),
        build_abstract_notebook("PIML", "02", "02_abstract_llm_piml_core_holdout.ipynb"),
        build_abstract_notebook("CombFinal", "03", "03_abstract_llm_combfinal_core_holdout.ipynb"),
        build_abstract_notebook("ML_Multimodal", "04", "04_abstract_llm_ml_core_holdout.ipynb"),
        build_scibert_notebook(),
        build_pipe2_notebook(),
        build_pipe3_notebook(),
        build_pipe4_notebook(),
        build_pipe9_notebook(),
    ]

    for path, cells in notebook_specs:
        write_notebook(path, cells)


if __name__ == "__main__":
    main()
