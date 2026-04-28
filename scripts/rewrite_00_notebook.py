from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path


NOTEBOOK_PATH = Path(
    r"C:\Users\andre\odrive\Google Drive\Unicamp\artigo bibliometria\grounded-scientometrics-solarphysics-retrieval\notebooks\00_consolidacao_rebuild_core_holdout.ipynb"
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


CELL_4 = r'''
from google.colab import drive
drive.mount("/content/drive")

import json
import re
import shutil
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import bibtexparser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
import seaborn as sns
from dateutil import parser as dtparser
from rapidfuzz import fuzz
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")

pd.set_option("display.max_columns", 160)
pd.set_option("display.max_colwidth", 220)
sns.set_theme(style="whitegrid")

print("Bibliotecas carregadas com sucesso.")
print("Timestamp:", datetime.now().isoformat(timespec="seconds"))
'''


CELL_5 = r'''
# =========================
# CONFIGURACAO GERAL
# =========================

DRIVE_ROOT = Path("/content/drive/MyDrive/Unicamp")
PROJECT_ROOT = DRIVE_ROOT / "artigo bibliometria" / "grounded-scientometrics-solarphysics-retrieval"
DATA_ROOT = DRIVE_ROOT / "artigo bibliometria" / "base de dados" / "Artigo_Bibliometria Base Bruta" / "BASES_UNIFICADAS_POR_TEMA"

RESET_CORPUS_OUTPUTS_ON_START = True
KEEP_GLOBAL_RUN_LOG_HISTORY = True
WRITE_RDATA = True

CORE_END = pd.Timestamp("2025-09-30")
HOLDOUT_START = pd.Timestamp("2025-10-01")
HOLDOUT_END = pd.Timestamp("2026-03-31")

CORPORA = ["Nucleo", "PIML", "CombFinal", "ML_Multimodal"]
BASE_PRIORITY = {"WOS": 3, "SCOPUS": 2, "OPENALEX": 1, "UNKNOWN": 0}
SUPPORTED_RAW_EXTENSIONS = {".bib", ".csv", ".txt"}

RUN_FUZZY_DEDUP = True
FUZZY_THRESHOLD = 98
FUZZY_MAX_GROUP_SIZE = 250

FILE_PROGRESS_EVERY = 5
EXACT_GROUP_PROGRESS_EVERY = 100
FUZZY_GROUP_PROGRESS_EVERY = 100

KEY_FIELDS = ["AU", "TI", "SO", "PY", "DI", "AB", "DE", "ID", "TC", "C1", "publication_date"]
CANONICAL_COLS = [
    "corpus", "base", "bucket", "source_file", "slice_label", "record_origin",
    "AU", "TI", "SO", "PY", "DI", "DE", "ID", "AB", "CR", "LA", "DT", "C1",
    "TC", "VL", "IS", "BP", "EP", "publication_date", "month_hint",
    "open_access", "funding", "publisher", "native_row_json"
]

assert PROJECT_ROOT.exists(), f"PROJECT_ROOT nao encontrado: {PROJECT_ROOT}"
assert DATA_ROOT.exists(), f"DATA_ROOT nao encontrado: {DATA_ROOT}"

print("PROJECT_ROOT =", PROJECT_ROOT)
print("DATA_ROOT    =", DATA_ROOT)
print("CORE_END     =", CORE_END.date())
print("HOLDOUT      =", HOLDOUT_START.date(), "->", HOLDOUT_END.date())
print("RESET_CORPUS_OUTPUTS_ON_START =", RESET_CORPUS_OUTPUTS_ON_START)
print("WRITE_RDATA                  =", WRITE_RDATA)
print("Fuzzy dedup                  =", RUN_FUZZY_DEDUP, "| threshold =", FUZZY_THRESHOLD, "| max_group =", FUZZY_MAX_GROUP_SIZE)
print("Observabilidade              =", {"file_every": FILE_PROGRESS_EVERY, "exact_group_every": EXACT_GROUP_PROGRESS_EVERY, "fuzzy_group_every": FUZZY_GROUP_PROGRESS_EVERY})
'''


CELL_6 = r'''
# =========================
# SAIDAS, LOG E HELPERS DE PROGRESSO
# =========================

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


def corpus_output_dir(corpus: str) -> Path:
    return DATA_ROOT / corpus / "04_rebuild_outputs" / "00_consolidacao"


def elapsed_seconds() -> float:
    return time.time() - PIPE_START_TS


def fmt_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


GLOBAL_LOG_DIR = ensure_dir(PROJECT_ROOT / "outputs" / "camada0_logs")
GLOBAL_LOG_FILE = GLOBAL_LOG_DIR / f"00_consolidacao_{RUN_TS}.txt"
CORPUS_LOG_FILES = {}


def log(message: str, corpus: str | None = None) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{now} | +{fmt_seconds(elapsed_seconds())}]"
    line = f"{prefix} {message}"
    print(line, flush=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if corpus is not None and corpus in CORPUS_LOG_FILES:
        with open(CORPUS_LOG_FILES[corpus], "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def stage_banner(title: str, corpus: str | None = None) -> None:
    bar = "=" * 96
    log(bar, corpus=corpus)
    log(title, corpus=corpus)
    log(bar, corpus=corpus)


for corpus in CORPORA:
    base_dir = corpus_output_dir(corpus)
    if RESET_CORPUS_OUTPUTS_ON_START:
        reset_dir(base_dir)
        reset_mode = "reset"
    else:
        ensure_dir(base_dir)
        reset_mode = "reuse"

    ensure_dir(base_dir / "native_unions")
    ensure_dir(base_dir / "logs")
    ensure_dir(base_dir / "article_validation")
    ensure_dir(base_dir / "exploratory_analytics")
    ensure_dir(base_dir / "article_bibliometrics")
    CORPUS_LOG_FILES[corpus] = base_dir / "logs" / f"{corpus}_00_consolidacao_{RUN_TS}.txt"
    log(f"Output {reset_mode} pronto para {corpus}: {base_dir}", corpus=corpus)

print("GLOBAL_LOG_FILE =", GLOBAL_LOG_FILE)
'''


CELL_7 = r'''
# =========================
# INVENTARIO DE ARQUIVOS
# =========================

RAW_BUCKETS = {
    "historico_raw": "01_historico_bruto_disponivel",
    "historico_consolidado_ref": "02_historico_consolidado",
    "complemento_raw": "03_complemento_bruto_2025-09_2026-03",
}


def detect_base(path: Path) -> str:
    name = path.name.lower()
    full = str(path).lower()
    if "openalex" in name or "openalex" in full:
        return "OPENALEX"
    if "scopus" in name or "scopus" in full:
        return "SCOPUS"
    if "wos" in name or "webofscience" in name or "wos" in full:
        return "WOS"
    return "UNKNOWN"


def detect_bucket(path: Path) -> str:
    full = str(path)
    for bucket_name, bucket_dir in RAW_BUCKETS.items():
        if bucket_dir in full:
            return bucket_name
    return "unknown"


def parse_slice_from_name(name: str):
    month_match = re.search(r"(20\d{2}-\d{2})(?:_part\d+)?", name)
    if month_match:
        return month_match.group(1)
    year_match = re.search(r"(20\d{2}-20\d{2})(?:_\d+)?", name)
    if year_match:
        return year_match.group(1)
    return None


stage_banner("INVENTARIO DE ARQUIVOS")
inventory_rows = []

for corpus in CORPORA:
    corpus_root = DATA_ROOT / corpus
    log(f"[inventory] varrendo {corpus_root}", corpus=corpus)
    corpus_files = [p for p in corpus_root.rglob("*") if p.is_file() and not p.name.startswith("manifest_")]
    for idx, file_path in enumerate(corpus_files, start=1):
        inventory_rows.append(
            {
                "corpus": corpus,
                "bucket": detect_bucket(file_path),
                "base": detect_base(file_path),
                "ext": file_path.suffix.lower(),
                "slice_label": parse_slice_from_name(file_path.name),
                "file_name": file_path.name,
                "full_path": str(file_path),
                "size_bytes": file_path.stat().st_size,
            }
        )
        if idx % 250 == 0 or idx == len(corpus_files):
            log(f"[inventory] {corpus} -> {idx}/{len(corpus_files)} arquivos indexados", corpus=corpus)

inventory_df = pd.DataFrame(inventory_rows).sort_values(["corpus", "bucket", "base", "file_name"]).reset_index(drop=True)

inventory_summary = (
    inventory_df.groupby(["corpus", "bucket", "base", "ext"], dropna=False)
    .size()
    .reset_index(name="n_files")
    .sort_values(["corpus", "bucket", "base", "ext"])
)

display(inventory_df.head(20))
display(inventory_summary)

inventory_all_out = GLOBAL_LOG_DIR / f"00_inventory_all_corpora_{RUN_TS}.csv"
inventory_df.to_csv(inventory_all_out, index=False)

for corpus in CORPORA:
    out = corpus_output_dir(corpus) / "raw_inventory.csv"
    inventory_df[inventory_df["corpus"] == corpus].to_csv(out, index=False)
    log(f"Inventario salvo: {out}", corpus=corpus)

print("Inventario global salvo em:", inventory_all_out)
'''


CELL_8 = r'''
# =========================
# HELPERS DE NORMALIZACAO E COBERTURA
# =========================

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
    text = re.sub(r"\s+", " ", text)
    return text


def clean_list_like(value):
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA
    parts = [p.strip() for p in re.split(r"[;|]", str(text)) if p and p.strip()]
    parts = list(dict.fromkeys(parts))
    return "; ".join(parts) if parts else pd.NA


def coalesce_value(*values):
    for value in values:
        text = clean_text(value)
        if pd.notna(text):
            return text
    return pd.NA


def normalize_doi(value):
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA
    text = str(text).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = text.replace("doi:", "").strip()
    return text if text else pd.NA


def normalize_title(value):
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else pd.NA


def normalize_source_title(value):
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA
    text = str(text).upper().strip()
    text = re.sub(r"\s+", " ", text)
    return text if text else pd.NA


def safe_int(value):
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
    match = re.search(r"-?\d+", text.replace(",", ""))
    if not match:
        return pd.NA
    try:
        return int(match.group(0))
    except Exception:
        return pd.NA


def parse_publication_date(value):
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA
    try:
        parsed = dtparser.parse(str(text), default=datetime(1900, 1, 1))
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return pd.NA


def parse_pages(value):
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA, pd.NA
    parts = re.split(r"\s*[-–]\s*", str(text), maxsplit=1)
    if len(parts) == 2:
        return clean_text(parts[0]), clean_text(parts[1])
    return clean_text(text), pd.NA


def parse_cited_by_from_note(value):
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA
    match = re.search(r"Cited by:\s*(\d+)", str(text), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return pd.NA


def present_mask(series: pd.Series) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series([], dtype=bool)
    s = series.astype("string")
    return (~s.isna()) & (s.str.strip() != "")


def present_pct(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return float(present_mask(series).mean() * 100)


def count_pipe_values(value) -> int:
    if value is None:
        return 0
    try:
        if pd.isna(value):
            return 0
    except Exception:
        pass
    parts = [p.strip() for p in str(value).split("|") if p.strip()]
    return len(parts)


def completeness_score(df: pd.DataFrame, fields: list[str] | None = None) -> pd.Series:
    fields = fields or ["AU", "TI", "SO", "PY", "DI", "DE", "ID", "AB", "CR", "C1", "TC", "publication_date"]
    present_frames = []
    for field in fields:
        if field in df.columns:
            present_frames.append(present_mask(df[field]).astype(int))
        else:
            present_frames.append(pd.Series(0, index=df.index, dtype="int64"))
    return pd.concat(present_frames, axis=1).sum(axis=1)


def json_row(record: dict) -> str:
    clean = {}
    for k, v in record.items():
        if isinstance(v, (np.integer, np.floating)):
            clean[k] = v.item()
        elif isinstance(v, (dict, list)):
            clean[k] = v
        elif v is None:
            clean[k] = None
        else:
            try:
                clean[k] = None if pd.isna(v) else v
            except Exception:
                clean[k] = v
    return json.dumps(clean, ensure_ascii=False, default=str)


def safe_read_csv(path: Path, **kwargs):
    if path.exists():
        return pd.read_csv(path, **kwargs)
    return pd.DataFrame()


def split_terms(series: pd.Series) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype="string")
    values = []
    for value in series.dropna().astype(str):
        parts = [p.strip() for p in re.split(r";|\|", value) if p.strip()]
        values.extend(parts)
    return pd.Series(values, dtype="string")


def derive_month_series(df: pd.DataFrame) -> pd.Series:
    month_from_pub = pd.to_datetime(df.get("publication_date", pd.Series(index=df.index, dtype="string")), errors="coerce")
    month_str = month_from_pub.dt.to_period("M").astype("string")
    if "month_hint" in df.columns:
        fallback = df["month_hint"].astype("string").str.extract(r"(20\d{2}-\d{2})", expand=False)
        month_str = month_str.fillna(fallback)
    if "slice_label" in df.columns:
        fallback2 = df["slice_label"].astype("string").str.extract(r"(20\d{2}-\d{2})", expand=False)
        month_str = month_str.fillna(fallback2)
    return month_str
'''


CELL_9 = r'''
# =========================
# PARSERS NATIVOS CORRIGIDOS
# =========================

def empty_record():
    return {column: pd.NA for column in CANONICAL_COLS}


def parse_bibtex_file(file_path: Path, corpus: str, base: str, bucket: str, slice_label: str):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        bib_db = bibtexparser.load(fh)

    rows = []
    native_rows = []

    for entry in bib_db.entries:
        bp, ep = parse_pages(entry.get("pages"))
        rec = empty_record()
        rec.update({
            "corpus": corpus,
            "base": base,
            "bucket": bucket,
            "source_file": file_path.name,
            "slice_label": slice_label,
            "record_origin": f"bibtex_{base.lower()}",
            "AU": clean_list_like(entry.get("author")),
            "TI": clean_text(entry.get("title")),
            "SO": clean_text(coalesce_value(entry.get("journal"), entry.get("booktitle"), entry.get("source"))),
            "PY": safe_int(coalesce_value(entry.get("year"), entry.get("publication_year"))),
            "DI": normalize_doi(entry.get("doi")),
            "DE": clean_list_like(coalesce_value(entry.get("author_keywords"), entry.get("keywords"))),
            "ID": clean_list_like(entry.get("keywords")),
            "AB": clean_text(entry.get("abstract")),
            "CR": clean_text(coalesce_value(entry.get("references"), entry.get("ref"))),
            "LA": clean_text(entry.get("language")),
            "DT": clean_text(coalesce_value(entry.get("document_type"), entry.get("type"))),
            "C1": clean_text(coalesce_value(entry.get("affiliations"), entry.get("affiliation"), entry.get("address"))),
            "TC": safe_int(coalesce_value(entry.get("citedby"), entry.get("times_cited"), parse_cited_by_from_note(entry.get("note")))),
            "VL": clean_text(entry.get("volume")),
            "IS": clean_text(entry.get("number")),
            "BP": bp,
            "EP": ep,
            "publication_date": parse_publication_date(coalesce_value(entry.get("date"), entry.get("year"))),
            "month_hint": clean_text(slice_label),
            "open_access": clean_text(coalesce_value(entry.get("open_access"), entry.get("oa_status"))),
            "funding": clean_text(coalesce_value(entry.get("funding"), entry.get("funders"))),
            "publisher": clean_text(entry.get("publisher")),
        })
        rec["native_row_json"] = json_row(entry)
        rows.append(rec)
        native_rows.append({**entry, "corpus": corpus, "base": base, "bucket": bucket, "source_file": file_path.name, "slice_label": slice_label})

    return pd.DataFrame(rows), pd.DataFrame(native_rows)


def parse_scopus_csv(file_path: Path, corpus: str, base: str, bucket: str, slice_label: str):
    native_df = pd.read_csv(file_path, dtype=str, low_memory=False)
    rows = []
    proxy_date = f"{slice_label}-01" if slice_label else pd.NA

    for _, row in native_df.iterrows():
        rec = empty_record()
        rec.update({
            "corpus": corpus,
            "base": base,
            "bucket": bucket,
            "source_file": file_path.name,
            "slice_label": slice_label,
            "record_origin": "scopus_csv",
            "AU": clean_list_like(coalesce_value(row.get("Authors"), row.get("Author full names"))),
            "TI": clean_text(row.get("Title")),
            "SO": clean_text(row.get("Source title")),
            "PY": safe_int(row.get("Year")),
            "DI": normalize_doi(row.get("DOI")),
            "DE": clean_list_like(row.get("Author Keywords")),
            "ID": clean_list_like(row.get("Index Keywords")),
            "AB": clean_text(row.get("Abstract")),
            "CR": clean_text(row.get("References")),
            "LA": clean_text(row.get("Language of Original Document")),
            "DT": clean_text(coalesce_value(row.get("Document Type"), row.get("Publication Stage"))),
            "C1": clean_text(coalesce_value(row.get("Affiliations"), row.get("Authors with affiliations"))),
            "TC": safe_int(row.get("Cited by")),
            "VL": clean_text(row.get("Volume")),
            "IS": clean_text(row.get("Issue")),
            "BP": clean_text(row.get("Page start")),
            "EP": clean_text(row.get("Page end")),
            # Para exports mensais do Scopus, usamos Publication Date quando existir,
            # depois Cover Date e, em ultimo caso, a proxy YYYY-MM-01 derivada do slice.
            "publication_date": parse_publication_date(
                coalesce_value(
                    row.get("Publication Date"),
                    row.get("Cover Date"),
                    proxy_date,
                )
            ),
            "month_hint": clean_text(slice_label),
            "open_access": clean_text(row.get("Open Access")),
            "funding": clean_text(coalesce_value(row.get("Funding Details"), row.get("Funding Texts"))),
            "publisher": clean_text(row.get("Publisher")),
        })
        rec["native_row_json"] = json_row(row.to_dict())
        rows.append(rec)

    return pd.DataFrame(rows), native_df


def parse_openalex_csv(file_path: Path, corpus: str, base: str, bucket: str, slice_label: str):
    native_df = pd.read_csv(file_path, dtype=str, low_memory=False)
    rows = []

    is_new_schema = any(column in native_df.columns for column in ["publication_year", "host_organization", "cited_by_count"])
    record_origin = "openalex_csv_newschema" if is_new_schema else "openalex_csv_oldschema"

    for _, row in native_df.iterrows():
        bp = clean_text(coalesce_value(row.get("page_start"), row.get("first_page")))
        ep = clean_text(coalesce_value(row.get("page_end"), row.get("last_page")))
        rec = empty_record()
        rec.update({
            "corpus": corpus,
            "base": base,
            "bucket": bucket,
            "source_file": file_path.name,
            "slice_label": slice_label,
            "record_origin": record_origin,
            "AU": clean_list_like(coalesce_value(row.get("authors_full"), row.get("authors"))),
            "TI": clean_text(row.get("title")),
            "SO": clean_text(coalesce_value(row.get("source"), row.get("host_organization"))),
            "PY": safe_int(coalesce_value(row.get("year"), row.get("publication_year"))),
            "DI": normalize_doi(row.get("doi")),
            "DE": clean_list_like(coalesce_value(row.get("keywords"), row.get("topics"))),
            "ID": clean_list_like(coalesce_value(row.get("topics"), row.get("sdgs"), row.get("concepts_top10"))),
            "AB": clean_text(row.get("abstract")),
            "CR": clean_text(coalesce_value(row.get("refs_count"), row.get("referenced_works_count"))),
            "LA": clean_text(row.get("language")),
            "DT": clean_text(coalesce_value(row.get("doc_subtype"), row.get("type"), row.get("type_crossref"))),
            "C1": clean_text(coalesce_value(row.get("affiliations"), row.get("institutions"), row.get("countries"))),
            "TC": safe_int(coalesce_value(row.get("cited_by"), row.get("cited_by_count"))),
            "VL": clean_text(row.get("volume")),
            "IS": clean_text(row.get("issue")),
            "BP": bp,
            "EP": ep,
            "publication_date": parse_publication_date(row.get("publication_date")),
            "month_hint": clean_text(coalesce_value(row.get("month"), slice_label)),
            "open_access": clean_text(coalesce_value(row.get("open_access"), row.get("oa_status"), row.get("is_oa"))),
            "funding": clean_text(coalesce_value(row.get("funding"), row.get("funders"))),
            "publisher": clean_text(row.get("publisher")),
        })
        rec["native_row_json"] = json_row(row.to_dict())
        rows.append(rec)

    return pd.DataFrame(rows), native_df


def parse_wos_tabdelim(file_path: Path, corpus: str, base: str, bucket: str, slice_label: str):
    native_df = pd.read_csv(file_path, sep="\t", dtype=str, low_memory=False)
    rows = []

    for _, row in native_df.iterrows():
        rec = empty_record()
        rec.update({
            "corpus": corpus,
            "base": base,
            "bucket": bucket,
            "source_file": file_path.name,
            "slice_label": slice_label,
            "record_origin": "wos_tabdelim",
            "AU": clean_list_like(coalesce_value(row.get("AU"), row.get("AF"))),
            "TI": clean_text(row.get("TI")),
            "SO": clean_text(row.get("SO")),
            "PY": safe_int(row.get("PY")),
            "DI": normalize_doi(row.get("DI")),
            "DE": clean_list_like(row.get("DE")),
            "ID": clean_list_like(row.get("ID")),
            "AB": clean_text(row.get("AB")),
            "CR": clean_text(row.get("CR")),
            "LA": clean_text(row.get("LA")),
            "DT": clean_text(row.get("DT")),
            "C1": clean_text(coalesce_value(row.get("C1"), row.get("C3"))),
            "TC": safe_int(row.get("TC")),
            "VL": clean_text(row.get("VL")),
            "IS": clean_text(row.get("IS")),
            "BP": clean_text(row.get("BP")),
            "EP": clean_text(row.get("EP")),
            "publication_date": parse_publication_date(coalesce_value(row.get("DA"), row.get("EA"), f"{row.get('PY', '')}-{row.get('PD', '')}")),
            "month_hint": clean_text(slice_label),
            "open_access": clean_text(row.get("OA")),
            "funding": clean_text(coalesce_value(row.get("FU"), row.get("FX"))),
            "publisher": clean_text(row.get("PU")),
        })
        rec["native_row_json"] = json_row(row.to_dict())
        rows.append(rec)

    return pd.DataFrame(rows), native_df


def parse_file(file_path: Path, corpus: str, base: str, bucket: str, slice_label: str):
    suffix = file_path.suffix.lower()
    if suffix == ".bib":
        return parse_bibtex_file(file_path, corpus, base, bucket, slice_label)
    if suffix == ".csv" and base == "SCOPUS":
        return parse_scopus_csv(file_path, corpus, base, bucket, slice_label)
    if suffix == ".csv" and base == "OPENALEX":
        return parse_openalex_csv(file_path, corpus, base, bucket, slice_label)
    if suffix == ".txt" and base == "WOS":
        return parse_wos_tabdelim(file_path, corpus, base, bucket, slice_label)
    raise ValueError(f"Formato/base nao suportado: {suffix} | {base} | {file_path.name}")
'''


CELL_10 = r'''
# =========================
# LEITURA DOS BRUTOS, UNIAO NATIVA E AUDITORIA DE SCHEMA
# =========================

stage_banner("LEITURA DOS BRUTOS")

raw_candidates = inventory_df[
    inventory_df["bucket"].isin(["historico_raw", "complemento_raw"])
    & inventory_df["ext"].isin(list(SUPPORTED_RAW_EXTENSIONS))
    & inventory_df["base"].isin(["SCOPUS", "WOS", "OPENALEX"])
].copy()

display(raw_candidates.groupby(["corpus", "bucket", "base", "ext"]).size().reset_index(name="n_files"))

all_canonical = []
native_buffers = defaultdict(list)
file_audit_rows = []
schema_rows = []

for corpus in CORPORA:
    corpus_files = raw_candidates[raw_candidates["corpus"] == corpus].sort_values(["bucket", "base", "file_name"]).reset_index(drop=True)
    log(f"[parse] iniciando {corpus} com {len(corpus_files)} arquivos", corpus=corpus)

    for idx, row in enumerate(corpus_files.to_dict("records"), start=1):
        file_path = Path(row["full_path"])
        try:
            canonical_df, native_df = parse_file(file_path, corpus, row["base"], row["bucket"], row["slice_label"])
            if not canonical_df.empty:
                all_canonical.append(canonical_df)
            if not native_df.empty:
                native_buffers[(corpus, row["base"])].append(native_df.assign(_source_file=file_path.name))

            schema_rows.append({
                "corpus": corpus,
                "bucket": row["bucket"],
                "base": row["base"],
                "file_name": file_path.name,
                "n_rows_native": len(native_df),
                "n_rows_canonical": len(canonical_df),
                "native_columns_json": json.dumps(list(native_df.columns), ensure_ascii=False),
            })
            file_audit_rows.append({
                "corpus": corpus,
                "bucket": row["bucket"],
                "base": row["base"],
                "file_name": file_path.name,
                "ext": row["ext"],
                "slice_label": row["slice_label"],
                "status": "ok",
                "n_rows_canonical": len(canonical_df),
                "n_rows_native": len(native_df),
            })
        except Exception as exc:
            file_audit_rows.append({
                "corpus": corpus,
                "bucket": row["bucket"],
                "base": row["base"],
                "file_name": file_path.name,
                "ext": row["ext"],
                "slice_label": row["slice_label"],
                "status": f"erro: {exc}",
                "n_rows_canonical": 0,
                "n_rows_native": 0,
            })
            log(f"[parse] ERRO em {file_path.name}: {exc}", corpus=corpus)

        if idx % FILE_PROGRESS_EVERY == 0 or idx == len(corpus_files):
            ok_count = sum(1 for item in file_audit_rows if item["corpus"] == corpus and item["status"] == "ok")
            log(f"[parse] {corpus} -> {idx}/{len(corpus_files)} arquivos | ok={ok_count}", corpus=corpus)

raw_df = pd.concat(all_canonical, ignore_index=True) if all_canonical else pd.DataFrame(columns=CANONICAL_COLS)
file_audit_df = pd.DataFrame(file_audit_rows)
schema_df = pd.DataFrame(schema_rows)

print("Total de registros harmonizados antes da deduplicacao:", len(raw_df))
display(raw_df.head(5))
display(file_audit_df.head(20))

for corpus in CORPORA:
    base_dir = corpus_output_dir(corpus)
    corpus_raw = raw_df[raw_df["corpus"] == corpus].copy()
    corpus_raw.to_csv(base_dir / "raw_harmonized_union.csv.gz", index=False, compression="gzip")
    file_audit_df[file_audit_df["corpus"] == corpus].to_csv(base_dir / "raw_file_parse_audit.csv", index=False)
    schema_df[schema_df["corpus"] == corpus].to_csv(base_dir / "raw_schema_audit.csv", index=False)

for (corpus, base), frames in native_buffers.items():
    native_union = pd.concat(frames, ignore_index=True)
    native_out = corpus_output_dir(corpus) / "native_unions" / f"{base.lower()}_native_union.csv.gz"
    native_union.to_csv(native_out, index=False, compression="gzip")
    log(f"Native union salvo: {native_out} | rows={len(native_union)}", corpus=corpus)

ml_schema = schema_df[
    (schema_df["corpus"] == "ML_Multimodal")
    & (schema_df["bucket"] == "complemento_raw")
    & (schema_df["base"] == "SCOPUS")
]
if not ml_schema.empty:
    observed_cols = set()
    for payload in ml_schema["native_columns_json"].tolist():
        observed_cols.update(json.loads(payload))
    expected_textual = {"Abstract", "Author Keywords", "Index Keywords", "Affiliations", "Publication Date"}
    missing_expected = sorted(expected_textual - observed_cols)
    if missing_expected:
        log(
            f"[ML_Multimodal] ALERTA: os CSVs holdout da Scopus nao trazem campos textuais esperados -> {missing_expected}. "
            "Isto nao e corrigivel so com parser; a reexportacao pode ser necessaria.",
            corpus="ML_Multimodal",
        )
'''


CELL_11 = r'''
# =========================
# ENRIQUECIMENTO TEMPORAL, CHAVES E PRIORIDADE
# =========================

stage_banner("ENRIQUECIMENTO TEMPORAL E CHAVES")

raw_df["publication_date"] = raw_df["publication_date"].map(parse_publication_date)
raw_df["publication_ts"] = pd.to_datetime(raw_df["publication_date"], errors="coerce")
raw_df["slice_month"] = raw_df["slice_label"].astype("string").str.extract(r"(20\d{2}-\d{2})", expand=False)
raw_df["slice_month_ts"] = pd.to_datetime(raw_df["slice_month"], format="%Y-%m", errors="coerce")

raw_df["period"] = "unknown"
raw_df["period_reason"] = "bucket_nao_esperado"

mask_hist = raw_df["bucket"].eq("historico_raw")
raw_df.loc[mask_hist, "period"] = "core"
raw_df.loc[mask_hist, "period_reason"] = "historico_raw_assumido_ate_2025_09_30"

mask_comp = raw_df["bucket"].eq("complemento_raw")
mask_comp_slice = mask_comp & raw_df["slice_month_ts"].notna()
raw_df.loc[mask_comp_slice & (raw_df["slice_month_ts"] < HOLDOUT_START), "period"] = "core"
raw_df.loc[mask_comp_slice & (raw_df["slice_month_ts"] < HOLDOUT_START), "period_reason"] = "complemento_raw_slice_pre_holdout"
raw_df.loc[mask_comp_slice & (raw_df["slice_month_ts"] >= HOLDOUT_START), "period"] = "holdout"
raw_df.loc[mask_comp_slice & (raw_df["slice_month_ts"] >= HOLDOUT_START), "period_reason"] = "complemento_raw_slice_holdout"

mask_comp_pubdate = mask_comp & raw_df["slice_month_ts"].isna() & raw_df["publication_ts"].notna()
raw_df.loc[mask_comp_pubdate & (raw_df["publication_ts"] <= CORE_END), "period"] = "core"
raw_df.loc[mask_comp_pubdate & (raw_df["publication_ts"] <= CORE_END), "period_reason"] = "publication_date_le_core_end"
raw_df.loc[mask_comp_pubdate & (raw_df["publication_ts"] > CORE_END), "period"] = "holdout"
raw_df.loc[mask_comp_pubdate & (raw_df["publication_ts"] > CORE_END), "period_reason"] = "publication_date_gt_core_end"

raw_df["PY_num"] = pd.to_numeric(raw_df["PY"], errors="coerce").astype("Int64")
mask_comp_year = mask_comp & raw_df["period"].eq("unknown") & raw_df["PY_num"].notna()
raw_df.loc[mask_comp_year & (raw_df["PY_num"] >= 2026), "period"] = "holdout"
raw_df.loc[mask_comp_year & (raw_df["PY_num"] >= 2026), "period_reason"] = "fallback_year_2026"
raw_df.loc[mask_comp_year & (raw_df["PY_num"] <= 2025), "period"] = "core"
raw_df.loc[mask_comp_year & (raw_df["PY_num"] <= 2025), "period_reason"] = "fallback_year_le_2025"

raw_df["DI_norm"] = raw_df["DI"].map(normalize_doi)
raw_df["TI_norm"] = raw_df["TI"].map(normalize_title)
raw_df["SO_norm"] = raw_df["SO"].map(normalize_source_title)
raw_df["TC_num"] = pd.to_numeric(raw_df["TC"], errors="coerce")
raw_df["row_completeness"] = completeness_score(raw_df)
raw_df["source_priority"] = raw_df["base"].map(BASE_PRIORITY).fillna(0).astype(int)
raw_df["title_year_key"] = np.where(
    raw_df["TI_norm"].notna() & raw_df["PY_num"].notna(),
    raw_df["TI_norm"] + "::" + raw_df["PY_num"].astype(str),
    pd.NA,
)
raw_df["first_author_key"] = (
    raw_df["AU"].fillna("").astype(str).str.split(";").str[0].str.lower().str.strip().replace("", pd.NA)
)

period_summary = raw_df.groupby(["corpus", "period", "base"]).size().reset_index(name="n_records")
period_reason_summary = raw_df.groupby(["corpus", "period_reason"]).size().reset_index(name="n_records")

display(period_summary)
display(period_reason_summary)

for corpus in CORPORA:
    base_dir = corpus_output_dir(corpus)
    raw_df[raw_df["corpus"] == corpus].to_csv(base_dir / "raw_harmonized_enriched.csv.gz", index=False, compression="gzip")
    period_summary[period_summary["corpus"] == corpus].to_csv(base_dir / "period_assignment_summary.csv", index=False)
    period_reason_summary[period_reason_summary["corpus"] == corpus].to_csv(base_dir / "period_assignment_reason_summary.csv", index=False)
    log(f"Resumo temporal salvo para {corpus}", corpus=corpus)
'''


CELL_12 = r'''
# =========================
# DEDUPLICACAO AUDITAVEL COM ENRIQUECIMENTO DE CAMPOS
# =========================

stage_banner("DEDUPLICACAO AUDITAVEL")

DEDUP_ENRICH_FIELDS = [
    "AU", "TI", "SO", "PY", "DI", "DE", "ID", "AB", "CR", "LA", "DT", "C1",
    "VL", "IS", "BP", "EP", "publication_date", "month_hint", "open_access",
    "funding", "publisher"
]
LOOKUP_BACKFILL_FIELDS = DEDUP_ENRICH_FIELDS + ["TC"]


def merge_pipe_text(values) -> str | pd._libs.missing.NAType:
    items = []
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if not text or text.lower() in {"nan", "<na>", "none"}:
            continue
        parts = [p.strip() for p in text.split("|") if p.strip()]
        items.extend(parts if parts else [text])
    items = list(dict.fromkeys(items))
    return " | ".join(items) if items else pd.NA


def initialize_provenance_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["provenance_bases"] = df["base"]
    df["provenance_buckets"] = df["bucket"]
    df["provenance_slice_labels"] = df["slice_label"]
    df["provenance_source_files"] = df["source_file"]
    df["provenance_record_origins"] = df["record_origin"]
    df["duplicate_group_size"] = 1
    df["dedup_rule"] = "singleton_raw"
    df["group_recovered_fields"] = pd.NA
    return df


def rank_group(grp: pd.DataFrame) -> pd.DataFrame:
    ranked = grp.copy()
    ranked["TC_num"] = pd.to_numeric(ranked["TC"], errors="coerce")
    ranked["tc_sort"] = ranked["TC_num"].fillna(-1)
    ranked = ranked.sort_values(
        ["source_priority", "row_completeness", "tc_sort", "base", "source_file"],
        ascending=[False, False, False, True, True],
        na_position="last",
    )
    return ranked


def first_present_value(series: pd.Series):
    if series is None or len(series) == 0:
        return pd.NA
    mask = present_mask(series)
    if not mask.any():
        return pd.NA
    return series[mask].iloc[0]


def build_survivor(grp: pd.DataFrame, rule: str, key_name: str, key_value: str):
    ranked = rank_group(grp)
    survivor = ranked.iloc[0].copy()
    recovered_fields = []

    for field in DEDUP_ENRICH_FIELDS:
        if field not in ranked.columns:
            continue
        if not present_mask(pd.Series([survivor.get(field)])).iloc[0]:
            candidate = first_present_value(ranked[field])
            if pd.notna(candidate):
                survivor[field] = candidate
                recovered_fields.append(field)

    if not present_mask(pd.Series([survivor.get("TC")])).iloc[0]:
        tc_candidates = pd.to_numeric(ranked["TC"], errors="coerce").dropna()
        if not tc_candidates.empty:
            survivor["TC"] = str(int(tc_candidates.max()))
            recovered_fields.append("TC")

    survivor["dedup_rule"] = rule
    survivor["duplicate_group_size"] = int(len(grp))
    survivor["duplicate_group_key"] = f"{key_name}::{key_value}"
    survivor["group_recovered_fields"] = "; ".join(sorted(set(recovered_fields))) if recovered_fields else pd.NA
    survivor["provenance_bases"] = merge_pipe_text(list(grp.get("provenance_bases", pd.Series(dtype="string"))) + grp["base"].tolist())
    survivor["provenance_buckets"] = merge_pipe_text(list(grp.get("provenance_buckets", pd.Series(dtype="string"))) + grp["bucket"].tolist())
    survivor["provenance_slice_labels"] = merge_pipe_text(list(grp.get("provenance_slice_labels", pd.Series(dtype="string"))) + grp["slice_label"].tolist())
    survivor["provenance_source_files"] = merge_pipe_text(list(grp.get("provenance_source_files", pd.Series(dtype="string"))) + grp["source_file"].tolist())
    survivor["provenance_record_origins"] = merge_pipe_text(list(grp.get("provenance_record_origins", pd.Series(dtype="string"))) + grp["record_origin"].tolist())
    survivor["n_provenance_bases"] = count_pipe_values(survivor["provenance_bases"])
    survivor["n_provenance_files"] = count_pipe_values(survivor["provenance_source_files"])

    audit = {
        "rule": rule,
        "group_key": f"{key_name}::{key_value}",
        "group_size": int(len(grp)),
        "kept_base": survivor.get("base"),
        "kept_source_file": survivor.get("source_file"),
        "recovered_fields": survivor.get("group_recovered_fields"),
    }
    return survivor, audit


def collapse_exact(df: pd.DataFrame, key_col: str, rule: str, corpus: str):
    with_key = df[df[key_col].notna()].copy()
    without_key = df[df[key_col].isna()].copy()
    survivors = []
    audits = []
    total_groups = int(with_key[key_col].nunique(dropna=True))

    if total_groups == 0:
        return df.copy(), []

    for idx, (key_value, grp) in enumerate(with_key.groupby(key_col, sort=False), start=1):
        survivor, audit = build_survivor(grp, rule, key_col, str(key_value))
        survivors.append(pd.DataFrame([survivor]))
        if len(grp) > 1:
            audit["corpus"] = corpus
            audits.append(audit)
        if idx % EXACT_GROUP_PROGRESS_EVERY == 0 or idx == total_groups:
            log(f"[exact] {corpus} | {rule} -> {idx}/{total_groups} grupos", corpus=corpus)

    collapsed = pd.concat(survivors + [without_key], ignore_index=True)
    return collapsed, audits


def fuzzy_collapse(df: pd.DataFrame, corpus: str):
    if not RUN_FUZZY_DEDUP:
        return df.copy(), [], []

    eligible = df[
        df["DI_norm"].isna()
        & df["TI_norm"].notna()
        & df["first_author_key"].notna()
        & df["PY_num"].notna()
    ].copy()
    ineligible = df.drop(index=eligible.index).copy()

    kept_frames = [ineligible]
    audits = []
    skipped = []
    grouped = list(eligible.groupby(["period", "PY_num", "first_author_key"], sort=False))

    def find(parent, item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(parent, a, b):
        ra, rb = find(parent, a), find(parent, b)
        if ra != rb:
            parent[rb] = ra

    total_groups = len(grouped)
    for idx, ((period, year, author), grp) in enumerate(grouped, start=1):
        group_size = len(grp)
        if group_size == 1:
            kept_frames.append(grp)
        elif group_size > FUZZY_MAX_GROUP_SIZE:
            kept_frames.append(grp)
            skipped.append({
                "corpus": corpus,
                "period": period,
                "PY_num": int(year),
                "first_author_key": author,
                "group_size": group_size,
                "reason": f"size_gt_{FUZZY_MAX_GROUP_SIZE}",
            })
        else:
            row_ids = grp.index.tolist()
            titles = grp["TI_norm"].to_dict()
            parent = {row_id: row_id for row_id in row_ids}

            for i in range(len(row_ids)):
                for j in range(i + 1, len(row_ids)):
                    left = row_ids[i]
                    right = row_ids[j]
                    score = fuzz.token_sort_ratio(str(titles[left]), str(titles[right]))
                    if score >= FUZZY_THRESHOLD:
                        union(parent, left, right)

            clusters = defaultdict(list)
            for row_id in row_ids:
                clusters[find(parent, row_id)].append(row_id)

            for cluster_rows in clusters.values():
                cluster_df = grp.loc[cluster_rows].copy()
                if len(cluster_df) == 1:
                    kept_frames.append(cluster_df)
                else:
                    cluster_key = f"{period}::{year}::{author}::{len(cluster_df)}"
                    survivor, audit = build_survivor(cluster_df, f"fuzzy_title_author_year_{FUZZY_THRESHOLD}", "fuzzy_cluster", cluster_key)
                    audits.append({"corpus": corpus, **audit})
                    kept_frames.append(pd.DataFrame([survivor]))

        if idx % FUZZY_GROUP_PROGRESS_EVERY == 0 or idx == total_groups:
            log(f"[fuzzy] {corpus} -> {idx}/{total_groups} grupos candidatos", corpus=corpus)

    collapsed = pd.concat(kept_frames, ignore_index=True)
    return collapsed, audits, skipped


def first_present_lookup(df: pd.DataFrame, key_col: str, value_col: str):
    if key_col not in df.columns or value_col not in df.columns:
        return pd.Series(dtype="object")
    tmp = df[[key_col, value_col]].copy()
    tmp = tmp[tmp[key_col].notna()]
    tmp = tmp[present_mask(tmp[value_col])]
    if tmp.empty:
        return pd.Series(dtype="object")
    tmp = tmp.drop_duplicates(subset=[key_col], keep="first")
    return tmp.set_index(key_col)[value_col]


def max_numeric_lookup(df: pd.DataFrame, key_col: str, value_col: str):
    if key_col not in df.columns or value_col not in df.columns:
        return pd.Series(dtype="float64")
    tmp = df[[key_col, value_col]].copy()
    tmp = tmp[tmp[key_col].notna()]
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    if tmp.empty:
        return pd.Series(dtype="float64")
    return tmp.groupby(key_col)[value_col].max()


def build_lookups(raw_corpus_df: pd.DataFrame):
    raw_sorted = raw_corpus_df.sort_values(
        ["source_priority", "row_completeness", "TC_num", "base", "source_file"],
        ascending=[False, False, False, True, True],
        na_position="last",
    ).copy()

    lookups = {"doi": {}, "ty": {}}
    for field in DEDUP_ENRICH_FIELDS:
        lookups["doi"][field] = first_present_lookup(raw_sorted, "DI_norm", field)
        lookups["ty"][field] = first_present_lookup(raw_sorted, "title_year_key", field)

    lookups["doi"]["TC"] = max_numeric_lookup(raw_sorted, "DI_norm", "TC")
    lookups["ty"]["TC"] = max_numeric_lookup(raw_sorted, "title_year_key", "TC")
    return lookups


def backfill_from_lookups(survivors_df: pd.DataFrame, lookups: dict, corpus: str):
    survivors = survivors_df.copy()
    audit_rows = []

    for field in LOOKUP_BACKFILL_FIELDS:
        if field not in survivors.columns:
            survivors[field] = pd.NA

        before_n = int(present_mask(survivors[field]).sum())

        if field == "TC":
            missing = ~present_mask(survivors[field])
            doi_map = survivors["DI_norm"].map(lookups["doi"].get("TC", pd.Series(dtype="float64")))
            ty_map = survivors["title_year_key"].map(lookups["ty"].get("TC", pd.Series(dtype="float64")))
            mapped = doi_map.fillna(ty_map)
            survivors.loc[missing & mapped.notna(), field] = mapped[missing & mapped.notna()].astype(int).astype(str)
        else:
            missing = ~present_mask(survivors[field])
            doi_map = survivors["DI_norm"].map(lookups["doi"].get(field, pd.Series(dtype="object")))
            survivors.loc[missing & present_mask(doi_map.astype("string")), field] = doi_map[missing & present_mask(doi_map.astype("string"))]
            missing = ~present_mask(survivors[field])
            ty_map = survivors["title_year_key"].map(lookups["ty"].get(field, pd.Series(dtype="object")))
            survivors.loc[missing & present_mask(ty_map.astype("string")), field] = ty_map[missing & present_mask(ty_map.astype("string"))]

        after_n = int(present_mask(survivors[field]).sum())
        audit_rows.append({
            "corpus": corpus,
            "field": field,
            "present_before": before_n,
            "present_after": after_n,
            "recovered_n": after_n - before_n,
            "recovered_pct_points": round(((after_n - before_n) / len(survivors) * 100), 2) if len(survivors) else 0.0,
        })

    return survivors, pd.DataFrame(audit_rows)


raw_df = initialize_provenance_columns(raw_df)
global_stage_rows = []
global_audit_rows = []
global_lookup_audit_rows = []

for corpus in CORPORA:
    stage_banner(f"DEDUP - {corpus}", corpus=corpus)
    base_dir = corpus_output_dir(corpus)
    corpus_raw = raw_df[raw_df["corpus"] == corpus].copy().reset_index(drop=True)
    corpus_raw["row_id"] = np.arange(len(corpus_raw))

    log(
        f"[complexidade] {corpus}: exact DOI/title-year ~ O(n log n); fuzzy ~ soma dos grupos O(g*k^2) com k<={FUZZY_MAX_GROUP_SIZE}. "
        "Bottleneck principal esperado: CPU + Drive I/O.",
        corpus=corpus,
    )

    exact_doi_df, doi_audits = collapse_exact(corpus_raw, "DI_norm", "exact_doi", corpus)
    exact_ty_df, ty_audits = collapse_exact(exact_doi_df, "title_year_key", "exact_title_year", corpus)
    fuzzy_df, fuzzy_audits, fuzzy_skipped = fuzzy_collapse(exact_ty_df, corpus)

    lookups = build_lookups(corpus_raw)
    enriched_df, lookup_audit_df = backfill_from_lookups(fuzzy_df, lookups, corpus)

    enriched_df["n_provenance_bases"] = enriched_df["provenance_bases"].map(count_pipe_values)
    enriched_df["n_provenance_files"] = enriched_df["provenance_source_files"].map(count_pipe_values)
    enriched_df = enriched_df.sort_values(["period", "PY_num", "TI"], na_position="last").reset_index(drop=True)

    stage_counts = pd.DataFrame([
        {"corpus": corpus, "stage": "raw_harmonized_input", "n_rows": len(corpus_raw)},
        {"corpus": corpus, "stage": "after_exact_doi", "n_rows": len(exact_doi_df)},
        {"corpus": corpus, "stage": "after_exact_title_year", "n_rows": len(exact_ty_df)},
        {"corpus": corpus, "stage": "after_fuzzy", "n_rows": len(fuzzy_df)},
        {"corpus": corpus, "stage": "after_lookup_backfill", "n_rows": len(enriched_df)},
    ])

    dedup_audit_df = pd.DataFrame(doi_audits + ty_audits + fuzzy_audits)
    fuzzy_skipped_df = pd.DataFrame(fuzzy_skipped)

    dedup_path = base_dir / "dedup_survivors.csv.gz"
    enriched_df.to_csv(dedup_path, index=False, compression="gzip")
    stage_counts.to_csv(base_dir / "dedup_stage_counts.csv", index=False)
    dedup_audit_df.to_csv(base_dir / "dedup_audit.csv", index=False)
    fuzzy_skipped_df.to_csv(base_dir / "dedup_fuzzy_skipped.csv", index=False)
    lookup_audit_df.to_csv(base_dir / "post_dedup_enrichment_audit.csv", index=False)

    final_cols_order = []
    preferred_front = CANONICAL_COLS + [
        "period", "period_reason", "DI_norm", "TI_norm", "SO_norm", "PY_num", "TC_num",
        "title_year_key", "first_author_key", "row_completeness", "source_priority",
        "dedup_rule", "duplicate_group_size", "duplicate_group_key",
        "provenance_bases", "provenance_buckets", "provenance_slice_labels",
        "provenance_source_files", "provenance_record_origins",
        "n_provenance_bases", "n_provenance_files", "group_recovered_fields",
    ]
    for column in preferred_front:
        if column in enriched_df.columns and column not in final_cols_order:
            final_cols_order.append(column)
    for column in enriched_df.columns:
        if column not in final_cols_order:
            final_cols_order.append(column)
    enriched_df = enriched_df[final_cols_order]

    for period in ["core", "holdout"]:
        out_df = enriched_df[enriched_df["period"] == period].copy()
        csv_path = base_dir / f"{corpus}_{period}_bibliometrix_clean.csv"
        out_df.to_csv(csv_path, index=False)
        log(f"CSV final salvo: {csv_path} | rows={len(out_df)}", corpus=corpus)

        if WRITE_RDATA:
            try:
                rdata_path = base_dir / f"{corpus}_{period}_bibliometrix_clean.RData"
                pyreadr.write_rdata(rdata_path, out_df, df_name="M")
                log(f"RData salvo: {rdata_path}", corpus=corpus)
            except Exception as exc:
                log(f"Falha ao salvar RData para {corpus}/{period}: {exc}", corpus=corpus)

    log(
        f"[dedup] {corpus} | raw={len(corpus_raw)} -> doi={len(exact_doi_df)} -> title_year={len(exact_ty_df)} -> fuzzy={len(fuzzy_df)} -> final={len(enriched_df)}",
        corpus=corpus,
    )

    if corpus == "ML_Multimodal":
        holdout_df = enriched_df[enriched_df["period"] == "holdout"].copy()
        log(
            f"[ML holdout] AB={present_pct(holdout_df.get('AB')):.2f}% | DE={present_pct(holdout_df.get('DE')):.2f}% | "
            f"ID={present_pct(holdout_df.get('ID')):.2f}% | C1={present_pct(holdout_df.get('C1')):.2f}% | "
            f"publication_date={present_pct(holdout_df.get('publication_date')):.2f}%",
            corpus=corpus,
        )

    global_stage_rows.extend(stage_counts.to_dict("records"))
    if not dedup_audit_df.empty:
        global_audit_rows.extend(dedup_audit_df.to_dict("records"))
    if not lookup_audit_df.empty:
        global_lookup_audit_rows.extend(lookup_audit_df.to_dict("records"))

global_stage_df = pd.DataFrame(global_stage_rows)
global_audit_df = pd.DataFrame(global_audit_rows)
global_lookup_audit_df = pd.DataFrame(global_lookup_audit_rows)

global_stage_df.to_csv(GLOBAL_LOG_DIR / f"00_dedup_stage_counts_{RUN_TS}.csv", index=False)
global_audit_df.to_csv(GLOBAL_LOG_DIR / f"00_dedup_audit_{RUN_TS}.csv", index=False)
global_lookup_audit_df.to_csv(GLOBAL_LOG_DIR / f"00_post_dedup_lookup_audit_{RUN_TS}.csv", index=False)

display(global_stage_df)
'''


CELL_13 = r'''
# =========================
# METRICAS DE COBERTURA, MISSINGNESS E SUMARIO DE ARTIGO
# =========================

stage_banner("METRICAS DE COBERTURA E MISSINGNESS")


def coverage_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    n_rows = len(df)
    for field in KEY_FIELDS:
        if field in df.columns:
            pct = present_pct(df[field])
            present_n = int(round((pct / 100) * n_rows))
        else:
            pct = 0.0
            present_n = 0
        rows.append({
            "scope": label,
            "field": field,
            "present_n": present_n,
            "present_pct": round(pct, 2),
            "missing_n": n_rows - present_n,
            "missing_pct": round(100 - pct, 2),
        })
    return pd.DataFrame(rows)


for corpus in CORPORA:
    base_dir = corpus_output_dir(corpus)
    core_path = base_dir / f"{corpus}_core_bibliometrix_clean.csv"
    holdout_path = base_dir / f"{corpus}_holdout_bibliometrix_clean.csv"

    core_df = safe_read_csv(core_path, dtype=str, low_memory=False)
    holdout_df = safe_read_csv(holdout_path, dtype=str, low_memory=False)
    combined = pd.concat([core_df.assign(period="core"), holdout_df.assign(period="holdout")], ignore_index=True)

    if combined.empty:
        log(f"[coverage] {corpus}: base vazia, pulando", corpus=corpus)
        continue

    combined["PY_num"] = pd.to_numeric(combined.get("PY"), errors="coerce")
    combined["TC_num"] = pd.to_numeric(combined.get("TC"), errors="coerce")

    multi_mask = combined.get("provenance_bases", pd.Series(index=combined.index, dtype="string")).map(count_pipe_values) >= 2
    single_mask = ~multi_mask

    coverage_all = pd.concat([
        coverage_table(combined, "overall"),
        coverage_table(combined[combined["period"] == "core"].copy(), "core"),
        coverage_table(combined[combined["period"] == "holdout"].copy(), "holdout"),
        coverage_table(combined[multi_mask].copy(), "multi_source"),
        coverage_table(combined[single_mask].copy(), "single_source"),
    ], ignore_index=True)

    summary_by_base = combined.groupby(["period", "base"]).size().reset_index(name="n_docs")
    summary_by_year = (
        combined.dropna(subset=["PY_num"])
        .groupby(["period", "PY_num"])
        .size()
        .reset_index(name="n_docs")
        .sort_values(["period", "PY_num"])
    )
    missingness = coverage_all[coverage_all["scope"] == "overall"].copy()
    record_origin_summary = combined.groupby(["period", "record_origin"]).size().reset_index(name="n_docs")

    coverage_all.to_csv(base_dir / "coverage_all_scopes.csv", index=False)
    summary_by_base.to_csv(base_dir / "summary_by_base.csv", index=False)
    summary_by_year.to_csv(base_dir / "summary_by_year.csv", index=False)
    missingness.to_csv(base_dir / "missingness_report.csv", index=False)
    record_origin_summary.to_csv(base_dir / "record_origin_summary.csv", index=False)

    with open(base_dir / "article_ready_summary.md", "w", encoding="utf-8") as fh:
        fh.write(f"# {corpus} - Camada 0\\n\\n")
        fh.write(f"- total_docs: {len(combined)}\\n")
        fh.write(f"- core_docs: {len(core_df)}\\n")
        fh.write(f"- holdout_docs: {len(holdout_df)}\\n")
        fh.write(f"- multi_source_ratio_pct: {round(float(multi_mask.mean() * 100), 2) if len(combined) else 0.0}\\n\\n")
        fh.write("## Cobertura por base\\n\\n")
        fh.write(summary_by_base.to_markdown(index=False))
        fh.write("\\n\\n## Evolucao por ano\\n\\n")
        fh.write(summary_by_year.to_markdown(index=False))
        fh.write("\\n\\n## Missingness overall\\n\\n")
        fh.write(missingness.to_markdown(index=False))
        fh.write("\\n\\n## Record origin\\n\\n")
        fh.write(record_origin_summary.to_markdown(index=False))

    log(
        f"[coverage] {corpus} | total={len(combined)} | core={len(core_df)} | holdout={len(holdout_df)} | "
        f"PY={present_pct(combined.get('PY')):.2f}% | SO={present_pct(combined.get('SO')):.2f}% | TC={present_pct(combined.get('TC')):.2f}%",
        corpus=corpus,
    )
    display({"corpus": corpus, "n_total": len(combined), "n_core": len(core_df), "n_holdout": len(holdout_df)})
    display(summary_by_base)
    display(missingness)
'''


CELL_14 = r'''
# =========================
# VALIDACAO FINAL E GRAFICOS DE SAUDE DA BASE
# =========================

stage_banner("VALIDACAO FINAL E GRAFICOS DE SAUDE")

all_flags = []
global_summary_rows = []

for corpus in CORPORA:
    base_dir = corpus_output_dir(corpus)
    validation_dir = ensure_dir(base_dir / "article_validation")

    core_path = base_dir / f"{corpus}_core_bibliometrix_clean.csv"
    holdout_path = base_dir / f"{corpus}_holdout_bibliometrix_clean.csv"

    core_df = safe_read_csv(core_path, dtype=str, low_memory=False)
    holdout_df = safe_read_csv(holdout_path, dtype=str, low_memory=False)
    combined = pd.concat([core_df.assign(period="core"), holdout_df.assign(period="holdout")], ignore_index=True)

    if combined.empty:
        log(f"[validation] {corpus}: base vazia, pulando", corpus=corpus)
        continue

    combined["PY_num"] = pd.to_numeric(combined.get("PY"), errors="coerce")
    combined["TC_num"] = pd.to_numeric(combined.get("TC"), errors="coerce")
    combined["month_ym"] = derive_month_series(combined)
    combined["SO_clean"] = combined.get("SO", pd.Series(index=combined.index, dtype="string")).astype("string").fillna("").str.strip()

    multi_source_mask = combined.get("provenance_bases", pd.Series(index=combined.index, dtype="string")).map(count_pipe_values) >= 2
    single_source_mask = ~multi_source_mask

    coverage_all = pd.concat([
        coverage_table(combined, "overall"),
        coverage_table(combined[combined["period"] == "core"].copy(), "core"),
        coverage_table(combined[combined["period"] == "holdout"].copy(), "holdout"),
        coverage_table(combined[multi_source_mask].copy(), "multi_source"),
        coverage_table(combined[single_source_mask].copy(), "single_source"),
    ], ignore_index=True)

    pubs_by_year = (
        combined.dropna(subset=["PY_num"])
        .groupby(["period", "PY_num"])
        .size()
        .reset_index(name="n_docs")
        .sort_values(["period", "PY_num"])
    )
    pubs_by_month = (
        combined.dropna(subset=["month_ym"])
        .groupby(["period", "month_ym"])
        .size()
        .reset_index(name="n_docs")
        .sort_values(["period", "month_ym"])
    )
    pubs_by_base = (
        combined.groupby(["period", "base"])
        .size()
        .reset_index(name="n_docs")
        .sort_values(["period", "n_docs"], ascending=[True, False])
    )
    citations_by_year = (
        combined.dropna(subset=["PY_num"])
        .groupby(["period", "PY_num"])["TC_num"]
        .agg(["count", "sum", "mean", "median", "max"])
        .reset_index()
        .rename(columns={"count": "n_docs", "sum": "tc_sum", "mean": "tc_mean", "median": "tc_median", "max": "tc_max"})
        .sort_values(["period", "PY_num"])
    )

    merge_diag = pd.DataFrame([
        {"metric": "rows_total", "value": len(combined)},
        {"metric": "rows_core", "value": len(core_df)},
        {"metric": "rows_holdout", "value": len(holdout_df)},
        {"metric": "rows_multi_source", "value": int(multi_source_mask.sum())},
        {"metric": "rows_single_source", "value": int(single_source_mask.sum())},
        {"metric": "multi_source_ratio_pct", "value": round(float(multi_source_mask.mean() * 100), 2)},
        {"metric": "year_present_pct_overall", "value": round(present_pct(combined.get("PY")), 2)},
        {"metric": "source_present_pct_overall", "value": round(present_pct(combined.get("SO")), 2)},
        {"metric": "citations_present_pct_overall", "value": round(present_pct(combined.get("TC")), 2)},
        {"metric": "abstract_present_pct_overall", "value": round(present_pct(combined.get("AB")), 2)},
        {"metric": "affiliation_present_pct_overall", "value": round(present_pct(combined.get("C1")), 2)},
        {"metric": "year_present_pct_multi_source", "value": round(present_pct(combined.loc[multi_source_mask, "PY"]) if multi_source_mask.any() and "PY" in combined.columns else 0.0, 2)},
        {"metric": "source_present_pct_multi_source", "value": round(present_pct(combined.loc[multi_source_mask, "SO"]) if multi_source_mask.any() and "SO" in combined.columns else 0.0, 2)},
        {"metric": "citations_present_pct_multi_source", "value": round(present_pct(combined.loc[multi_source_mask, "TC"]) if multi_source_mask.any() and "TC" in combined.columns else 0.0, 2)},
        {"metric": "abstract_present_pct_multi_source", "value": round(present_pct(combined.loc[multi_source_mask, "AB"]) if multi_source_mask.any() and "AB" in combined.columns else 0.0, 2)},
        {"metric": "affiliation_present_pct_multi_source", "value": round(present_pct(combined.loc[multi_source_mask, "C1"]) if multi_source_mask.any() and "C1" in combined.columns else 0.0, 2)},
    ])

    flags = []

    def add_flag(condition: bool, message: str, severity: str = "ALERTA"):
        if condition:
            flags.append({"corpus": corpus, "severity": severity, "message": message})
            all_flags.append({"corpus": corpus, "severity": severity, "message": message})

    year_pct = present_pct(combined.get("PY"))
    source_pct = present_pct(combined.get("SO"))
    tc_pct = present_pct(combined.get("TC"))
    abs_pct = present_pct(combined.get("AB"))
    multi_year_pct = present_pct(combined.loc[multi_source_mask, "PY"]) if multi_source_mask.any() and "PY" in combined.columns else 0.0
    multi_source_pct = present_pct(combined.loc[multi_source_mask, "SO"]) if multi_source_mask.any() and "SO" in combined.columns else 0.0
    multi_tc_pct = present_pct(combined.loc[multi_source_mask, "TC"]) if multi_source_mask.any() and "TC" in combined.columns else 0.0

    add_flag(year_pct < 95, f"cobertura de PY abaixo de 95% no corpus final ({year_pct:.2f}%).")
    add_flag(source_pct < 80, f"cobertura de SO abaixo de 80% no corpus final ({source_pct:.2f}%).")
    add_flag(tc_pct < 60, f"cobertura de TC abaixo de 60% no corpus final ({tc_pct:.2f}%).")
    add_flag(abs_pct < 70, f"cobertura de AB abaixo de 70% no corpus final ({abs_pct:.2f}%).", severity="ATENCAO")
    add_flag(multi_source_mask.any() and multi_year_pct < 95, f"registros multi_source continuam com baixa cobertura de PY ({multi_year_pct:.2f}%), sugerindo merge/metadado insuficiente.")
    add_flag(multi_source_mask.any() and multi_source_pct < 80, f"registros multi_source continuam com baixa cobertura de SO ({multi_source_pct:.2f}%), sugerindo merge/metadado insuficiente.")
    add_flag(multi_source_mask.any() and multi_tc_pct < 60, f"registros multi_source continuam com baixa cobertura de TC ({multi_tc_pct:.2f}%), sugerindo merge/metadado insuficiente.")
    add_flag(len(holdout_df) > len(core_df), f"holdout maior que core ({len(holdout_df)} vs {len(core_df)}). Revisar comparabilidade temporal e amplitude da query.", severity="ATENCAO")

    if corpus == "ML_Multimodal":
        holdout_text_block = all(present_pct(holdout_df.get(field)) == 0.0 for field in ["AB", "DE", "ID", "C1", "publication_date"])
        add_flag(holdout_text_block, "holdout de ML continua sem campos textuais essenciais (AB/DE/ID/C1/publication_date). Recoleta/reexportacao segue necessaria.", severity="ALERTA")

    flags_df = pd.DataFrame(flags) if flags else pd.DataFrame(columns=["corpus", "severity", "message"])

    merge_diag.to_csv(validation_dir / f"{corpus}_merge_diagnostics.csv", index=False)
    coverage_all.to_csv(validation_dir / f"{corpus}_coverage_all_scopes.csv", index=False)
    pubs_by_year.to_csv(validation_dir / f"{corpus}_publications_by_year.csv", index=False)
    pubs_by_month.to_csv(validation_dir / f"{corpus}_publications_by_month.csv", index=False)
    pubs_by_base.to_csv(validation_dir / f"{corpus}_publications_by_base_period.csv", index=False)
    citations_by_year.to_csv(validation_dir / f"{corpus}_citations_by_year.csv", index=False)
    flags_df.to_csv(validation_dir / f"{corpus}_validation_flags.csv", index=False)

    print("\n" + "=" * 96)
    print(f"VALIDACAO BIBLIOMETRICA FINAL - {corpus}")
    print("=" * 96)
    print(f"Total={len(combined)} | core={len(core_df)} | holdout={len(holdout_df)} | multi_source={int(multi_source_mask.sum())}")
    display(merge_diag)
    display(coverage_all[coverage_all["scope"].isin(["overall", "core", "holdout"])])
    if not flags_df.empty:
        display(flags_df)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    if not pubs_by_year.empty:
        sns.barplot(data=pubs_by_year, x="PY_num", y="n_docs", hue="period", ax=axes[0, 0])
        axes[0, 0].set_title(f"{corpus} - Publicacoes por ano")
        axes[0, 0].set_xlabel("Ano")
        axes[0, 0].set_ylabel("N. documentos")
        axes[0, 0].tick_params(axis="x", rotation=45)
    else:
        axes[0, 0].text(0.5, 0.5, "Sem dados", ha="center", va="center")

    if not citations_by_year.empty:
        sns.lineplot(data=citations_by_year, x="PY_num", y="tc_sum", hue="period", marker="o", ax=axes[0, 1])
        axes[0, 1].set_title(f"{corpus} - Citacoes totais por ano")
        axes[0, 1].set_xlabel("Ano")
        axes[0, 1].set_ylabel("Soma de citacoes")
        axes[0, 1].tick_params(axis="x", rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, "Sem dados", ha="center", va="center")

    if not citations_by_year.empty:
        sns.lineplot(data=citations_by_year, x="PY_num", y="tc_mean", hue="period", marker="o", ax=axes[1, 0])
        axes[1, 0].set_title(f"{corpus} - Media de citacoes por ano")
        axes[1, 0].set_xlabel("Ano")
        axes[1, 0].set_ylabel("Media de citacoes")
        axes[1, 0].tick_params(axis="x", rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, "Sem dados", ha="center", va="center")

    if not pubs_by_base.empty:
        sns.barplot(data=pubs_by_base, x="base", y="n_docs", hue="period", ax=axes[1, 1])
        axes[1, 1].set_title(f"{corpus} - Cobertura por base e periodo")
        axes[1, 1].set_xlabel("Base")
        axes[1, 1].set_ylabel("N. documentos")
    else:
        axes[1, 1].text(0.5, 0.5, "Sem dados", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(validation_dir / f"{corpus}_figure_validation_overview.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    if not pubs_by_month.empty:
        plt.figure(figsize=(16, 4))
        sns.barplot(data=pubs_by_month, x="month_ym", y="n_docs", hue="period")
        plt.title(f"{corpus} - Publicacoes por mes")
        plt.xlabel("Mes")
        plt.ylabel("N. documentos")
        plt.xticks(rotation=60)
        plt.tight_layout()
        plt.savefig(validation_dir / f"{corpus}_figure_publications_by_month.png", dpi=180, bbox_inches="tight")
        plt.show()
        plt.close()

    global_summary_rows.append({
        "corpus": corpus,
        "rows_total": len(combined),
        "rows_core": len(core_df),
        "rows_holdout": len(holdout_df),
        "multi_source_ratio_pct": round(float(multi_source_mask.mean() * 100), 2),
        "year_present_pct": round(year_pct, 2),
        "source_present_pct": round(source_pct, 2),
        "citations_present_pct": round(tc_pct, 2),
        "abstract_present_pct": round(abs_pct, 2),
        "n_flags": len(flags_df),
    })

global_summary_df = pd.DataFrame(global_summary_rows)
global_flags_df = pd.DataFrame(all_flags) if all_flags else pd.DataFrame(columns=["corpus", "severity", "message"])

global_summary_df.to_csv(GLOBAL_LOG_DIR / f"final_bibliometric_validation_summary_{RUN_TS}.csv", index=False)
global_flags_df.to_csv(GLOBAL_LOG_DIR / f"final_bibliometric_validation_flags_{RUN_TS}.csv", index=False)

print("\n" + "#" * 96)
print("RESUMO GLOBAL DA VALIDACAO BIBLIOMETRICA FINAL")
print("#" * 96)
display(global_summary_df)
if not global_flags_df.empty:
    display(global_flags_df)
'''


CELL_15 = r'''
# =========================
# PARIDADE VS CONSOLIDADO HISTORICO
# =========================

stage_banner("PARIDADE VS CONSOLIDADO HISTORICO")


def find_historical_csv(corpus: str):
    ref_dir = DATA_ROOT / corpus / "02_historico_consolidado"
    matches = sorted(ref_dir.glob("*_bibliometrix_clean.csv"))
    return matches[0] if matches else None


def historical_keyset(df: pd.DataFrame):
    doi_keys = set(df["DI_norm"].dropna().astype(str))
    title_year_keys = set(df["title_year_key"].dropna().astype(str))
    return doi_keys, title_year_keys


for corpus in CORPORA:
    hist_csv = find_historical_csv(corpus)
    core_csv = corpus_output_dir(corpus) / f"{corpus}_core_bibliometrix_clean.csv"
    log(f"[parity] iniciando {corpus}", corpus=corpus)

    if hist_csv is None or not core_csv.exists():
        log(f"[parity] pulado para {corpus} | hist_exists={hist_csv is not None} | core_exists={core_csv.exists()}", corpus=corpus)
        continue

    hist_df = pd.read_csv(hist_csv, dtype=str, low_memory=False)
    rebuilt_df = pd.read_csv(core_csv, dtype=str, low_memory=False)

    for df in (hist_df, rebuilt_df):
        df["DI_norm"] = df["DI"].map(normalize_doi) if "DI" in df.columns else pd.NA
        df["TI_norm"] = df["TI"].map(normalize_title) if "TI" in df.columns else pd.NA
        df["PY_num"] = pd.to_numeric(df["PY"], errors="coerce").astype("Int64") if "PY" in df.columns else pd.Series(dtype="Int64")
        df["title_year_key"] = np.where(
            df["TI_norm"].notna() & df["PY_num"].notna(),
            df["TI_norm"] + "::" + df["PY_num"].astype(str),
            pd.NA,
        )

    hist_doi, hist_ty = historical_keyset(hist_df)
    new_doi, new_ty = historical_keyset(rebuilt_df)

    parity = pd.DataFrame([
        {"metric": "historical_rows", "value": len(hist_df)},
        {"metric": "rebuilt_core_rows", "value": len(rebuilt_df)},
        {"metric": "delta_rows", "value": len(rebuilt_df) - len(hist_df)},
        {"metric": "rebuilt_vs_hist_row_ratio_pct", "value": round((len(rebuilt_df) / len(hist_df) * 100), 2) if len(hist_df) else np.nan},
        {"metric": "historical_unique_doi", "value": len(hist_doi)},
        {"metric": "rebuilt_unique_doi", "value": len(new_doi)},
        {"metric": "doi_overlap", "value": len(hist_doi & new_doi)},
        {"metric": "doi_overlap_pct_vs_hist", "value": round((len(hist_doi & new_doi) / len(hist_doi) * 100), 2) if len(hist_doi) else np.nan},
        {"metric": "historical_unique_title_year", "value": len(hist_ty)},
        {"metric": "rebuilt_unique_title_year", "value": len(new_ty)},
        {"metric": "title_year_overlap", "value": len(hist_ty & new_ty)},
        {"metric": "title_year_overlap_pct_vs_hist", "value": round((len(hist_ty & new_ty) / len(hist_ty) * 100), 2) if len(hist_ty) else np.nan},
    ])

    parity_out = corpus_output_dir(corpus) / "parity_check_vs_historico.csv"
    parity.to_csv(parity_out, index=False)
    log(f"[parity] salvo: {parity_out}", corpus=corpus)
    display(parity)
'''


CELL_16 = r'''
# =========================
# ANALISE BIBLIOMETRICA BASICA DAS BASES FINAIS
# =========================

stage_banner("ANALISE BIBLIOMETRICA BASICA")

for corpus in CORPORA:
    base_dir = corpus_output_dir(corpus)
    analytics_dir = ensure_dir(base_dir / "article_bibliometrics")

    core_df = safe_read_csv(base_dir / f"{corpus}_core_bibliometrix_clean.csv", dtype=str, low_memory=False)
    holdout_df = safe_read_csv(base_dir / f"{corpus}_holdout_bibliometrix_clean.csv", dtype=str, low_memory=False)
    combined = pd.concat([core_df.assign(period="core"), holdout_df.assign(period="holdout")], ignore_index=True)

    if combined.empty:
        log(f"[bibliometria] {corpus}: base vazia, pulando", corpus=corpus)
        continue

    combined["PY_num"] = pd.to_numeric(combined.get("PY"), errors="coerce")
    combined["TC_num"] = pd.to_numeric(combined.get("TC"), errors="coerce")
    combined["month_ym"] = derive_month_series(combined)
    combined["SO_norm"] = combined.get("SO", pd.Series(index=combined.index, dtype="string")).map(normalize_source_title)

    pubs_by_year = (
        combined.dropna(subset=["PY_num"])
        .groupby(["period", "PY_num"])
        .size()
        .reset_index(name="n_docs")
        .sort_values(["period", "PY_num"])
    )
    pubs_by_month = (
        combined.dropna(subset=["month_ym"])
        .groupby(["period", "month_ym"])
        .size()
        .reset_index(name="n_docs")
        .sort_values(["period", "month_ym"])
    )
    citations_by_year = (
        combined.dropna(subset=["PY_num"])
        .groupby(["period", "PY_num"])["TC_num"]
        .agg(["count", "sum", "mean", "median", "max"])
        .reset_index()
        .rename(columns={"count": "n_docs", "sum": "tc_sum", "mean": "tc_mean", "median": "tc_median", "max": "tc_max"})
        .sort_values(["period", "PY_num"])
    )

    source_tmp = combined[present_mask(combined.get("SO"))].copy()
    if not source_tmp.empty:
        source_tmp["SO_norm"] = source_tmp["SO"].map(normalize_source_title)
        source_label_map = (
            source_tmp.groupby(["SO_norm", "SO"]).size().reset_index(name="n_docs")
            .sort_values(["SO_norm", "n_docs", "SO"], ascending=[True, False, True])
            .drop_duplicates(subset=["SO_norm"], keep="first")
            .set_index("SO_norm")["SO"]
        )
        top_sources = (
            source_tmp.groupby("SO_norm").size().reset_index(name="n_docs")
            .sort_values("n_docs", ascending=False)
            .head(15)
        )
        top_sources["SO"] = top_sources["SO_norm"].map(source_label_map)
        top_sources = top_sources[["SO", "n_docs", "SO_norm"]]
    else:
        top_sources = pd.DataFrame(columns=["SO", "n_docs", "SO_norm"])

    kw_series = split_terms(
        combined.get("DE", pd.Series(index=combined.index, dtype="string")).fillna(
            combined.get("ID", pd.Series(index=combined.index, dtype="string"))
        )
    )
    top_keywords = kw_series.value_counts().reset_index()
    if not top_keywords.empty:
        top_keywords.columns = ["keyword", "n_docs"]
        top_keywords = top_keywords.head(20)

    keep_cols = [column for column in ["period", "base", "TI", "SO", "PY_num", "TC_num", "DI", "provenance_bases", "provenance_source_files"] if column in combined.columns]
    top_cited_docs = combined[keep_cols].copy().sort_values(["TC_num", "PY_num"], ascending=[False, False]).head(20) if keep_cols else pd.DataFrame()

    pubs_by_year.to_csv(analytics_dir / f"{corpus}_publications_by_year.csv", index=False)
    pubs_by_month.to_csv(analytics_dir / f"{corpus}_publications_by_month.csv", index=False)
    citations_by_year.to_csv(analytics_dir / f"{corpus}_citations_by_year.csv", index=False)
    top_sources.to_csv(analytics_dir / f"{corpus}_top_sources.csv", index=False)
    if not top_keywords.empty:
        top_keywords.to_csv(analytics_dir / f"{corpus}_top_keywords.csv", index=False)
    if not top_cited_docs.empty:
        top_cited_docs.to_csv(analytics_dir / f"{corpus}_top_cited_docs.csv", index=False)

    print("\n" + "=" * 96)
    print(f"ANALISE BIBLIOMETRICA BASICA - {corpus}")
    print("=" * 96)
    display(pubs_by_year.head(20))
    display(citations_by_year.head(20))
    if not top_sources.empty:
        display(top_sources.head(10))
    if not top_cited_docs.empty:
        display(top_cited_docs.head(10))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    if not pubs_by_year.empty:
        sns.barplot(data=pubs_by_year, x="PY_num", y="n_docs", hue="period", ax=axes[0, 0])
        axes[0, 0].set_title(f"{corpus} - Numero de artigos por ano")
        axes[0, 0].set_xlabel("Ano")
        axes[0, 0].set_ylabel("N. artigos")
        axes[0, 0].tick_params(axis="x", rotation=45)
    else:
        axes[0, 0].text(0.5, 0.5, "Sem dados", ha="center", va="center")

    if not citations_by_year.empty:
        sns.lineplot(data=citations_by_year, x="PY_num", y="tc_sum", hue="period", marker="o", ax=axes[0, 1])
        axes[0, 1].set_title(f"{corpus} - Citacoes totais por ano")
        axes[0, 1].set_xlabel("Ano")
        axes[0, 1].set_ylabel("Soma de citacoes")
        axes[0, 1].tick_params(axis="x", rotation=45)
    else:
        axes[0, 1].text(0.5, 0.5, "Sem dados", ha="center", va="center")

    if not citations_by_year.empty:
        sns.lineplot(data=citations_by_year, x="PY_num", y="tc_mean", hue="period", marker="o", ax=axes[1, 0])
        axes[1, 0].set_title(f"{corpus} - Media de citacoes por ano")
        axes[1, 0].set_xlabel("Ano")
        axes[1, 0].set_ylabel("Media de citacoes")
        axes[1, 0].tick_params(axis="x", rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, "Sem dados", ha="center", va="center")

    if not top_sources.empty:
        sns.barplot(data=top_sources.head(12), y="SO", x="n_docs", ax=axes[1, 1], color="#4C78A8")
        axes[1, 1].set_title(f"{corpus} - Top fontes/periódicos")
        axes[1, 1].set_xlabel("N. artigos")
        axes[1, 1].set_ylabel("Fonte")
    else:
        axes[1, 1].text(0.5, 0.5, "Sem dados", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(analytics_dir / f"{corpus}_figure_bibliometric_overview.png", dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    if not pubs_by_month.empty:
        plt.figure(figsize=(16, 4))
        sns.barplot(data=pubs_by_month, x="month_ym", y="n_docs", hue="period")
        plt.title(f"{corpus} - Evolucao mensal de publicacoes")
        plt.xlabel("Mes")
        plt.ylabel("N. artigos")
        plt.xticks(rotation=60)
        plt.tight_layout()
        plt.savefig(analytics_dir / f"{corpus}_figure_publications_by_month.png", dpi=180, bbox_inches="tight")
        plt.show()
        plt.close()

log("Analise bibliometrica basica concluida para todos os corpora.")
'''


CELL_17 = r'''
# =========================
# PAINEL EXECUTIVO FINAL DA CAMADA 0
# =========================

stage_banner("PAINEL EXECUTIVO FINAL")

executive_rows = []
artifact_rows = []
validation_rows = []

for corpus in CORPORA:
    base_dir = corpus_output_dir(corpus)

    stage_counts = safe_read_csv(base_dir / "dedup_stage_counts.csv")
    core_df = safe_read_csv(base_dir / f"{corpus}_core_bibliometrix_clean.csv", dtype=str, low_memory=False)
    holdout_df = safe_read_csv(base_dir / f"{corpus}_holdout_bibliometrix_clean.csv", dtype=str, low_memory=False)
    coverage_all = safe_read_csv(base_dir / "coverage_all_scopes.csv")
    parity_df = safe_read_csv(base_dir / "parity_check_vs_historico.csv")
    flags_df = safe_read_csv(base_dir / "article_validation" / f"{corpus}_validation_flags.csv")
    lookup_audit_df = safe_read_csv(base_dir / "post_dedup_enrichment_audit.csv")

    combined = pd.concat([core_df.assign(period="core"), holdout_df.assign(period="holdout")], ignore_index=True)
    multi_source_ratio_pct = round(float((combined.get("provenance_bases", pd.Series(index=combined.index, dtype="string")).map(count_pipe_values) >= 2).mean() * 100), 2) if len(combined) else 0.0

    def get_coverage(field: str):
        if coverage_all.empty:
            return np.nan
        subset = coverage_all[(coverage_all["scope"] == "overall") & (coverage_all["field"] == field)]
        return float(subset["present_pct"].iloc[0]) if not subset.empty else np.nan

    def get_metric(metric: str):
        if parity_df.empty:
            return np.nan
        subset = parity_df[parity_df["metric"] == metric]
        return float(subset["value"].iloc[0]) if not subset.empty else np.nan

    executive_rows.append({
        "corpus": corpus,
        "rows_total": len(combined),
        "rows_core": len(core_df),
        "rows_holdout": len(holdout_df),
        "multi_source_ratio_pct": multi_source_ratio_pct,
        "year_present_pct": get_coverage("PY"),
        "source_present_pct": get_coverage("SO"),
        "citations_present_pct": get_coverage("TC"),
        "abstract_present_pct": get_coverage("AB"),
        "affiliation_present_pct": get_coverage("C1"),
        "doi_overlap_pct_vs_hist": get_metric("doi_overlap_pct_vs_hist"),
        "title_year_overlap_pct_vs_hist": get_metric("title_year_overlap_pct_vs_hist"),
        "lookup_recovered_fields": int(lookup_audit_df["recovered_n"].sum()) if not lookup_audit_df.empty else 0,
        "n_validation_flags": len(flags_df),
    })

    major_artifacts = [
        base_dir / "dedup_survivors.csv.gz",
        base_dir / f"{corpus}_core_bibliometrix_clean.csv",
        base_dir / f"{corpus}_holdout_bibliometrix_clean.csv",
        base_dir / "coverage_all_scopes.csv",
        base_dir / "parity_check_vs_historico.csv",
        base_dir / "article_validation" / f"{corpus}_merge_diagnostics.csv",
        base_dir / "article_bibliometrics" / f"{corpus}_figure_bibliometric_overview.png",
    ]
    for artifact in major_artifacts:
        artifact_rows.append({
            "corpus": corpus,
            "artifact": str(artifact),
            "exists": artifact.exists(),
        })

    if not flags_df.empty:
        validation_rows.extend(flags_df.to_dict("records"))

executive_df = pd.DataFrame(executive_rows)
artifact_df = pd.DataFrame(artifact_rows)
validation_df = pd.DataFrame(validation_rows) if validation_rows else pd.DataFrame(columns=["corpus", "severity", "message"])

executive_path = GLOBAL_LOG_DIR / f"00_executive_summary_{RUN_TS}.csv"
artifact_path = GLOBAL_LOG_DIR / f"00_artifact_checklist_{RUN_TS}.csv"
flags_path = GLOBAL_LOG_DIR / f"00_validation_flags_{RUN_TS}.csv"

executive_df.to_csv(executive_path, index=False)
artifact_df.to_csv(artifact_path, index=False)
validation_df.to_csv(flags_path, index=False)

print("\n" + "#" * 96)
print("RESUMO EXECUTIVO FINAL DA CAMADA 0")
print("#" * 96)
display(executive_df)
display(artifact_df.head(20))
if not validation_df.empty:
    display(validation_df)

print("Arquivos globais salvos em:")
print("-", executive_path)
print("-", artifact_path)
print("-", flags_path)
'''


CELL_18 = r'''
## Próximo passo após este notebook

Só avance para os próximos pipes quando os alertas críticos de `PY`, `SO`, `TC` e do `ML_holdout` estiverem resolvidos ou explicitamente aceitos metodologicamente.

1. congelar os `*_core_bibliometrix_clean.csv` e `*_holdout_bibliometrix_clean.csv`;
2. usar esses outputs como entrada dos novos `Abstract_LLM_*`;
3. manter os relatórios de paridade, cobertura e validação para justificar comparabilidade com o paper original;
4. se `ML_Multimodal_holdout` continuar sem `AB/DE/ID/C1/publication_date`, fazer reexportação/recoleta antes dos pipes textuais.

### Lembrete operacional

- este notebook: **CPU / High-RAM**
- `Abstract_LLM_*`: **GPU**
- `SciBERT_SolarPhysics_Search`: **GPU**
- `PIPE_4`: **CPU**
'''


def main() -> None:
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Notebook nao encontrado: {NOTEBOOK_PATH}")

    backup_path = NOTEBOOK_PATH.parent / f"{NOTEBOOK_PATH.stem}.pre_rewrite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
    original_text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    backup_path.write_text(original_text, encoding="utf-8")

    nb = json.loads(original_text)
    preserved_prefix = nb["cells"][:4]
    replacement_cells = [
        code_cell(CELL_4),
        code_cell(CELL_5),
        code_cell(CELL_6),
        code_cell(CELL_7),
        code_cell(CELL_8),
        code_cell(CELL_9),
        code_cell(CELL_10),
        code_cell(CELL_11),
        code_cell(CELL_12),
        code_cell(CELL_13),
        code_cell(CELL_14),
        code_cell(CELL_15),
        code_cell(CELL_16),
        code_cell(CELL_17),
        md_cell(CELL_18),
    ]
    nb["cells"] = preserved_prefix + replacement_cells

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Notebook reescrito com sucesso:", NOTEBOOK_PATH)
    print("Backup salvo em:", backup_path)


if __name__ == "__main__":
    main()
