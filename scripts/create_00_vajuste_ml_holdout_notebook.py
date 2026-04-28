from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_DIR = Path(
    r"C:\Users\andre\odrive\Google Drive\Unicamp\artigo bibliometria\grounded-scientometrics-solarphysics-retrieval\notebooks"
)
CANONICAL_NOTEBOOK = NOTEBOOK_DIR / "00_consolidacao_rebuild_core_holdout.ipynb"
OUTPUT_NOTEBOOK = NOTEBOOK_DIR / "00_vajuste_ml_holdout_refresh.ipynb"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").splitlines()],
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").splitlines()],
    }


def load_canonical() -> dict:
    return json.loads(CANONICAL_NOTEBOOK.read_text(encoding="utf-8"))


def get_code_source(nb: dict, idx: int) -> str:
    return "".join(nb["cells"][idx]["source"]).strip("\n")


def build_config_cell() -> str:
    return """
# =========================
# CONFIGURACAO GERAL DO AJUSTE ML HOLDOUT
# =========================

DRIVE_ROOT = Path("/content/drive/MyDrive/Unicamp")
PROJECT_ROOT = DRIVE_ROOT / "artigo bibliometria" / "grounded-scientometrics-solarphysics-retrieval"
DATA_ROOT = DRIVE_ROOT / "artigo bibliometria" / "base de dados" / "Artigo_Bibliometria Base Bruta" / "BASES_UNIFICADAS_POR_TEMA"

TARGET_CORPUS = "ML_Multimodal"
PROCESS_CORPORA = [TARGET_CORPUS]
ALL_CORPORA = ["Nucleo", "PIML", "CombFinal", TARGET_CORPUS]

OUTPUT_STAGE_NAME = "00_vajuste_ml_holdout_refresh"
CANONICAL_STAGE_NAME = "00_consolidacao"

RESET_CORPUS_OUTPUTS_ON_START = True
KEEP_GLOBAL_RUN_LOG_HISTORY = True
WRITE_RDATA = True
PUBLISH_TO_CANONICAL = True
REQUIRE_NONZERO_CRITICAL_FIELDS = True

CORE_END = pd.Timestamp("2025-09-30")
HOLDOUT_START = pd.Timestamp("2025-10-01")
HOLDOUT_END = pd.Timestamp("2026-03-31")

CORPORA = PROCESS_CORPORA[:]
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

EXPECTED_HOLDOUT_SCOPUS_GLOBS = [
    "Scopus_Flares_ML_Multimodal_2025-10*.csv",
    "Scopus_Flares_ML_Multimodal_2025-11*.csv",
    "Scopus_Flares_ML_Multimodal_2025-12*.csv",
    "Scopus_Flares_ML_Multimodal_2026-01*.csv",
    "Scopus_Flares_ML_Multimodal_2026-02*.csv",
    "Scopus_Flares_ML_Multimodal_2026-03*.csv",
]
EXPECTED_CRITICAL_HOLDOUT_FIELDS = ["AB", "DE", "ID", "C1", "publication_date"]

assert PROJECT_ROOT.exists(), f"PROJECT_ROOT nao encontrado: {PROJECT_ROOT}"
assert DATA_ROOT.exists(), f"DATA_ROOT nao encontrado: {DATA_ROOT}"

print("PROJECT_ROOT =", PROJECT_ROOT)
print("DATA_ROOT    =", DATA_ROOT)
print("TARGET_CORPUS =", TARGET_CORPUS)
print("OUTPUT_STAGE_NAME =", OUTPUT_STAGE_NAME)
print("CANONICAL_STAGE_NAME =", CANONICAL_STAGE_NAME)
print("CORE_END     =", CORE_END.date())
print("HOLDOUT      =", HOLDOUT_START.date(), "->", HOLDOUT_END.date())
print("RESET_CORPUS_OUTPUTS_ON_START =", RESET_CORPUS_OUTPUTS_ON_START)
print("WRITE_RDATA                  =", WRITE_RDATA)
print("PUBLISH_TO_CANONICAL         =", PUBLISH_TO_CANONICAL)
print("REQUIRE_NONZERO_CRITICAL_FIELDS =", REQUIRE_NONZERO_CRITICAL_FIELDS)
print("Fuzzy dedup                  =", RUN_FUZZY_DEDUP, "| threshold =", FUZZY_THRESHOLD, "| max_group =", FUZZY_MAX_GROUP_SIZE)
print("Observabilidade              =", {"file_every": FILE_PROGRESS_EVERY, "exact_group_every": EXACT_GROUP_PROGRESS_EVERY, "fuzzy_group_every": FUZZY_GROUP_PROGRESS_EVERY})
""".strip(
        "\n"
    )


def build_outputs_cell() -> str:
    return """
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
    return DATA_ROOT / corpus / "04_rebuild_outputs" / OUTPUT_STAGE_NAME


def canonical_output_dir(corpus: str) -> Path:
    return DATA_ROOT / corpus / "04_rebuild_outputs" / CANONICAL_STAGE_NAME


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
GLOBAL_LOG_FILE = GLOBAL_LOG_DIR / f"00_vajuste_ml_holdout_{RUN_TS}.txt"
CORPUS_LOG_FILES = {}


def log(message: str, corpus: str | None = None) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{now} | +{fmt_seconds(elapsed_seconds())}]"
    line = f"{prefix} {message}"
    print(line, flush=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\\n")
    if corpus is not None and corpus in CORPUS_LOG_FILES:
        with open(CORPUS_LOG_FILES[corpus], "a", encoding="utf-8") as fh:
            fh.write(line + "\\n")


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
    CORPUS_LOG_FILES[corpus] = base_dir / "logs" / f"{corpus}_00_vajuste_ml_holdout_{RUN_TS}.txt"
    log(f"Output {reset_mode} pronto para {corpus}: {base_dir}", corpus=corpus)

print("GLOBAL_LOG_FILE =", GLOBAL_LOG_FILE)
print("ML holdout bruto esperado em:", DATA_ROOT / TARGET_CORPUS / "03_complemento_bruto_2025-09_2026-03" / "Scopus")
""".strip(
        "\n"
    )


def build_precheck_cell() -> str:
    return """
# =========================
# PRECHECK DOS NOVOS ARQUIVOS DO ML HOLDOUT
# =========================

stage_banner("PRECHECK DOS NOVOS CSVs DO ML HOLDOUT", corpus=TARGET_CORPUS)

scopus_dir = DATA_ROOT / TARGET_CORPUS / "03_complemento_bruto_2025-09_2026-03" / "Scopus"
assert scopus_dir.exists(), f"Pasta de Scopus nao encontrada: {scopus_dir}"

coverage_rows = []
for pattern in EXPECTED_HOLDOUT_SCOPUS_GLOBS:
    matches = sorted(scopus_dir.glob(pattern))
    coverage_rows.append({"pattern": pattern, "n_matches": len(matches), "files": " | ".join(p.name for p in matches)})

coverage_df = pd.DataFrame(coverage_rows)
display(coverage_df)

missing_patterns = coverage_df[coverage_df["n_matches"] == 0]
if not missing_patterns.empty:
    raise FileNotFoundError(
        "Faltam slices mensais do novo ML holdout. Padroes sem arquivo: "
        + ", ".join(missing_patterns["pattern"].tolist())
    )

sample_rows = []
for pattern in EXPECTED_HOLDOUT_SCOPUS_GLOBS:
    for file_path in sorted(scopus_dir.glob(pattern)):
        sample_df = pd.read_csv(file_path, nrows=3, dtype=str, low_memory=False)
        cols = list(sample_df.columns)
        has_explicit_pubdate = ("Publication Date" in cols) or ("Cover Date" in cols)
        sample_rows.append({
            "file_name": file_path.name,
            "n_columns": len(cols),
            "has_Abstract": "Abstract" in cols,
            "has_Author_Keywords": "Author Keywords" in cols,
            "has_Index_Keywords": "Index Keywords" in cols,
            "has_Affiliations": "Affiliations" in cols,
            "has_explicit_Publication_Date": has_explicit_pubdate,
            "slice_label_proxy_available": bool(parse_slice_from_name(file_path.name)),
            "columns_preview": " | ".join(cols[:18]),
        })

sample_schema_df = pd.DataFrame(sample_rows).sort_values("file_name").reset_index(drop=True)
display(sample_schema_df)

for column_name in ["has_Abstract", "has_Author_Keywords", "has_Index_Keywords", "has_Affiliations"]:
    if not bool(sample_schema_df[column_name].all()):
        bad_files = sample_schema_df.loc[~sample_schema_df[column_name], "file_name"].tolist()
        raise ValueError(f"Arquivos ainda sem campo critico ({column_name}): {bad_files}")

if not bool(sample_schema_df["has_explicit_Publication_Date"].all()):
    bad_files = sample_schema_df.loc[~sample_schema_df["has_explicit_Publication_Date"], "file_name"].tolist()
    proxy_ok = bool(sample_schema_df["slice_label_proxy_available"].all())
    if not proxy_ok:
        raise ValueError(
            "Arquivos sem Publication Date/Cover Date e sem slice_label utilizavel para proxy: "
            + str(bad_files)
        )
    log(
        "Scopus nao trouxe Publication Date explicita; o ajuste vai derivar publication_date_proxy = YYYY-MM-01 a partir do slice mensal.",
        corpus=TARGET_CORPUS,
    )

log("Precheck do novo ML holdout passou. CSVs parecem ricos em metadados.", corpus=TARGET_CORPUS)
""".strip(
        "\n"
    )


def build_scopus_override_cell() -> str:
    return """
# =========================
# OVERRIDE DO PARSER SCOPUS PARA O AJUSTE ML
# =========================

stage_banner("OVERRIDE PARSER SCOPUS PARA ML", corpus=TARGET_CORPUS)


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

    canonical_df = pd.DataFrame(rows)
    return canonical_df, native_df


log(
    "parse_scopus_csv override ativo: usa Publication Date, depois Cover Date, depois proxy YYYY-MM-01 derivada do slice_label.",
    corpus=TARGET_CORPUS,
)
""".strip(
        "\n"
    )


def build_publish_cell() -> str:
    return """
# =========================
# PUBLICACAO DO AJUSTE DO ML NO CANONICO
# =========================

stage_banner("PUBLICACAO DO AJUSTE ML NO CANONICO", corpus=TARGET_CORPUS)

src_dir = corpus_output_dir(TARGET_CORPUS)
dst_dir = canonical_output_dir(TARGET_CORPUS)
ensure_dir(dst_dir)

holdout_csv = src_dir / f"{TARGET_CORPUS}_holdout_bibliometrix_clean.csv"
core_csv = src_dir / f"{TARGET_CORPUS}_core_bibliometrix_clean.csv"
assert holdout_csv.exists(), f"Output holdout nao encontrado: {holdout_csv}"
assert core_csv.exists(), f"Output core nao encontrado: {core_csv}"

holdout_df = pd.read_csv(holdout_csv, dtype=str, low_memory=False)
critical_rows = []
for field in EXPECTED_CRITICAL_HOLDOUT_FIELDS:
    critical_rows.append({
        "field": field,
        "present_pct": round(present_pct(holdout_df.get(field)), 2),
        "present_n": int(present_mask(holdout_df.get(field)).sum()) if field in holdout_df.columns else 0,
        "n_rows": len(holdout_df),
    })

critical_df = pd.DataFrame(critical_rows)
display(critical_df)

if REQUIRE_NONZERO_CRITICAL_FIELDS:
    zero_fields = critical_df.loc[critical_df["present_n"] == 0, "field"].tolist()
    if zero_fields:
        raise RuntimeError(
            "Publicacao bloqueada: ML holdout ainda tem campos criticos zerados: "
            + ", ".join(zero_fields)
        )

backup_dir = ensure_dir(dst_dir / f"backup_pre_vajuste_{RUN_TS}")

publish_files = [
    src_dir / "raw_inventory.csv",
    src_dir / "file_parse_audit.csv",
    src_dir / "schema_audit.csv",
    src_dir / "raw_harmonized_enriched.csv.gz",
    src_dir / "dedup_survivors.csv.gz",
    src_dir / "dedup_stage_counts.csv",
    src_dir / "dedup_audit.csv",
    src_dir / "dedup_fuzzy_skipped.csv",
    src_dir / "post_dedup_enrichment_audit.csv",
    src_dir / f"{TARGET_CORPUS}_core_bibliometrix_clean.csv",
    src_dir / f"{TARGET_CORPUS}_holdout_bibliometrix_clean.csv",
    src_dir / f"{TARGET_CORPUS}_core_bibliometrix_clean.RData",
    src_dir / f"{TARGET_CORPUS}_holdout_bibliometrix_clean.RData",
]

for file_path in publish_files:
    if not file_path.exists():
        continue
    dst_file = dst_dir / file_path.name
    if dst_file.exists():
        shutil.copy2(dst_file, backup_dir / file_path.name)
    shutil.copy2(file_path, dst_file)
    log(f"Publicado no canonico: {dst_file}", corpus=TARGET_CORPUS)

for folder_name in ["native_unions", "logs"]:
    src_folder = src_dir / folder_name
    if src_folder.exists():
        dst_folder = dst_dir / folder_name
        if dst_folder.exists():
            archived_folder = backup_dir / folder_name
            if archived_folder.exists():
                shutil.rmtree(archived_folder)
            shutil.move(str(dst_folder), str(archived_folder))
        shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
        log(f"Pasta publicada no canonico: {dst_folder}", corpus=TARGET_CORPUS)

print("Backup canonico salvo em:", backup_dir)
""".strip(
        "\n"
    )


def build_switch_cell() -> str:
    return """
# =========================
# TROCA PARA O CANONICO E RECOMPOSICAO GLOBAL
# =========================

CORPORA = ALL_CORPORA[:]


def corpus_output_dir(corpus: str) -> Path:
    return canonical_output_dir(corpus)


stage_banner("RECOMPOSICAO GLOBAL POS-AJUSTE")
print("CORPORA para recomposicao global =", CORPORA)
print("Agora as etapas finais leem o canonico em:", CANONICAL_STAGE_NAME)
""".strip(
        "\n"
    )


def build_notebook() -> dict:
    canonical = load_canonical()
    cells = []

    cells.append(
        md_cell(
            """
# 00_vajuste - ML Holdout Refresh

Notebook incremental para reprocessar apenas `ML_Multimodal` depois da recoleta do `holdout`.

## Onde colocar os novos arquivos

Substitua ou adicione os novos CSVs ricos da Scopus em:

`/content/drive/MyDrive/Unicamp/artigo bibliometria/base de dados/Artigo_Bibliometria Base Bruta/BASES_UNIFICADAS_POR_TEMA/ML_Multimodal/03_complemento_bruto_2025-09_2026-03/Scopus`

No Windows local, isso corresponde a:

`C:\\Users\\andre\\odrive\\Google Drive\\Unicamp\\artigo bibliometria\\base de dados\\Artigo_Bibliometria Base Bruta\\BASES_UNIFICADAS_POR_TEMA\\ML_Multimodal\\03_complemento_bruto_2025-09_2026-03\\Scopus`

Use os nomes canônicos:

- `Scopus_Flares_ML_Multimodal_2025-10.csv`
- `Scopus_Flares_ML_Multimodal_2025-11.csv`
- `Scopus_Flares_ML_Multimodal_2025-12.csv`
- `Scopus_Flares_ML_Multimodal_2026-01.csv`
- `Scopus_Flares_ML_Multimodal_2026-02.csv`
- `Scopus_Flares_ML_Multimodal_2026-03.csv`

Se houver particionamento por limite de exportação, use sufixos como `_part01`, `_part02`.

## O que este notebook faz

1. valida se os novos CSVs do `ML_holdout` têm `Abstract`, `Author Keywords`, `Index Keywords`, `Affiliations` e `Publication Date`;
2. reroda apenas o pipeline do corpus `ML_Multimodal`;
3. publica os artefatos atualizados do `ML` de volta no `00_consolidacao` canônico;
4. recompõe as métricas finais globais usando `Nucleo`, `PIML` e `CombFinal` já congelados.
"""
        )
    )

    cells.append(code_cell(get_code_source(canonical, 3)))
    cells.append(code_cell(get_code_source(canonical, 4)))
    cells.append(code_cell(build_config_cell()))
    cells.append(code_cell(build_outputs_cell()))
    cells.append(code_cell(build_precheck_cell()))
    cells.append(code_cell(get_code_source(canonical, 7)))
    cells.append(code_cell(get_code_source(canonical, 8)))
    cells.append(code_cell(get_code_source(canonical, 9)))
    cells.append(code_cell(build_scopus_override_cell()))
    cells.append(code_cell(get_code_source(canonical, 10)))
    cells.append(code_cell(get_code_source(canonical, 11)))
    cells.append(code_cell(get_code_source(canonical, 12)))
    cells.append(code_cell(build_publish_cell()))
    cells.append(code_cell(build_switch_cell()))
    cells.append(code_cell(get_code_source(canonical, 13)))
    cells.append(code_cell(get_code_source(canonical, 14)))
    cells.append(code_cell(get_code_source(canonical, 15)))
    cells.append(code_cell(get_code_source(canonical, 16)))
    cells.append(code_cell(get_code_source(canonical, 17)))

    return {
        "cells": cells,
        "metadata": canonical.get("metadata", {}),
        "nbformat": canonical.get("nbformat", 4),
        "nbformat_minor": canonical.get("nbformat_minor", 5),
    }


def main() -> None:
    notebook = build_notebook()
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Notebook gerado em: {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
