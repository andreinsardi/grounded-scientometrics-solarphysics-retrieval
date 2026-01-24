# SciBERT-SolarPhysics-Search — Reproducible Notebooks (PIPE 1–4)

This repository contains **four end-to-end Jupyter notebooks** reproducing the experimental pipeline:

- **PIPE 1**: **Train & package** `SciBERT-SolarPhysics-Search` (DAPT + supervised contrastive FT) and **build the FAISS index**
- **PIPE 2**: Retriever analytics (coverage / alignment / connectivity diagnostics)
- **PIPE 3**: Grounded LLM agent experiment (A1–A4), exporting claim-level metrics (ICR/UCR)
- **PIPE 4**: Statistical validation and robustness checks

Hugging Face model (published):
- `andreinsardi/SciBERT-SolarPhysics-Search`

> No service keys are committed. Use `.env` / environment variables.

---

## Suggested repository name

- `scibert-solarphysics-search-pipelines`

---

## Structure

```
.
├── notebooks/
│   ├── 01_PIPE1_Train_Model_And_Build_Index.ipynb
│   ├── 02_PIPE2_Retriever_Analytics.ipynb
│   ├── 03_PIPE3_Agent_Scientometrics.ipynb
│   └── 04_PIPE4_Statistical_Validation.ipynb
├── data/
│   ├── corpus.parquet            # required for PIPE 1 (or set CORPUS_PATH)
│   └── pairs.jsonl               # optional supervised pairs for contrastive FT
├── artifacts/
│   ├── pipe1/
│   │   ├── model/
│   │   └── embeddings/
│   └── retriever/
│       ├── faiss.index
│       └── docs.parquet
├── runs/
│   └── (exported logs/metrics)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` and edit values.

---

## Run order

1. `01_PIPE1_Train_Model_And_Build_Index.ipynb`
2. `02_PIPE2_Retriever_Analytics.ipynb`
3. `03_PIPE3_Agent_Scientometrics.ipynb`
4. `04_PIPE4_Statistical_Validation.ipynb`

---

## GPU usage

Install a CUDA-enabled PyTorch build matching your CUDA version.

Example (CUDA 12.1):
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Performance knobs:
- `EMB_BATCH`: increase until VRAM limit.
- `MAX_LENGTH_*`: reduce if you see OOM.

---

## Reproducibility notes

- Outputs are not committed; notebooks export into `artifacts/` and `runs/`.
- PIPE 3 requires your own LLM API key via env var if you run generation.

---
