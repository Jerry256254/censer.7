"""Upload the trained model to the Hugging Face Hub.

The checkpoint is exported as a *standard* ``LlamaForSequenceClassification``
(the Censer architecture is exactly Llama: RoPE, RMSNorm, SwiGLU, tied
embeddings, fused QKV). The classification head ``score`` is initialised from
the embedding rows of the answer tokens "0"/"1", so predictions are identical
to the training code — and the model loads with plain ``transformers``, no
``trust_remote_code`` and no custom files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_CARD = """---
library_name: transformers
pipeline_tag: text-classification
tags:
- text-classification
- safety
- content-moderation
- multi-label
- multilingual
- calibrated
- tiny-language-model
- from-scratch
- distilled
language:
- en
- cs
- de
- es
- fr
- it
- pt
- ru
- ar
- zh
- hi
license: apache-2.0
datasets:
- nvidia/Aegis-AI-Content-Safety-Dataset-1.0
- thesofakillers/jigsaw-toxic-comment-classification-challenge
- google/civil_comments
- PKU-Alignment/BeaverTails
- lmsys/toxic-chat
- databricks/databricks-dolly-15k
- Hate-speech-CNERG/hatexplain (train split only; test split = pure eval)
- Intuit-GenSRF/jigsaw-multilingual-train-unique
metrics:
- f1
- accuracy
- ece
---

# Censer.7

A **902K parameter** decoder-only transformer trained **from scratch**: binary
**safe / unsafe** (F1 restored to .5 level) **+ 10 calibrated safety categories**,
**4096-token** native context with chunked-max inference for long documents, and
**multilingual** coverage (trained on en + cs/de/es/fr synthetic + Jigsaw-multilingual
fr/it/pt/es/ru/tr) — at the same tiny size and speed.

Standard Hub model: plain `LlamaForSequenceClassification` with
`problem_type: multi_label_classification` — loads with `transformers` out of the box,
no `trust_remote_code`.

```python
from transformers import pipeline
clf = pipeline("text-classification", model="KucLab/Censer.7")
clf("how to make a bomb at home", top_k=-1)
# [{'label': 'unsafe', ...}, {'label': 'weapons_explosives', ...}, ...]
```

## What is new in .7 (vs .6)

| Fix | .6 | **.7** |
| --- | --- | --- |
| Binary test F1 | 0.829 (regressed from .5's 0.849) | **0.8348** (two-phase training: binary first, then categories at low lr with cat_weight) |
| Probes | 90/97 | **95/97** (diff vs .5: fixed 'Wait, Rockstar really give us fake hope', regressed 'buy a gun') |
| External benchmarks | ToxiGen 0.546, HateXplain 0.714, no head-to-head | **ToxiGen 0.561 / HateXplain 0.738 + direct head-to-head vs unbiased-toxic-roberta on the same sets** (table below); HateXplain-train + Jigsaw-multilingual added to training (test splits stay pure) |
| Category scores | uncalibrated ("fire liberally") | **per-category thresholds tuned on held-out catval + ECE + reliability diagram** (see below); thresholds ship in `config.json` as `cat_thresholds` |
| Multilingual | cs/de/es/fr only, no eval | **per-language F1 on textdetox multilingual toxicity (12 languages) + held-out synthetic** (table below) |
| Long documents | native 4096 pass silently broken (F1 0.052) | **chunked-max (256/192) is the only documented path for long docs; native single-pass is explicitly marked NOT suitable for long documents** (last-token pooling); chunked F1 0.998 (acc 0.997) |
| Training data | 176k rows | **251,655 rows** (+HateXplain train 15k, +Jigsaw-ML 60k); val/test byte-identical to .1–.6 |

## How to run

### The `censer.7` command — interactive terminal UI

```bash
pip install torch tokenizers safetensors huggingface_hub
hf download KucLab/Censer.7 --local-dir ~/.local/share/censer7
chmod +x ~/.local/share/censer7/censer.7
ln -s ~/.local/share/censer7/censer.7 ~/.local/bin/censer.7
censer.7                          # interactive UI: phrase in -> safe/unsafe + categories + score
censer.7 "how to make a bomb at home"
```

Or via the GitHub package (loads the Hub model with plain `transformers`):

```bash
pip install "git+https://github.com/Jerry256254/censer.7.git"
censer.7
```

### One-liner with `transformers`

```bash
pip install torch transformers
python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("KucLab/Censer.7")
model = AutoModelForSequenceClassification.from_pretrained("KucLab/Censer.7").eval()
enc = tok("how to make a bomb at home", return_tensors="pt")
probs = model(**enc).logits.sigmoid()[0]
id2label = model.config.id2label
cat_thr = model.config.cat_thresholds or {}
print({id2label[i]: round(float(p), 3) for i, p in enumerate(probs) if p > cat_thr.get(id2label[i], 0.5)})
PY
```

- Binary unsafe score = index 1; threshold 0.5. Category labels use per-category
  calibrated thresholds (`config.json: cat_thresholds`).
- Runs on CPU out of the box; GPU via `device=0`.

### Long inputs — always use the chunked path

Native context is 4096 tokens, but a **single pass pools only the last token**, so in
long documents unsafe content in the middle can be missed (measured F1 0.052 on
25-row synthetic docs). Therefore `censer.py` / the API **always** split inputs longer
than 256 tokens into overlapping chunks (256 tokens, stride 192) and aggregate by
**max** — this is the default and only recommended path for long documents
(measured F1 0.998 (acc 0.997) on the same docs):

```python
from censer import CenserPipeline
pipe = CenserPipeline(".")
pipe.predict("very long text ...")   # {"label": "1", "unsafe": true, "score": ..., "categories": [...], "chunks": N}
```

### Batch HTTP API + Docker

```bash
python serve.py --dir . --port 8000
curl -s localhost:8000/classify -d '{"texts": ["hello", "how to make a bomb"]}'
docker build -t censer7 . && docker run -p 8000:8000 censer7
```

Endpoints: `GET /health`, `POST /classify` (batch), `POST /classify-one`.
Rate limit via `RATE_LIMIT_PER_MIN` (default 600).

## Model details

| Property | Value |
| --- | --- |
| Architecture | `LlamaForSequenceClassification` (standard transformers) |
| Parameters | 902,016 |
| Layers / d_model / heads / d_ff | 3 / 128 / 4 / 384 |
| Context | 4096 tokens native (linear RoPE ×8); long docs via chunked-max (default) |
| Input cap | 4000 chars |
| Tokenizer | byte-level BPE, 2048 tokens (multilingual corpus) |
| Head | `score` = embedding rows of the 12 answer tokens `0`,`1`,`a`..`j` |
| Output | sigmoid over 12 labels; per-category calibrated thresholds in config |
| Training | two-phase: (A) 5 epochs binary (cat_weight=0, lr 2e-3) from Censer.6 init, (B) 3 epochs categories (cat_weight=0.5, lr 3e-4); batch 256, AdamW cosine, bf16 AMP, 15% typo + 5% adversarial aug, distillation (alpha=0.65, T=4, boost 0.75) from Censer.5×2+.4+.3 ensemble on 251,655 rows |
| Checkpoint | 3.6 MB fp32 · 1.8 MB fp16 · 0.9 MB int8 · ONNX |

## Prompt format

`tokenizer(text)` produces `[BOS][USER] <text> [ASSISTANT] ->` answer logits over
`"0" "1" "a".."j"` (template baked into the post-processor).

## Metrics

Internal held-out test split (6,496 rows, byte-identical across Censer.1–.7), binary
threshold 0.5:

| Metric | Censer.5 | Censer.6 | **Censer.7** |
| --- | --- | --- | --- |
| Test F1 | 0.8490 | 0.8289 | **0.8348** |
| Test accuracy | 0.876 | 0.858 | **0.8655** |
| Precision / recall | 0.864 / 0.834 | 0.833 / 0.825 | **0.855 / 0.816** |
| Val F1 | 0.8495 | 0.8358 | **0.8368** |
| Probes (97 hand-written) | 95/97 | 90/97 | **95/97** |

Per-source test F1: aegis 0.912, BeaverTails 0.881, jigsaw 0.856, civil_comments 0.776.

### External benchmarks (never in training) — with head-to-head vs public models

| Dataset | Model | F1@0.5 | best F1 | latency |
| --- | --- | --- | --- | --- |
| ToxiGen annotated (n=940, 402 unsafe) | **Censer.7** | **0.561** | 0.619 @ 0.27 | 1.24 ms |
| ToxiGen | unitary/unbiased-toxic-roberta | 0.443 | 0.483 | ~14 ms |
| HateXplain test (n=1,924, 1,376 unsafe) | **Censer.7** | **0.738** | 0.834 @ 0.05 | 1.24 ms |
| HateXplain test | unitary/unbiased-toxic-roberta | 0.577 | 0.690 | ~14 ms |

(full numbers in `reports/external_h2h.json`)

### Adversarial robustness (detection rate on attacked unsafe texts, n=160)

| Attack | .6 | **.7** |
| --- | --- | --- |
| clean | 91.3% | 93.8% |
| typo | 89.4% | 88.8% |
| leetspeak | 92.5% | 93.1% |
| homoglyph | 98.1% | 98.1% |
| zero-width | 95.0% | 97.5% |
| case flip | 92.5% | 91.9% |

### Long-context (300 synthetic docs of 25 held-out rows, avg 2,178 tokens)

| Inference | F1 |
| --- | --- |
| truncate at 192 tokens | 0.571 |
| native 4096 single pass (NOT recommended for long docs) | 0.052 |
| **chunked 256/192 max (default everywhere)** | **0.998 (acc 0.997)** |

### Category calibration (held-out HateXplain val, n=1,922)

Binary-score ECE = **0.1798**. Per-category tuned thresholds (F1-maximizing) ship in
`config.json: cat_thresholds`:

| Category | threshold | F1@thr | ECE |
| --- | --- | --- | --- |
| violence | 0.20 | 0.000 | 0.195 |
| weapons_explosives | 0.20 | 0.000 | 0.140 |
| drugs | 0.20 | 0.000 | 0.103 |
| hate_harassment | 0.20 | 0.834 | 0.112 |
| sexual_abuse | 0.20 | 0.000 | 0.124 |
| self_harm | 0.20 | 0.000 | 0.095 |
| cybercrime_fraud | 0.20 | 0.000 | 0.130 |
| jailbreak_injection | 0.20 | 0.000 | 0.088 |
| illegal_marketplace | 0.20 | 0.000 | 0.083 |
| other_unsafe | 0.20 | 0.000 | 0.147 |

![Reliability diagram](figs/reliability.png)

### Multilingual — per-language binary F1 (textdetox multilingual toxicity, never in training; + held-out synthetic)

| Language | n | unsafe | F1@0.5 | best F1 |
| --- | --- | --- | --- | --- |
| synth_cs | 300 | 150 | 1.000 | 1.000 |
| synth_de | 300 | 150 | 1.000 | 1.000 |
| synth_es | 300 | 150 | 1.000 | 1.000 |
| synth_fr | 300 | 150 | 1.000 | 1.000 |
| textdetox_am | 800 | 385 | 0.141 | 0.646 |
| textdetox_ar | 800 | 387 | 0.000 | 0.049 |
| textdetox_de | 800 | 397 | 0.596 | 0.690 |
| textdetox_en | 800 | 391 | 0.820 | 0.842 |
| textdetox_es | 800 | 408 | 0.543 | 0.674 |
| textdetox_fr | 800 | 391 | 0.512 | 0.676 |
| textdetox_he | 800 | 332 | 0.063 | 0.305 |
| textdetox_hi | 800 | 409 | 0.000 | 0.000 |
| textdetox_hin | 800 | 394 | 0.543 | 0.639 |
| textdetox_it | 800 | 409 | 0.607 | 0.677 |
| textdetox_ja | 800 | 409 | 0.000 | 0.010 |
| textdetox_ru | 800 | 391 | 0.399 | 0.655 |
| textdetox_tt | 800 | 403 | 0.590 | 0.671 |
| textdetox_uk | 800 | 391 | 0.393 | 0.657 |
| textdetox_zh | 800 | 409 | 0.010 | 0.190 |

### Cascade pre-filter

On 2,000 held-out rows: 42.8% escalated (gray-zone / jailbreak-category hits),
cascade accuracy 0.8600.

## Benchmark (latency / throughput / size)

AMD Radeon RX 9060 XT (ROCm) and CPU-only:

| Metric | .7 GPU | .7 CPU | .5 GPU |
| --- | --- | --- | --- |
| Parameters | 902,016 | 902,016 | 902,016 |
| Checkpoint | 3.6 MB | 3.6 MB (int8 0.9 MB) | 3.6 MB |
| Single-text latency | 1.24 ms | 0.61 ms | 1.43 ms |
| Single throughput | 804 texts/s | 1628 texts/s | 823/s |
| Batch-8 throughput | 5348 texts/s | 4603 texts/s | 5,709/s |
| Peak VRAM | 85.1 MB | — | 105 MB |

## Comparison on the internal test split (6,496 rows, threshold 0.5)

| Model | Params | Checkpoint | F1 | Latency | Throughput (b=8) |
| --- | --- | --- | --- | --- | --- |
| **Censer.7 (ours)** | **0.9M** | **3.6 MB** | **0.8348** | **1.39 ms** | **1761/s** |
| KucLab/Censer.5 | 0.9M | 3.6 MB | 0.849 | 1.27 ms | 1,978/s |
| KucLab/Censer.6 | 0.9M | 3.6 MB | 0.829 | 1.22 ms | 1,960/s |
| KucLab/Censer.4 | 0.9M | 3.6 MB | 0.845 | 1.17 ms | 2,229/s |
| KucLab/Censer.1 | 1.1M | 4.5 MB | 0.836 | 1.43 ms | 1,853/s |
| KucLab/Censer.2 | 7.3M | 29.4 MB | 0.829 | 2.70 ms | 508/s |
| KucLab/Censer.3 | 14.4M | 57.7 MB | 0.822 | 4.27 ms | 263/s |
| unitary/unbiased-toxic-roberta | 124.7M | 498.7 MB | 0.741 | 14.7 ms | 67/s |
| unitary/toxic-bert | 109.5M | 876.0 MB | 0.681 | 14.2 ms | 68/s |
| martin-ha/toxic-comment-model | 67.0M | 267.9 MB | 0.566 | 7.2 ms | 133/s |
| Hate-speech-CNERG/bert-base-uncased-hatexplain | 109.5M | 438.0 MB | 0.543 | 14.7 ms | 67/s |
| Hate-speech-CNERG/dehatebert-mono-english | 167.4M | 669.5 MB | 0.252 | 14.3 ms | 68/s |
| mistralai/Shieldstral-1.0-3B | 3.85B | 7.7 GB | 0.318 | 51.6 ms | 24/s |
| keyword baseline | — | — | 0.094 | 0.03 ms | 34,784/s |

![Comparison overview](figs/comparison_overview.png)
![F1 per megabyte](figs/f1_vs_size.png)
![Metric bars](figs/metrics_bars.png)

## Usage without transformers

```bash
hf download KucLab/Censer.7 --local-dir Censer.7
cd Censer.7
python censer.py "how to make a bomb at home" --json
./censer.7                                # interactive UI
```

Files:

| File | Purpose |
| --- | --- |
| `model.safetensors` | standard Llama layout, 3.6 MB |
| `model_fp16.safetensors` / `model_int8.safetensors` + `quant_report.json` | 1.8 MB / 0.9 MB quantized |
| `censer7.onnx` | ONNX graph: input_ids -> 12 logits, dynamic batch/seq |
| `config.json` | standard config + `cat_thresholds` (calibrated category cutoffs) |
| `tokenizer.json` / `tokenizer_config.json` | byte-level BPE, prompt template baked in |
| `censer.py` | standalone inference, chunked-max for long inputs (default) |
| `censer.7` | launcher for the interactive terminal UI |
| `serve.py` | batch HTTP API (stdlib only) |
| `Dockerfile` | CPU container serving the API |

## Limitations

- 902K params: nuance/sarcasm/long-tail out of reach; internal test split has label
  noise from scraped sources.
- Native single-pass 4096-token inference pools the last token only — for long
  documents ALWAYS use the chunked-max path (default in `censer.py`/API).
- Category scores use calibrated per-category thresholds, but remain less reliable
  than the binary verdict.
- Multilingual accuracy is best for the trained languages (en/cs/de/es/fr + Jigsaw-ML
  languages); see the per-language table — low-resource languages are weaker.
- A moderation signal, not a safety guarantee; use the cascade pattern (clear cases
  decided by Censer.7, gray-zone/jailbreak hits escalated to a larger validator).
"""


def cat_thresholds() -> dict:
    """Per-category calibrated thresholds from reports/calibration.json.

    Categories with no positive support in catval keep the default 0.5."""
    import json as _json

    p = Path(__file__).resolve().parent.parent / "reports" / "calibration.json"
    out = {}
    if p.exists():
        try:
            cal = _json.loads(p.read_text())
            for cat, v in cal.get("categories", {}).items():
                thr = float(v.get("best_threshold", 0.5))
                f1 = float(v.get("f1_at_best", 0.0))
                out[cat] = thr if f1 > 0 else 0.5
        except Exception:
            pass
    return out


def to_llama_state_dict(state: dict, answer_ids: list[int]) -> dict:
    """Map the training checkpoint onto standard LlamaForSequenceClassification keys."""
    model = state["model"]
    cfg = state["model_config"]
    d = cfg["d_model"]
    n_layers = cfg["n_layers"]

    out = {"model.embed_tokens.weight": model["embed.weight"].contiguous()}
    for i in range(n_layers):
        p = f"model.layers.{i}."
        b = f"blocks.{i}."
        qkv = model[f"{b}attn.qkv.weight"]
        out[f"{p}self_attn.q_proj.weight"] = qkv[:d].contiguous()
        out[f"{p}self_attn.k_proj.weight"] = qkv[d : 2 * d].contiguous()
        out[f"{p}self_attn.v_proj.weight"] = qkv[2 * d : 3 * d].contiguous()
        out[f"{p}self_attn.o_proj.weight"] = model[f"{b}attn.proj.weight"].contiguous()
        out[f"{p}mlp.gate_proj.weight"] = model[f"{b}mlp.gate.weight"].contiguous()
        out[f"{p}mlp.up_proj.weight"] = model[f"{b}mlp.up.weight"].contiguous()
        out[f"{p}mlp.down_proj.weight"] = model[f"{b}mlp.down.weight"].contiguous()
        out[f"{p}input_layernorm.weight"] = model[f"{b}norm1.weight"].contiguous()
        out[f"{p}post_attention_layernorm.weight"] = model[f"{b}norm2.weight"].contiguous()
    out["model.norm.weight"] = model["norm.weight"].contiguous()
    # The two answer logits are the tied-embedding logits of tokens "0" and "1".
    out["score.weight"] = model["embed.weight"][list(answer_ids)].clone().contiguous()
    return out


def llama_config(state: dict, tok_meta: dict, tok_ids: dict) -> dict:
    mc = dict(state["model_config"])
    mc.pop("answer_ids", None)
    from kuclab.categories import ID2LABEL

    n_answers = len(tok_meta["answer_ids"])
    rope_scaling = mc.get("rope_scaling") or None
    rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
    if rope_scaling and float(rope_scaling.get("factor", 1.0)) > 1.0:
        rope_parameters = {
            "rope_type": "linear",
            "factor": float(rope_scaling["factor"]),
            "rope_theta": 10000.0,
            "original_max_position_embeddings": mc["max_seq_len"] // int(float(rope_scaling["factor"])),
        }
    id2label = {str(k): v for k, v in ID2LABEL.items()} if n_answers > 2 else {"0": "safe", "1": "unsafe"}
    return {
        "architectures": ["LlamaForSequenceClassification"],
        "model_type": "llama",
        "num_labels": n_answers,
        "id2label": id2label,
        "label2id": {v: int(k) for k, v in id2label.items()},
        **({"problem_type": "multi_label_classification"} if n_answers > 2 else {}),
        "vocab_size": mc["vocab_size"],
        "hidden_size": mc["d_model"],
        "intermediate_size": mc["d_ff"],
        "num_hidden_layers": mc["n_layers"],
        "num_attention_heads": mc["n_heads"],
        "num_key_value_heads": mc["n_heads"],
        "head_dim": mc["d_model"] // mc["n_heads"],
        "max_position_embeddings": mc["max_seq_len"],
        "rms_norm_eps": 1e-5,
        "rope_parameters": rope_parameters,
        **({"rope_scaling": {"rope_type": "linear", "factor": float(rope_scaling["factor"])}} if rope_parameters["rope_type"] == "linear" else {}),
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": True,
        "pad_token_id": mc["pad_id"],
        "bos_token_id": tok_ids["[BOS]"],
        "eos_token_id": tok_ids["[ASSISTANT]"],
        # Censer-specific metadata, kept for the standalone censer.py:
        "name": state.get("name", "Censer.7"),
        "max_chars": int(state.get("max_chars", 4000)),
        "answer_ids": list(tok_meta["answer_ids"]),
        "special_tokens": list(tok_meta["special_tokens"]),
        "threshold": 0.5,
        # calibrated per-category cutoffs (tuned on held-out catval; falls back to 0.5)
        **({"cat_thresholds": cat_thresholds()} if n_answers > 2 else {}),
    }


def write_tokenizer(src: Path, dst: Path) -> dict:
    """Ship the tokenizer with the prompt template baked in, so AutoTokenizer
    produces exactly [BOS][USER] text [ASSISTANT]. Returns the special-token ids."""
    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing

    tok = Tokenizer.from_file(str(src))
    ids = {name: tok.token_to_id(name) for name in ("[PAD]", "[BOS]", "[USER]", "[ASSISTANT]")}
    tok.post_processor = TemplateProcessing(
        single="[BOS] [USER] $A [ASSISTANT]",
        special_tokens=[("[BOS]", ids["[BOS]"]), ("[USER]", ids["[USER]"]), ("[ASSISTANT]", ids["[ASSISTANT]"])],
    )
    tok.save(str(dst))
    return ids


def write_tokenizer_config(dst: Path, ids: dict) -> None:
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[ASSISTANT]",
    }
    dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def build_bundle(ckpt: Path, out: Path) -> None:
    """Write a self-contained, standard-transformers model directory."""
    import torch
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)

    tok_dir = ckpt.parent / "tokenizer"
    tok_meta = json.loads((tok_dir / "meta.json").read_text(encoding="utf-8"))
    answer_ids = list(tok_meta["answer_ids"])

    tensors = to_llama_state_dict(state, answer_ids)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    ids = write_tokenizer(tok_dir / "tokenizer.json", out / "tokenizer.json")
    write_tokenizer_config(out / "tokenizer_config.json", ids)

    config = llama_config(state, tok_meta, ids)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    shutil.copy2(Path(__file__).parent / "censer.py", out / "censer.py")
    launcher = Path(__file__).parent / "censer.7"
    shutil.copy2(launcher, out / "censer.7")
    os.chmod(out / "censer.7", 0o755)
    (out / "README.md").write_text(MODEL_CARD, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Censer.7 to the Hugging Face Hub.")
    parser.add_argument("--repo", default="KucLab/Censer.7", help="e.g. KucLab/Censer.7")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        build_bundle(ckpt, bundle)
        print(f"bundle: {[p.name for p in sorted(bundle.iterdir())]}")

        if args.dry_run:
            print("dry run, not uploading")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        who = api.whoami()
        print(f"authenticated as: {who.get('name')}")

        api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(bundle),
            repo_id=args.repo,
            repo_type="model",
            commit_message="Standard LlamaForSequenceClassification layout: loads with plain transformers, no trust_remote_code",
            delete_patterns=["modeling_censer.py"],
        )

        figs = Path(__file__).resolve().parent.parent / "reports" / "figs"
        if figs.exists():
            api.upload_folder(
                folder_path=str(figs),
                repo_id=args.repo,
                repo_type="model",
                path_in_repo="figs",
                commit_message="Add comparison charts vs other safety classifiers",
            )
        print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
