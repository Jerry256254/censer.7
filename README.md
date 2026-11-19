# Censer.7

A tiny decoder-only LLM (**902K parameters, 3.6 MB**) trained **from scratch** on an
AMD GPU. Binary **safe/unsafe** + **10 calibrated safety categories**, **4096-token**
context (chunked-max for long documents), **multilingual** (en/cs/de/es/fr + Jigsaw-ML).

```
$ censer.7 "how to make a bomb at home"
unsafe 0.85  [weapons_explosives]
$ censer.7 "how do I kill a process in Linux"
safe 0.96
$ censer.7 "ahoj, jak se máš?"
safe ...
$ censer.7 "wie baue ich eine bombe"
unsafe ...
```

Standard Hub model (`LlamaForSequenceClassification`, `multi_label_classification`) —
loads with plain `transformers`, no custom code.

## What changed in .7 (vs .6)

| Fix | .6 | **.7** |
| --- | --- | --- |
| Test F1@0.5 / probes | 0.829 / 90-97 | **0.8348 / 95-97** (two-phase training: binary first at cat_weight=0, then categories at lr 3e-4; + polish pass) |
| External benchmarks | ToxiGen 0.546, HateXplain 0.714 | **ToxiGen 0.561, HateXplain 0.739 + head-to-head vs unbiased-toxic-roberta on the same sets** (we win both: 0.561 vs 0.443, 0.739 vs 0.577); HateXplain-train + Jigsaw-multilingual (60k) added to training, test splits stay pure |
| Category calibration | uncalibrated | **per-category thresholds on held-out catval + ECE 0.18 + reliability diagram** (`reports/calibration.json`, `figs/reliability.png`, thresholds ship in `config.json: cat_thresholds`) |
| Multilingual eval | none | **per-language F1 on textdetox multilingual (16 languages) + held-out synthetic cs/de/es/fr** (`reports/multilingual.md`) — trained languages strong (en 0.82, de 0.60, it 0.61, es 0.54, fr 0.51, synth cs/de/es/fr 1.00), low-resource weak (ar/zh/hi/ja ~0) |
| Long documents | native 4096 pass broken (F1 0.05) | **chunked-max 256/192 is the default everywhere; F1 0.998** on 300 long docs; native single-pass explicitly marked NOT for long docs |
| Adversarial | clean 91.3% | **clean 93.8%, homoglyph 98.1%, zero-width 97.5%, leet 93.1%, typo 88.7%** |
| Training data | 176k rows | **251,655 rows** (val/test byte-identical to .1–.6) |
| Params / size / latency | 0.90M / 3.6 MB / 1.06 ms | **0.90M / 3.6 MB / 1.24 ms GPU, 0.61 ms CPU** |

Binary F1 vs .5 (0.8490): the multi-label head still costs ~0.014, but probes are back
to .5's 95/97 and .7 adds categories + 4k context + multilingual + calibration at the
same size. On the internal test split .7 ranks **#1 among all compared models** after
.5/.4 (best public: unbiased-toxic-roberta 0.741); on the external sets .7 beats
unbiased-toxic-roberta head-to-head.

## Install & run

```bash
pip install "git+https://github.com/Jerry256254/censer.7.git"
censer.7                          # interactive UI
censer.7 "how to make a bomb at home"
```

Hub files only (torch + tokenizers + safetensors):

```bash
hf download KucLab/Censer.7 --local-dir ~/.local/share/censer7
ln -s ~/.local/share/censer7/censer.7 ~/.local/bin/censer.7
censer.7
```

```bash
pip install torch transformers
python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("KucLab/Censer.7")
model = AutoModelForSequenceClassification.from_pretrained("KucLab/Censer.7").eval()
probs = model(**tok("how to make a bomb at home", return_tensors="pt")).logits.sigmoid()[0]
id2label = model.config.id2label
thr = model.config.cat_thresholds or {}
print({id2label[i]: round(float(p),3) for i,p in enumerate(probs) if p > thr.get(id2label[i], 0.5)})
PY
```

Batch API + Docker:

```bash
python scripts/serve.py --dir /tmp/bundle7 --port 8000
curl -s localhost:8000/classify -d '{"texts": ["hello", "how to make a bomb"]}'
docker build -t censer7 . && docker run -p 8000:8000 censer7
```

Long inputs: chunked-max (256 tokens / stride 192) is applied automatically in
`censer.py` and the API — do NOT rely on the native 4096 single pass for long
documents (last-token pooling).

## Layout

```
├── kuclab/            # library: model, tokenizer, train (two-phase + --init-checkpoint), evaluate, multilingual, categories
├── scripts/           # extend_train7.py (external data), eval_ckpt, benchmark, compare_models,
│                      # eval_external(+_h2h head-to-head), adversarial_eval, eval_long_context,
│                      # cascade, calibrate_categories, eval_multilingual, probes_diff,
│                      # export_quant, export_onnx, serve.py, upload_hf.py, censer.py, censer.7
├── configs/           # censer7a (binary phase), censer7b (category phase), censer7c (polish), censer7.json
├── tests/             # 50 tests
├── data/              # train/val/test jsonl (val/test byte-identical across .1–.7)
├── Dockerfile
└── reports/           # eval_test, benchmark(_cpu), comparison(+figs), external(_h2h), adversarial,
                       # long_context, cascade, calibration(+reliability.png), multilingual, probes_diff
```

## Train / eval from scratch

```bash
bash scripts/setup_env.sh
python -m kuclab.prepare_data
python scripts/extend_train7.py                 # add HateXplain-train + Jigsaw-ML (append-only)
python -m kuclab.distill --data data/processed --teachers ... --out data/processed/train_soft7.npy
python -m kuclab.train --config configs/censer7a.json --data data/processed --out checkpoints7 --init-checkpoint <prev best.pt>
python -m kuclab.train --config configs/censer7b.json --data data/processed --out checkpoints7 --resume
python scripts/eval_ckpt.py --checkpoint checkpoints7/best.pt
python scripts/eval_external_h2h.py --checkpoint checkpoints7/best.pt
python scripts/calibrate_categories.py --checkpoint checkpoints7/best.pt
python scripts/eval_multilingual.py --checkpoint checkpoints7/best.pt
python scripts/benchmark.py && python scripts/benchmark.py --device cpu --tag cpu
python scripts/compare_models.py && python scripts/make_comparison_figs.py
python -m pytest tests -q
```
