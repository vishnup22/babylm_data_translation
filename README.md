# BabyLM Translated Data: English → Hindi & Telugu

Translates the BabyLM 2026 Strict corpus (train + dev + test) from English into Hindi and Telugu using [IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B).

---

## Source Datasets (English originals)

| Split | Source | Link |
|---|---|---|
| Train | BabyLM-2026-Strict (100M tokens) | https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict |
| Dev / Val | BabyLM-dev | https://huggingface.co/datasets/BabyLM-community/BabyLM-dev |
| Test | BabyLM-Test | https://huggingface.co/datasets/BabyLM-community/BabyLM-Test |

Each split covers the same six sub-corpora:

| Source | Description | Link |
|---|---|---|
| `bnc_spoken` | British National Corpus — spoken language | https://www.english-corpora.org/bnc/ |
| `childes` | Child-directed speech (CHILDES database) | https://childes.talkbank.org/ |
| `gutenberg` | Project Gutenberg literary texts | https://www.gutenberg.org |
| `open_subtitles` | Movie and TV subtitles | https://opus.nlpl.eu/OpenSubtitles.php |
| `simple_wiki` | Simple English Wikipedia | https://simple.wikipedia.org |
| `switchboard` | Telephone conversation transcripts | https://catalog.ldc.upenn.edu/LDC97S62 |

---

## Translated Datasets (outputs)

| Language | Files |
|---|---|
| Hindi | `hindi/train`, `hindi/val`, `hindi/test` |
| Telugu | `telugu/train`, `telugu/val`, `telugu/test` |

Anonymized view-only mirror (train + dev/val + test, produced by the scripts in this repo): **[osf.io/973gp](https://osf.io/973gp/?view_only=b255c4a0243341f498b6b243ddf528f6)**

- **Translation model:** [IndicTrans2 (`ai4bharat/indictrans2-en-indic-1B`)](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B), a 1B-parameter open-source NMT model covering all 22 scheduled Indian languages.
- **Direction:** `eng_Latn` → `hin_Deva` / `eng_Latn` → `tel_Telu`
- Both languages are translated from the same three English source splits above, so Hindi and Telugu are parallel to each other (and to the English original) at the sentence level.

---

## 1. Clone the Source Data

```bash
# train
git clone https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict babylm_data_train
# dev
git clone https://huggingface.co/datasets/BabyLM-community/BabyLM-dev babylm_data_dev
# test
git clone https://huggingface.co/datasets/BabyLM-community/BabyLM-Test babylm_data_test

for d in babylm_data_train babylm_data_dev babylm_data_test; do
    (cd $d && git lfs install && git lfs pull)
done
```

## 2. Set Up Environment

```bash
conda env create -f telugu_llm_env.yml -n translation
conda activate translation
```

## 3. Configure Accelerate

```bash
accelerate config
# set: multi-GPU, num_processes=4, fp16
```

## 4. Translate

Each script takes `--split {train,dev,test}`, pulling from the matching source folder above.

### Telugu
```bash
accelerate launch --num_processes 4 translate_telugu.py --split train --input_dir ./babylm_data_train --output_dir ./babylm_telugu
accelerate launch --num_processes 4 translate_telugu.py --split dev   --input_dir ./babylm_data_dev   --output_dir ./babylm_telugu
accelerate launch --num_processes 4 translate_telugu.py --split test  --input_dir ./babylm_data_test  --output_dir ./babylm_telugu
```

### Hindi
```bash
accelerate launch --num_processes 4 translate_hindi.py --split train --input_dir ./babylm_data_train --output_dir ./babylm_hindi
accelerate launch --num_processes 4 translate_hindi.py --split dev   --input_dir ./babylm_data_dev   --output_dir ./babylm_hindi
accelerate launch --num_processes 4 translate_hindi.py --split test  --input_dir ./babylm_data_test  --output_dir ./babylm_hindi
```

Jobs are resumable — each writes a `.ckpt_<lang>_<split>_<file>` checkpoint after every chunk, so a killed/failed job picks back up rather than restarting.

## 5. Run on SLURM

```bash
mkdir -p logs babylm_telugu babylm_hindi
bash submit_hindi_jobs.sh
```

Monitor:
```bash
squeue -u $USER
tail -f logs/<job_name>_<jobid>.log
du -sh babylm_telugu/*.txt babylm_hindi/*.txt
```

---

## Dataset Statistics

Sentence/word counts as published on each dataset card; byte sizes measured directly from the hosted files.

### Hindi (`translated-babylm-hindi`)

| Split | Sentences | Words | Bytes |
|---|---|---|---|
| Train | 11,579,880 | 118,309,059 | 1,393,482,920 |
| Val | 1,153,113 | 11,882,662 | 146,347,181 |
| Test | 1,097,453 | 11,106,875 | 130,956,946 |
| **Total** | **13,830,446** | **141,298,596** | **1,670,787,047** |

### Telugu (`translated-babylm-telugu`)

| Split | Sentences | Words | Bytes |
|---|---|---|---|
| Train | 11,579,880 | 79,179,423 | 1,482,914,710 |
| Val | 1,153,113 | 8,227,178 | 153,974,404 |
| Test | 1,096,313 | 7,455,904 | 139,585,901 |
| **Total** | **13,829,306** | **94,862,505** | **1,776,475,015** |

Per-file breakdowns (train split, both languages have identical sentence counts per source since they're translated from the same English lines):

| Source | Sentences | Hindi Words | Telugu Words |
|---|---|---|---|
| `bnc_spoken` | 797,548 | 8,568,128 | 5,637,643 |
| `childes` | 5,638,779 | 38,080,105 | 25,095,648 |
| `gutenberg` | 661,689 | 26,928,762 | 17,387,618 |
| `open_subtitles` | 3,808,717 | 27,181,937 | 18,275,525 |
| `simple_wiki` | 642,588 | 17,270,682 | 12,585,806 |
| `switchboard` | 30,559 | 279,445 | 197,183 |

Telugu words are consistently lower than Hindi for the same sentences — expected, since Telugu is agglutinative and tends to pack more morphemes per word than Hindi.

---

## Known Issues

- **Filename inconsistency:** some `val/` and `test/` files carry a redundant `.txt.train` fragment in their name (e.g. `val/childes.dev.txt.train.hi.txt` instead of `val/childes.dev.hi.txt`) — an artifact of an earlier translation run. Files are still correctly split by content; only the name is inconsistent.
- **Mislabeled file:** `val/simple_wiki.dev.te.txt` in the **Hindi** dataset (`translated-babylm-hindi`) has a `.te.txt` (Telugu) extension and is byte-identical to the Telugu dataset's own `val/simple_wiki.dev.te.txt` — this file is Telugu content sitting in the Hindi repo's `val/` folder, not a genuine Hindi translation. Needs to be regenerated and re-uploaded.

---

## Notes

- Translation uses [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) (1B model)
- Jobs are resumable — if a job fails, resubmit and it picks up from the last completed chunk
- `transformers==4.40.0` is required — newer versions break IndicTrans2's `past_key_values` handling
- Telugu translation additionally applies a `clean_sentences()` filter (drops empty/too-short lines) before batching
- Both scripts load the model directly onto each GPU rather than wrapping it with `accelerator.prepare()` — IndicTrans2 is a seq2seq inference model and DDP wrapping breaks `.generate()`
