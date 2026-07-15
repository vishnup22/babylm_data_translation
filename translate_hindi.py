"""
Translate BabyLM English data to Hindi using IndicTrans2.

Covers all three splits:
  - train: BabyLM-community/BabyLM-2026-Strict  (*.train.txt)
  - dev:   BabyLM-community/BabyLM-dev           (*.dev)
  - test:  BabyLM-community/BabyLM-Test          (*.test)

Run: accelerate launch --num_processes 4 translate_hindi.py --split train
     accelerate launch --num_processes 4 translate_hindi.py --split dev
     accelerate launch --num_processes 4 translate_hindi.py --split test
"""

import math
import logging
import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"
LANG_CODE = "hi"
MODEL_ID = "ai4bharat/indictrans2-en-indic-1B"
BATCH_SIZE = 64
MAX_SEQ_LEN = 256
CHUNK_LINES = 50_000

STEMS = ["switchboard", "bnc_spoken", "open_subtitles", "simple_wiki", "childes", "gutenberg"]

# Per-split source dataset + input filename suffix + output filename suffix
SPLIT_CONFIG = {
    "train": {
        "hf_repo": "BabyLM-community/BabyLM-2026-Strict",
        "input_suffix": ".train.txt",
        "output_suffix": f".train.{LANG_CODE}.txt",
    },
    "dev": {
        "hf_repo": "BabyLM-community/BabyLM-dev",
        "input_suffix": ".dev",
        "output_suffix": f".dev.{LANG_CODE}.txt",
    },
    "test": {
        "hf_repo": "BabyLM-community/BabyLM-Test",
        "input_suffix": ".test",
        "output_suffix": f".test.{LANG_CODE}.txt",
    },
}


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def read_lines_range(path, start, end):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= end:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def load_checkpoint(ckpt_file):
    p = Path(ckpt_file)
    if p.exists():
        return int(p.read_text().strip())
    return -1


def save_checkpoint(ckpt_file, chunk_idx):
    Path(ckpt_file).write_text(str(chunk_idx))


def translate_batch(sentences, model, tokenizer, ip, device):
    batch = ip.preprocess_batch(sentences, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=5,
            num_return_sequences=1,
            max_length=MAX_SEQ_LEN,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return ip.postprocess_batch(decoded, lang=TGT_LANG)


def translate_file(input_path, output_path, model, tokenizer, ip, accelerator, output_dir, split):
    filename = Path(input_path).name
    total_lines = count_lines(input_path)
    num_chunks = math.ceil(total_lines / CHUNK_LINES)
    ckpt_file = Path(output_dir) / f".ckpt_hi_{split}_{filename}"
    last_done = load_checkpoint(ckpt_file)

    if accelerator.is_main_process:
        logger.info(f"{filename}: {total_lines:,} lines, {num_chunks} chunks, starting at chunk {last_done + 1}")

    mode = "a" if last_done >= 0 else "w"
    out_file = open(output_path, mode, encoding="utf-8")

    for chunk_idx in range(num_chunks):
        if chunk_idx <= last_done:
            continue

        start = chunk_idx * CHUNK_LINES
        end = min(start + CHUNK_LINES, total_lines)
        lines = read_lines_range(input_path, start, end)

        process_idx = accelerator.process_index
        num_processes = accelerator.num_processes
        local_lines = lines[process_idx::num_processes]

        local_translations = []
        for batch in chunk_list(local_lines, BATCH_SIZE):
            if not batch:
                continue
            try:
                translated = translate_batch(batch, model, tokenizer, ip, accelerator.device)
                local_translations.extend(translated)
            except Exception as e:
                logger.warning(f"GPU {process_idx} batch error: {e}")
                local_translations.extend([""] * len(batch))

        all_translations = gather_object(
            [(process_idx, i, t) for i, t in enumerate(local_translations)]
        )

        if accelerator.is_main_process:
            ordered = [None] * len(lines)
            for gpu_idx, local_i, text in all_translations:
                original_i = gpu_idx + local_i * num_processes
                if original_i < len(ordered):
                    ordered[original_i] = text

            for t in ordered:
                out_file.write((t or "") + "\n")
            out_file.flush()

            save_checkpoint(ckpt_file, chunk_idx)
            logger.info(f"{filename}: chunk {chunk_idx + 1}/{num_chunks} ({end:,}/{total_lines:,} lines)")

        accelerator.wait_for_everyone()

    out_file.close()

    if accelerator.is_main_process:
        size_mb = Path(output_path).stat().st_size / 1e6
        logger.info(f"done: {output_path} ({size_mb:.1f} MB)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=list(SPLIT_CONFIG), default="train",
                        help="Which BabyLM split to translate: train, dev, or test")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing the source split's files "
                             "(default: ./babylm_data_<split>, matching --split's source dataset "
                             "cloned locally, e.g. BabyLM-community/BabyLM-dev for --split dev)")
    parser.add_argument("--output_dir", type=str, default="./babylm_hindi")
    parser.add_argument("--stems", type=str, nargs="+", default=STEMS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--chunk_lines", type=int, default=CHUNK_LINES)
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    cfg = SPLIT_CONFIG[args.split]
    input_dir = args.input_dir or f"./babylm_data_{args.split}"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        logger.info(f"English -> Hindi | split={args.split} (source: {cfg['hf_repo']}) | "
                    f"{accelerator.num_processes} GPUs | output: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id, trust_remote_code=True
    ).to(accelerator.device)
    model.eval()
    model = accelerator.prepare(model)
    ip = IndicProcessor(inference=True)

    for stem in args.stems:
        input_path = Path(input_dir) / f"{stem}{cfg['input_suffix']}"
        if not input_path.exists():
            if accelerator.is_main_process:
                logger.warning(f"not found, skipping: {input_path}")
            continue

        output_path = Path(args.output_dir) / f"{stem}{cfg['output_suffix']}"

        translate_file(
            input_path=str(input_path),
            output_path=str(output_path),
            model=model,
            tokenizer=tokenizer,
            ip=ip,
            accelerator=accelerator,
            output_dir=args.output_dir,
            split=args.split,
        )

    if accelerator.is_main_process:
        logger.info(f"Hindi {args.split} translation complete")
        for f in sorted(Path(args.output_dir).glob(f"*.{args.split}.{LANG_CODE}.txt")):
            lines = sum(1 for _ in open(f, encoding="utf-8"))
            size_mb = f.stat().st_size / 1e6
            logger.info(f"  {f.name:<40} {lines:>10,} lines  {size_mb:>7.1f} MB")


if __name__ == "__main__":
    main()
