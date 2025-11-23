import streamlit as st
from PyPDF2 import PdfReader
import io
import threading
import time
import json
import re
import os
import sys
import torch
import pandas as pd
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig

PREFERRED = "meta-llama/Llama-3.1-8B-Instruct"
FALLBACKS = ["HuggingFaceH4/zephyr-7b-beta", "gpt2"]
OFFLOAD_FOLDER = "./offload"
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 3000
os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

PROMPT_TEMPLATE = """You are a precise extractor. Input is a single block of text describing one person. Your job: extract every row for a table with columns exactly: "Key", "Value", "Comments".

Rules (follow exactly):
1. Output ONLY a single valid JSON array (start with "[" and end with "]") — no surrounding text, no explanation, no headings.
2. Each item in the array must be a JSON object with exactly three keys: "Key", "Value", "Comments".
3. Keys must be standardized, human-readable, Title Case. Use the following keys where appropriate (omit objects that do not apply).
4. Format rules:
   - Date values must use DD-MMM-YYYY if exact day is available (e.g., "15-Mar-1989"). If only year given, use "YYYY".
   - Salary numeric values must be digits only (no separators): e.g., 2800000. Put currency in the corresponding currency field (e.g., "INR").
   - Percent/CGPA keep original form (e.g., "92.5%", "8.7").
   - For multi-valued fields (e.g., multiple certifications), present them as a single semicolon-separated string in the "Value" field, and put any explanatory detail in "Comments".
   - For similar keys (e.g., certificates, salary, score), add a numeral suffix (e.g., Certificate 1, Certificate 2, Certificate 3 or Salary 1, Salary 2, Salary 3).
5. Comments: short human-readable explanation or origin (one sentence max). If none, set Comments to an empty string "".
6. Preserve exact names and tokens present in the input. Do NOT invent facts or values not present.
7. Do not return null/None — instead omit that key/object entirely.
8. Order the array so top-level personal identifiers appear first (First/Last/Date of Birth/Birth City/State/Age/Blood Group/Nationality), then professional history (joining dates, organizations, salaries), then education, then certifications, then technical proficiency.
9. Output must be syntactically valid JSON. Use double quotes for strings.

Now extract from the Context text (which follows the prompt). Return the JSON array only.
"""


def gpu_info():
    if not torch.cuda.is_available():
        return None
    try:
        idx = 0
        name = torch.cuda.get_device_name(idx)
    except Exception:
        name = "cuda:0"
    try:
        free, total = torch.cuda.mem_get_info(idx)
        return {"name": name, "free_gb": free/(1024**3), "total_gb": total/(1024**3)}
    except Exception:
        return {"name": name, "free_gb": None, "total_gb": None}


def build_max_memory_map(total_vram_gb):
    if total_vram_gb is None:
        return {0: "4GB", "cpu": "200GB"}
    gpu_allow = max(1, int(total_vram_gb) - 1)
    return {0: f"{gpu_allow}GB", "cpu": "200GB"}


def select_model_class(config):
    if config.model_type and "t5" in config.model_type:
        return AutoModelForSeq2SeqLM
    if getattr(config, "is_encoder_decoder", False):
        return AutoModelForSeq2SeqLM
    return AutoModelForCausalLM


def try_load(model_name):
    st.info(f"Attempting to load model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    gpu = gpu_info()
    total_vram = gpu["total_gb"] if gpu else None
    max_memory = build_max_memory_map(total_vram)

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    ModelClass = select_model_class(cfg)

    load_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "max_memory": max_memory,
        "low_cpu_mem_usage": True,
        "offload_folder": OFFLOAD_FOLDER,
    }

    model = ModelClass.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_model():
    last_exc = None
    for candidate in [PREFERRED] + FALLBACKS:
        try:
            tokenizer, model = try_load(candidate)
            return tokenizer, model, candidate
        except Exception as e:
            last_exc = e
            st.warning(f"Failed to load {candidate}: {type(e).__name__}: {e}")
    raise RuntimeError(f"Failed to load any model. Last error: {type(last_exc).__name__}: {last_exc}")

class StopOnSubstring(StoppingCriteria):
    def __init__(self, tokenizer, substring: str = "]", lookback_tokens: int = 64):
        self.tokenizer = tokenizer
        self.substring = substring
        self.lookback_tokens = lookback_tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        seq = input_ids[0, -self.lookback_tokens : ].tolist()
        try:
            text = self.tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception:
            text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.substring in text


def generate_with_progress_streamlit(model, tokenizer, context_text: str, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS,
                                     stop_substring: str = "]", progress_obj=None, log_area=None):
    full_input = f"{prompt}\n\nContext:\n{context_text}\n\nAnswer:"
    inputs = tokenizer(full_input, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_TOKENS)

    try:
        emb_device = model.get_input_embeddings().weight.device
    except Exception:
        emb_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(emb_device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    stop_crit = StopOnSubstring(tokenizer=tokenizer, substring=stop_substring, lookback_tokens=128)
    stopping = StoppingCriteriaList([stop_crit])

    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        streamer=streamer,
        stopping_criteria=stopping,
    )

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    collected = []
    token_count = 0
    try:
        for chunk in streamer:
            collected.append(chunk)
            token_inc = max(1, len(chunk.split()))
            token_count += token_inc
            if progress_obj:
                frac = min(1.0, token_count / max_new_tokens)
                progress_obj.progress(frac)
            if log_area:
                log_area.text(''.join(collected)[-1000:])
            if stop_substring in chunk:
                break
    finally:
        thread.join(timeout=1.0)

    full_text = ''.join(collected)
    return full_text

def extract_json_array(raw_text: str):
    norm = raw_text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    norm = norm.replace("—", "-").replace("–", "-").replace("\u00A0", " ").strip()

    m = re.search(r"Answer\s*:\s*(\[[\s\S]*?\])", norm, flags=re.IGNORECASE)
    candidate = None
    if m:
        candidate = m.group(1)
    else:
        m2 = re.search(r"\[.*?\]", norm, flags=re.DOTALL)
        if m2:
            candidate = m2.group(0)

    if not candidate:
        snippet = norm[:1000].replace("\n", " ")
        raise ValueError(f"No JSON array found. Raw snippet: {snippet}")

    cleaned = re.sub(r",\s*}", "}", candidate)
    cleaned = re.sub(r",\s*\]", "]", cleaned)
    if '"' not in cleaned and "'" in cleaned:
        cleaned = cleaned.replace("'", '"')

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        start = cleaned[:500]
        raise ValueError(f"JSON parsing failed: {e}\nStart of candidate JSON:\n{start}")


def json_array_to_dataframe(arr):
    rows = []
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        k = obj.get("Key")
        v = obj.get("Value", "")
        c = obj.get("Comments", "")
        if not k:
            continue
        rows.append({"Key": k, "Value": v, "Comments": c})
    df = pd.DataFrame(rows, columns=["Key", "Value", "Comments"])
    return df

def main():
    st.set_page_config(page_title="PDF -> Extracted Excel", layout="wide")
    st.title("PDF → Key/Value Table Extractor")

    with st.expander("Model status and load"):
        try:
            tokenizer, model, used = load_model()
            st.success(f"Loaded model: {used}")
            info = gpu_info()
            st.write("GPU info:", info)
        except Exception as e:
            st.error(f"Model load failed: {e}")
            return

    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"] )

    if not pdf_file:
        st.info("Upload a PDF to begin. The extracted Excel will be offered for download after processing.")
        return

    reader = PdfReader(pdf_file)
    text_data = ""
    for page in reader.pages:
        extracted = page.extract_text() or ""
        text_data += extracted + "\n"

    st.subheader("Preview of extracted text")
    st.text_area("Extracted Text", text_data, height=250)

    if st.button("Run extraction and produce Excel"):
        progress = st.progress(0.0)
        log = st.empty()
        try:
            raw = generate_with_progress_streamlit(model, tokenizer, text_data, PROMPT_TEMPLATE, max_new_tokens=MAX_NEW_TOKENS, progress_obj=progress, log_area=log)
            st.write("Raw model output (truncated):")
            st.code(raw[-2000:])
            arr = extract_json_array(raw)
            df = json_array_to_dataframe(arr)

            if df.empty:
                st.warning("Extraction returned an empty table. Check raw output above for debugging.")
            else:
                st.success("Extraction succeeded. Preview below:")
                st.dataframe(df)

                towrite = io.BytesIO()
                df.to_excel(towrite, index=False, sheet_name="Key & Value")
                towrite.seek(0)
                st.download_button("Download Excel", data=towrite, file_name="output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Processing failed: {type(e).__name__}: {e}")

if __name__ == '__main__':
    main()