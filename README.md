# DOCUMENT STRUCTURER : KEY-VALUE-COMMENT EXTRACTOR

A Streamlit-powered application that converts unstructured PDF text into a clean, structured **Key–Value–Comments** table using a large language model (LLM). The app automatically loads a preferred model (Llama 3.1 8B Instruct) with graceful fallback options, processes your PDF, extracts structured information, and provides a downloadable Excel file.

---

## Features

### 1. Upload Any PDF
- Reads PDF text using PyPDF2.
- Displays extracted raw text for verification.

### 2. LLM-Based Information Extraction
- Uses `meta-llama/Llama-3.1-8B-Instruct` by default.
- Falls back to `HuggingFaceH4/zephyr-7b-beta` and `gpt2` if needed.
- Handles:
  - Key detection and standardization
  - Date normalization (DD-MMM-YYYY or YYYY)
  - Salary formatting (digits only)
  - Multi-value aggregation (semicolon-separated)
  - JSON sanitization and cleanup

### 3. Real-Time Generation UI
- Progress bar and live streamed model output.
- Custom stopping criteria to capture completed JSON arrays.

### 4. Clean Excel Export
- Converts extracted JSON → Pandas DataFrame → Excel.
- One-click download of `extracted_table.xlsx`.

---

## Installation

Clone your repository and install dependencies:

```bash
git clone https://github.com/Hrishi1084/doc-struct-extractor
cd <project-folder>
pip install -r requirements.txt
```
Recommended Python: `3.10.x.` Ensure you have sufficient disk space for model downloads.

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```
First-run notes:

- The model will automatically download and load (may take time, depending on network and disk).
- An offload folder will be created if needed.
- GPU is used if available; otherwise CPU fallback occurs.

App workflow:
1. Upload a PDF.
2. Preview extracted text.
3. Click Run extraction and produce Excel.
4. Download the generated Excel file.

## How It Works (High-Level)

### Model Loading

- Attempts to load in order: `meta-llama/Llama-3.1-8B-Instruct`, `HuggingFaceH4/zephyr-7b-beta`, `gpt2`.
- Uses `bitsandbytes` 8-bit quantization, auto device placement, and offloading based on VRAM.

### Prompted Extraction

- Inserts the full PDF text into a strict prompt template that requires a single JSON array output.
- Each object must contain exactly: `Key`, `Value`, `Comments`.
- Enforces formatting rules (dates, salaries, multi-values) and ordering (personal → professional → education → certifications → skills).

### Streamed Generation

- Token stream is displayed in the UI while generation occurs.
- Custom stopping criteria detect the closing `]` of the JSON array.

### Validation + Output

- The raw model output is cleaned and parsed into JSON.
- Converted to a Pandas DataFrame and exported as an Excel file for download.

## Project Structure

Only these two files are required to run:
```bash
app.py               # Streamlit application (full extraction pipeline)
requirements.txt     # Python dependencies
```

The app creates:

```bash
/offload             # Temporary folder for model offload (auto-created)
```

## System Requirements

### Minimum

- CPU-only environment (slower)
- 8 GB RAM
- Python 3.10.x

### Recommended

- NVIDIA GPU with ≥ 8 GB VRAM (faster model loading and generation)
- CUDA-compatible PyTorch build

Model will run on CPU if no GPU is present; performance will be lower.

## Dependencies

All dependencies are listed in `requirements.txt`. Key packages include:

- `transformers`, `bitsandbytes`, `torch`
- `streamlit`, `PyPDF2`
- `pandas`, `openpyxl`

Install with:

```bash
pip install -r requirements.txt
```

## Output Example

After extraction the DataFrame will look like:

```sql
Key | Value | Comments
----------------------------------------
First Name | John | extracted from text
Last Name  | Doe  | normalized from header
Date of Birth | 15-Mar-1998 | normalized
...
```

You can download the result as `extracted_table.xlsx`.

## Notes

- No PDF contents are persisted by the app; processing is done in memory.
- The app includes JSON cleaning to handle common non-standard LLM outputs.
- Best suited for resume-like or semi-structured personal/business documents.

## License

Provided for educational and research purposes. Modify and extend as needed.

## Generated

README generated on: `22/11/2025`