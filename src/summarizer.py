"""
Automatic Text Summarization Engine
Module: src/summarizer.py

Implements Transformer-based abstractive summarization (BART, FLAN-T5)
and optional local LLM integration (Ollama).
Includes token-aware text chunking, recursive hierarchical summarization,
compression statistics calculation, and memory/device management.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    pypdf = None  # type: ignore
    PYPDF_AVAILABLE = False


# Supported Transformer Model Registry
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "BART (facebook/bart-large-cnn)": {
        "model_id": "facebook/bart-large-cnn",
        "type": "bart",
        "description": "Standard high-performance abstractive summarization model fine-tuned on CNN/DailyMail.",
        "max_input_tokens": 1024,
        "recommended_chunk_tokens": 750,
        "prefix": "",
        "default_min_length": 40,
        "default_max_length": 140,
        "default_num_beams": 4,
        "default_length_penalty": 2.0,
    },
    "FLAN-T5 (google/flan-t5-base)": {
        "model_id": "google/flan-t5-base",
        "type": "t5",
        "description": "Instruction-tuned sequence-to-sequence model capable of zero-shot conditioned summarization.",
        "max_input_tokens": 512,
        "recommended_chunk_tokens": 380,
        "prefix": "summarize: ",
        "default_min_length": 30,
        "default_max_length": 120,
        "default_num_beams": 3,
        "default_length_penalty": 1.5,
    },
}


@dataclass
class SummarizationResult:
    """Stores the generated summary and associated execution metadata."""
    summary_text: str
    model_name: str
    inference_time_sec: float
    original_words: int
    original_chars: int
    summary_words: int
    summary_chars: int
    compression_ratio: float
    reduction_percent: float
    was_chunked: bool = False
    num_chunks: int = 1
    hierarchy_depth: int = 0
    device_used: str = "cpu"
    metadata: Dict[str, Any] = field(default_factory=dict)


def clean_text(text: str) -> str:
    """
    Cleans and normalizes text for summarization.
    - Strips leading and trailing whitespaces
    - Normalizes multiple spaces, tabs, and abnormal newline repetitions
    - Retains normal punctuation and paragraph boundaries
    """
    if not text:
        return ""
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines into max two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple inline spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def calculate_word_and_char_count(text: str) -> Tuple[int, int]:
    """
    Computes word and character count of raw or cleaned text.
    Words are counted by non-whitespace sequences.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return 0, 0
    words = len(cleaned.split())
    chars = len(cleaned)
    return words, chars


def calculate_compression_metrics(
    original_text: str, summary_text: str
) -> Tuple[int, int, int, int, float, float]:
    """
    Calculates document size comparisons:
    Returns:
        (orig_words, orig_chars, sum_words, sum_chars, compression_ratio, reduction_percent)

    - compression_ratio: original_words / summary_words (>= 1.0 represents compression)
    - reduction_percent: % of text length removed, bounded in [0.0, 100.0]
    """
    orig_words, orig_chars = calculate_word_and_char_count(original_text)
    sum_words, sum_chars = calculate_word_and_char_count(summary_text)

    if sum_words == 0:
        compression_ratio = 1.0
        reduction_percent = 0.0
    else:
        compression_ratio = round(orig_words / sum_words, 2)
        reduction_raw = ((orig_words - sum_words) / orig_words) * 100 if orig_words > 0 else 0.0
        reduction_percent = round(max(0.0, min(100.0, reduction_raw)), 1)

    return orig_words, orig_chars, sum_words, sum_chars, compression_ratio, reduction_percent


# =========================================================================
# PDF & Document Processing & Text Extraction
# =========================================================================

class PDFExtractionError(Exception):
    """Base exception for PDF text extraction failures."""
    pass


class PDFEncryptedError(PDFExtractionError):
    """Raised when an uploaded PDF is password-protected or encrypted."""
    pass


class PDFScannedOrEmptyError(PDFExtractionError):
    """Raised when an uploaded PDF has no extractable text layer (e.g., scanned/image-only)."""
    pass


class TextFileExtractionError(Exception):
    """Base exception for text file reading or decoding failures."""
    pass


def extract_text_from_txt(txt_source: Any) -> str:
    """
    Safely reads and decodes plain text from an uploaded file, stream, or bytes.
    Tries multiple encodings ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1').

    Args:
        txt_source: File path, file-like stream (BytesIO, UploadedFile), or raw bytes.

    Returns:
        Cleaned text string.

    Raises:
        TextFileExtractionError: If file is empty, cannot be read, or contains only whitespace.
    """
    raw_bytes: bytes

    if isinstance(txt_source, (bytes, bytearray)):
        raw_bytes = bytes(txt_source)
    elif hasattr(txt_source, "read"):
        content = txt_source.read()
        if hasattr(txt_source, "seek"):
            txt_source.seek(0)
        if isinstance(content, str):
            raw_bytes = content.encode("utf-8")
        elif isinstance(content, (bytes, bytearray)):
            raw_bytes = bytes(content)
        else:
            raise TextFileExtractionError(f"Unsupported stream content type: {type(content)}")
    elif isinstance(txt_source, str):
        try:
            with open(txt_source, "rb") as f:
                raw_bytes = f.read()
        except Exception as exc:
            raise TextFileExtractionError(f"Failed to read file '{txt_source}': {str(exc)}") from exc
    else:
        raise TextFileExtractionError(f"Unsupported text file source type: {type(txt_source)}")

    if not raw_bytes or len(raw_bytes.strip()) == 0:
        raise TextFileExtractionError("The uploaded text file is empty. Please upload a file containing text.")

    # Try common encodings in order
    decoded_text = ""
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            decoded_text = raw_bytes.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if not decoded_text:
        # Fallback with replacement characters
        decoded_text = raw_bytes.decode("utf-8", errors="replace")

    cleaned = clean_text(decoded_text)
    if not cleaned or len(cleaned.split()) == 0:
        raise TextFileExtractionError("The uploaded text file contains only whitespace or unreadable characters.")

    return cleaned


def extract_text_from_upload(upload_source: Any, filename: Optional[str] = None) -> Tuple[str, str]:
    """
    Extracts text from an uploaded file (.pdf or .txt) and returns (extracted_text, metadata_summary).
    """
    fname = filename or getattr(upload_source, "name", "")
    if fname.lower().endswith(".pdf"):
        text, pages = extract_text_from_pdf(upload_source)
        return text, f"{pages} PDF page(s)"
    else:
        text = extract_text_from_txt(upload_source)
        return text, "plain text document"



def extract_text_from_pdf(pdf_source: Any) -> Tuple[str, int]:
    """
    Extracts selectable digital text from an uploaded PDF file, bytes, or binary stream.

    Args:
        pdf_source: File path, file-like stream (BytesIO, UploadedFile), or raw bytes.

    Returns:
        Tuple of (extracted_cleaned_text, total_page_count)

    Raises:
        PDFExtractionError: If file is not a valid PDF or is corrupted.
        PDFEncryptedError: If PDF is encrypted/password-protected.
        PDFScannedOrEmptyError: If PDF contains no extractable digital text layer.
    """
    if not PYPDF_AVAILABLE:
        raise ImportError(
            "The 'pypdf' library is required to extract text from PDF files. Please install it using 'pip install pypdf'."
        )

    import io

    # Normalize source into a readable binary stream
    if isinstance(pdf_source, (bytes, bytearray)):
        stream: io.BytesIO | Any = io.BytesIO(pdf_source)
    elif hasattr(pdf_source, "read"):
        content = pdf_source.read()
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)
        stream = io.BytesIO(content)
    elif isinstance(pdf_source, str):
        stream = open(pdf_source, "rb")
    else:
        raise PDFExtractionError(f"Unsupported PDF source type: {type(pdf_source)}")

    try:
        reader = pypdf.PdfReader(stream)
    except Exception as exc:
        raise PDFExtractionError(
            f"Failed to parse PDF document. The file may be corrupted, truncated, or invalid: {str(exc)}"
        ) from exc

    # Check encryption status
    if reader.is_encrypted:
        try:
            # Check if it opens with empty password (default permissions)
            decrypt_res = reader.decrypt("")
            if decrypt_res == 0:
                raise PDFEncryptedError(
                    "The uploaded PDF document is encrypted or password-protected. Please provide an unencrypted PDF."
                )
        except Exception as enc_err:
            if isinstance(enc_err, PDFEncryptedError):
                raise
            raise PDFEncryptedError(
                "The uploaded PDF document is encrypted or password-protected. Please provide an unencrypted PDF."
            ) from enc_err

    try:
        total_pages = len(reader.pages)
    except Exception as exc:
        raise PDFExtractionError(f"Error accessing PDF pages: {str(exc)}") from exc

    if total_pages == 0:
        raise PDFExtractionError("The uploaded PDF document contains 0 pages.")

    page_texts: List[str] = []
    for page_idx, page in enumerate(reader.pages):
        try:
            ptxt = page.extract_text()
            if ptxt and ptxt.strip():
                page_texts.append(ptxt.strip())
        except Exception:
            # Continue extracting remaining pages if one page encounters an issue
            continue

    combined_text = "\n\n".join(page_texts)
    cleaned = clean_text(combined_text)

    if not cleaned or len(cleaned.split()) == 0:
        raise PDFScannedOrEmptyError(
            f"The uploaded PDF ({total_pages} page{'s' if total_pages > 1 else ''}) contains no extractable digital text. "
            "It appears to be a scanned or image-only document without an embedded text layer (OCR required)."
        )

    return cleaned, total_pages


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into individual sentences using robust regex punctuation splitting.
    Preserves abbreviations, decimal numbers, and quotation marks where possible.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    # Pattern looks for ., !, ? followed by whitespace or end of line, avoiding decimals
    pattern = r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])"
    raw_sentences = re.split(pattern, cleaned)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [cleaned]


def chunk_text_by_tokens(
    text: str,
    tokenizer: Optional[PreTrainedTokenizer],
    max_tokens_per_chunk: int = 750,
) -> List[str]:
    """
    Partitions long text into token-aware chunks that do not exceed max_tokens_per_chunk.
    Preserves sentence boundaries whenever possible. If an individual sentence is longer
    than max_tokens_per_chunk, it falls back to sub-splitting by words.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # If no tokenizer provided, estimate ~1.3 tokens per word
    def estimate_tokens(s: str) -> int:
        if tokenizer is not None:
            return len(tokenizer.encode(s, truncation=False, add_special_tokens=False))
        return int(len(s.split()) * 1.3) + 1

    total_tokens = estimate_tokens(cleaned)
    if total_tokens <= max_tokens_per_chunk:
        return [cleaned]

    sentences = split_into_sentences(cleaned)
    chunks: List[str] = []
    current_chunk_sentences: List[str] = []
    current_chunk_tokens = 0

    for sentence in sentences:
        s_tokens = estimate_tokens(sentence)

        # Handle edge-case: single massive sentence exceeds max chunk size
        if s_tokens > max_tokens_per_chunk:
            # Flush existing chunk
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_chunk_tokens = 0

            # Split large sentence by words
            words = sentence.split()
            sub_words: List[str] = []
            sub_tokens = 0
            for w in words:
                w_tok = estimate_tokens(w + " ")
                if sub_tokens + w_tok > max_tokens_per_chunk and sub_words:
                    chunks.append(" ".join(sub_words))
                    sub_words = [w]
                    sub_tokens = w_tok
                else:
                    sub_words.append(w)
                    sub_tokens += w_tok
            if sub_words:
                chunks.append(" ".join(sub_words))
            continue

        if current_chunk_tokens + s_tokens <= max_tokens_per_chunk:
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += s_tokens
        else:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_chunk_tokens = s_tokens

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return [c.strip() for c in chunks if c.strip()]


def get_device() -> Tuple[str, str]:
    """
    Detects CPU or GPU hardware acceleration availability.
    Returns: (device_str, human_readable_description)
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return "cuda", f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)"
    return "cpu", "CPU (Hardware acceleration disabled / unavailable)"


class TransformerSummarizer:
    """
    Encapsulates Hugging Face Seq2Seq model loading, token-aware chunking,
    and hierarchical abstractive generation.
    """

    def __init__(self, model_key: str = "BART (facebook/bart-large-cnn)"):
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and Hugging Face Transformers are required to run TransformerSummarizer."
            )

        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}"
            )

        self.model_key = model_key
        self.config = MODEL_REGISTRY[model_key]
        self.model_id = self.config["model_id"]
        self.device, self.device_name = get_device()

        # Lazy loading holders
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None

    def load_model(
        self, progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> None:
        """
        Loads the tokenizer and model onto target device.
        Accepts an optional progress callback for UI updates.
        """
        if self.model is not None and self.tokenizer is not None:
            return  # Already loaded

        if progress_callback:
            progress_callback(0.2, f"Downloading/Loading tokenizer for {self.model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if progress_callback:
            progress_callback(0.5, f"Downloading/Loading weights for {self.model_id} onto {self.device}...")

        # Load model with memory-efficient dtype if on CUDA
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        if progress_callback:
            progress_callback(1.0, f"Model {self.model_id} ready on {self.device}!")

    def summarize_chunk(
        self,
        chunk_text: str,
        min_length: int = 40,
        max_length: int = 140,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        prefix: str = "",
    ) -> str:
        """
        Summarizes a single text chunk directly within model input limits.
        Dynamically adjusts min_length/max_length to prevent generation exceptions
        if input chunk is short.
        """
        if not self.model or not self.tokenizer:
            self.load_model()

        assert self.tokenizer is not None
        assert self.model is not None

        prepared_input = f"{prefix}{clean_text(chunk_text)}"
        inputs = self.tokenizer(
            prepared_input,
            max_length=self.config["max_input_tokens"],
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Guard against min_length >= input_length or max_length <= min_length
        safe_max_length = min(max_length, max(30, int(input_length * 0.85)))
        safe_min_length = min(min_length, max(10, int(safe_max_length * 0.4)))
        if safe_min_length >= safe_max_length:
            safe_max_length = safe_min_length + 20

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=max(1, int(num_beams)),
                min_length=safe_min_length,
                max_length=safe_max_length,
                length_penalty=float(length_penalty),
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return clean_text(summary)

    def summarize(
        self,
        text: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        length_penalty: Optional[float] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> SummarizationResult:
        """
        Main summarization pipeline:
        1. Checks input validity
        2. Applies token-aware chunking
        3. Executes hierarchical recursive summarization if document exceeds token limit
        4. Calculates compression metrics and returns SummarizationResult
        """
        cleaned_input = clean_text(text)
        if not cleaned_input:
            raise ValueError("Input document is empty. Please provide text to summarize.")

        # Resolve parameters to config defaults if unspecified
        cfg = self.config
        p_min = min_length if min_length is not None else cfg["default_min_length"]
        p_max = max_length if max_length is not None else cfg["default_max_length"]
        p_beams = num_beams if num_beams is not None else cfg["default_num_beams"]
        p_penalty = length_penalty if length_penalty is not None else cfg["default_length_penalty"]
        prefix = cfg["prefix"]

        start_time = time.time()
        self.load_model(progress_callback=progress_callback)

        assert self.tokenizer is not None

        # Partition into safe token chunks
        chunk_limit = cfg["recommended_chunk_tokens"]
        chunks = chunk_text_by_tokens(
            cleaned_input, tokenizer=self.tokenizer, max_tokens_per_chunk=chunk_limit
        )

        num_chunks = len(chunks)
        was_chunked = num_chunks > 1
        hierarchy_depth = 0

        if not was_chunked:
            if progress_callback:
                progress_callback(0.7, "Summarizing document directly...")
            final_summary = self.summarize_chunk(
                cleaned_input,
                min_length=p_min,
                max_length=p_max,
                num_beams=p_beams,
                length_penalty=p_penalty,
                prefix=prefix,
            )
        else:
            # Hierarchical Summarization with depth safety cap
            max_depth = 3
            current_chunks = chunks
            intermediate_summaries: List[str] = []

            while len(current_chunks) > 1 and hierarchy_depth < max_depth:
                hierarchy_depth += 1
                intermediate_summaries = []
                total_c = len(current_chunks)

                for idx, chunk in enumerate(current_chunks):
                    if progress_callback:
                        pct = 0.3 + 0.5 * (idx / total_c)
                        progress_callback(
                            pct,
                            f"Hierarchical Pass {hierarchy_depth}: Processing chunk {idx + 1}/{total_c}...",
                        )
                    chunk_summary = self.summarize_chunk(
                        chunk,
                        min_length=max(20, int(p_min * 0.7)),
                        max_length=p_max,
                        num_beams=p_beams,
                        length_penalty=p_penalty,
                        prefix=prefix,
                    )
                    intermediate_summaries.append(chunk_summary)

                # Re-check combined intermediate summaries length
                combined_intermediate = " ".join(intermediate_summaries)
                current_chunks = chunk_text_by_tokens(
                    combined_intermediate,
                    tokenizer=self.tokenizer,
                    max_tokens_per_chunk=chunk_limit,
                )

            # Final condensation pass to ensure a cohesive narrative
            if progress_callback:
                progress_callback(0.9, "Generating final consolidated summary...")
            consolidated_input = " ".join(intermediate_summaries) if intermediate_summaries else cleaned_input
            final_summary = self.summarize_chunk(
                consolidated_input,
                min_length=p_min,
                max_length=p_max,
                num_beams=p_beams,
                length_penalty=p_penalty,
                prefix=prefix,
            )

        elapsed = round(time.time() - start_time, 2)
        (
            orig_words,
            orig_chars,
            sum_words,
            sum_chars,
            comp_ratio,
            reduc_pct,
        ) = calculate_compression_metrics(cleaned_input, final_summary)

        return SummarizationResult(
            summary_text=final_summary,
            model_name=self.model_key,
            inference_time_sec=elapsed,
            original_words=orig_words,
            original_chars=orig_chars,
            summary_words=sum_words,
            summary_chars=sum_chars,
            compression_ratio=comp_ratio,
            reduction_percent=reduc_pct,
            was_chunked=was_chunked,
            num_chunks=num_chunks,
            hierarchy_depth=hierarchy_depth,
            device_used=self.device_name,
            metadata={
                "min_length": p_min,
                "max_length": p_max,
                "num_beams": p_beams,
                "length_penalty": p_penalty,
                "model_id": self.model_id,
            },
        )


# =========================================================================
# Ollama Local LLM Integration (Zero API Key, Local Inference)
# =========================================================================

OLLAMA_DEFAULT_URL = "http://localhost:11434"


def check_ollama_status(base_url: str = OLLAMA_DEFAULT_URL) -> Tuple[bool, List[str], str]:
    """
    Checks if a local Ollama daemon is active and queries installed models.
    Returns: (is_available, model_tags_list, status_message)
    """
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=2.5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            if models:
                return True, models, f"Ollama is running with {len(models)} model(s) available."
            return (
                True,
                [],
                "Ollama is running, but no models were found. Run `ollama pull llama3` or `ollama pull mistral` to install a model.",
            )
        return False, [], f"Ollama returned HTTP status {resp.status_code}."
    except requests.exceptions.ConnectionError:
        return (
            False,
            [],
            "Ollama server is not running at localhost:11434. To enable LLM mode, install Ollama from ollama.ai and run `ollama serve`.",
        )
    except Exception as exc:
        return False, [], f"Could not connect to Ollama: {str(exc)}"


def summarize_with_ollama(
    text: str,
    model_name: str,
    base_url: str = OLLAMA_DEFAULT_URL,
    custom_instruction: str = "Provide a concise, comprehensive abstractive summary of the key findings in the following text.",
    temperature: float = 0.2,
) -> SummarizationResult:
    """
    Executes zero-shot abstractive summarization via local Ollama REST API.
    Does not require any external internet connection or paid API key.
    """
    cleaned_input = clean_text(text)
    if not cleaned_input:
        raise ValueError("Input text is empty.")

    start_time = time.time()
    prompt = f"{custom_instruction}\n\nDocument:\n\"\"\"\n{cleaned_input}\n\"\"\"\n\nSummary:"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=120.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

        res_json = resp.json()
        summary_text = clean_text(res_json.get("response", ""))
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to communicate with Ollama: {str(e)}") from e

    elapsed = round(time.time() - start_time, 2)
    (
        orig_words,
        orig_chars,
        sum_words,
        sum_chars,
        comp_ratio,
        reduc_pct,
    ) = calculate_compression_metrics(cleaned_input, summary_text)

    return SummarizationResult(
        summary_text=summary_text,
        model_name=f"Ollama LLM ({model_name})",
        inference_time_sec=elapsed,
        original_words=orig_words,
        original_chars=orig_chars,
        summary_words=sum_words,
        summary_chars=sum_chars,
        compression_ratio=comp_ratio,
        reduction_percent=reduc_pct,
        was_chunked=False,
        num_chunks=1,
        hierarchy_depth=0,
        device_used="Ollama Local Host",
        metadata={"ollama_model": model_name, "temperature": temperature},
    )
