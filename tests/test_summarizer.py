"""
Unit and Integration Tests for Automatic Text Summarization Engine
Module: tests/test_summarizer.py
"""

import pytest

from src.evaluation import (
    calculate_rouge_scores,
    fallback_rouge_score,
    format_rouge_to_dataframe,
)
from src.summarizer import (
    MODEL_REGISTRY,
    PDFEncryptedError,
    PDFExtractionError,
    PDFScannedOrEmptyError,
    SummarizationResult,
    TextFileExtractionError,
    calculate_compression_metrics,
    calculate_word_and_char_count,
    check_ollama_status,
    chunk_text_by_tokens,
    clean_text,
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_upload,
    get_device,
    split_into_sentences,
)


class TestTextPreprocessing:
    """Tests for text normalization, cleaning, and sentence boundary detection."""

    def test_clean_text_normalizes_whitespace(self):
        raw = "  This   is  a\n\n\n\ntest\twith   extra   spaces.  "
        cleaned = clean_text(raw)
        assert cleaned == "This is a\n\ntest with extra spaces."

    def test_clean_text_empty_and_none(self):
        assert clean_text("") == ""
        assert clean_text("   \n\t  ") == ""

    def test_split_into_sentences_basic(self):
        text = "Deep learning is advancing fast. Transformers revolutionized NLP! Can they summarize texts? Yes."
        sentences = split_into_sentences(text)
        assert len(sentences) == 4
        assert sentences[0] == "Deep learning is advancing fast."
        assert sentences[1] == "Transformers revolutionized NLP!"
        assert sentences[2] == "Can they summarize texts?"
        assert sentences[3] == "Yes."

    def test_split_into_sentences_empty(self):
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []


class TestMetricsAndCounts:
    """Tests for word/character counting and compression statistics."""

    def test_calculate_word_and_char_count(self):
        text = "Natural Language Processing with Transformers"
        words, chars = calculate_word_and_char_count(text)
        assert words == 5
        assert chars == len(text)

    def test_calculate_word_and_char_count_empty(self):
        words, chars = calculate_word_and_char_count("")
        assert words == 0
        assert chars == 0

    def test_calculate_compression_metrics_normal(self):
        original = "The quick brown fox jumps over the lazy dog. It is an extraordinary afternoon in the quiet forest."
        summary = "The quick brown fox jumps over the lazy dog."
        (
            orig_w,
            orig_c,
            sum_w,
            sum_c,
            ratio,
            reduction,
        ) = calculate_compression_metrics(original, summary)

        assert orig_w > sum_w
        assert ratio > 1.0
        assert 0.0 < reduction < 100.0

    def test_calculate_compression_metrics_zero_division_safety(self):
        # Empty summary
        orig_w, orig_c, sum_w, sum_c, ratio, reduction = calculate_compression_metrics(
            "Some original text", ""
        )
        assert ratio == 1.0
        assert reduction == 0.0

        # Empty original
        orig_w, orig_c, sum_w, sum_c, ratio, reduction = calculate_compression_metrics(
            "", "Summary"
        )
        assert orig_w == 0
        assert sum_w == 1
        assert reduction == 0.0


class TestTokenAwareChunking:
    """Tests for document partitioning and token limits."""

    def test_chunking_small_document_returns_single_chunk(self):
        short_text = "Transformer models like BART and T5 are widely used for sequence-to-sequence tasks."
        chunks = chunk_text_by_tokens(short_text, tokenizer=None, max_tokens_per_chunk=100)
        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_chunking_large_document_splits_properly(self):
        # Build document with many distinct sentences
        sentences = [
            f"Sentence number {i} provides essential context for the case study on text summarization."
            for i in range(1, 30)
        ]
        large_document = " ".join(sentences)
        chunks = chunk_text_by_tokens(
            large_document, tokenizer=None, max_tokens_per_chunk=50
        )
        assert len(chunks) > 1
        # All sentences must be represented across chunks
        combined_chunks = " ".join(chunks)
        assert "Sentence number 1" in combined_chunks
        assert "Sentence number 29" in combined_chunks

    def test_chunking_massive_sentence_falls_back_to_words(self):
        # Single long sentence with no punctuation
        massive_sentence = " ".join(["word" + str(i) for i in range(200)])
        chunks = chunk_text_by_tokens(
            massive_sentence, tokenizer=None, max_tokens_per_chunk=30
        )
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.split()) <= 35


class TestROUGEEvaluation:
    """Tests for ROUGE-1, ROUGE-2, and ROUGE-L metric computations."""

    def test_rouge_identical_texts_yields_perfect_score(self):
        text = "Transformers utilize self-attention mechanisms to summarize complex academic texts."
        scores = calculate_rouge_scores(text, text)

        for metric in ["rouge1", "rouge2", "rougeL"]:
            assert scores[metric]["precision"] == 1.0
            assert scores[metric]["recall"] == 1.0
            assert scores[metric]["fmeasure"] == 1.0

    def test_rouge_disjoint_texts_yields_zero_score(self):
        cand = "Apples oranges bananas grapes."
        ref = "Quantum computing algorithms execute superposition."
        scores = calculate_rouge_scores(cand, ref)

        for metric in ["rouge1", "rouge2", "rougeL"]:
            assert scores[metric]["fmeasure"] == 0.0

    def test_rouge_partial_overlap(self):
        cand = "The president gave a major speech on economic development."
        ref = "The president addressed the nation concerning economic growth and development."
        scores = calculate_rouge_scores(cand, ref)

        assert 0.0 < scores["rouge1"]["fmeasure"] < 1.0
        assert 0.0 < scores["rougeL"]["fmeasure"] < 1.0

    def test_rouge_empty_input_raises_value_error(self):
        with pytest.raises(ValueError):
            calculate_rouge_scores("", "Some valid reference summary.")

        with pytest.raises(ValueError):
            calculate_rouge_scores("Some valid candidate summary.", "")

    def test_fallback_rouge_score_directly(self):
        cand = "AI models summarize documents."
        ref = "AI models can summarize long documents."
        scores = fallback_rouge_score(cand, ref)
        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores
        assert scores["rouge1"]["fmeasure"] > 0.5

    def test_format_rouge_to_dataframe(self):
        scores = {
            "rouge1": {"precision": 0.8, "recall": 0.75, "fmeasure": 0.774},
            "rouge2": {"precision": 0.6, "recall": 0.55, "fmeasure": 0.574},
            "rougeL": {"precision": 0.7, "recall": 0.65, "fmeasure": 0.674},
        }
        df = format_rouge_to_dataframe(scores, model_label="BART")
        assert df.shape[0] == 3
        assert "Metric" in df.columns
        assert "F1-Score (%)" in df.columns
        assert "Model" in df.columns

    def test_rouge_metric_columns_and_formatting_no_name_error(self):
        """Regression test ensuring ROUGE-1, ROUGE-2, and ROUGE-L columns unpack without NameError."""
        cand = "BART uses bidirectional encoder and autoregressive decoder."
        ref = "BART uses bidirectional encoders and autoregressive decoders for summarization."
        scores = calculate_rouge_scores(cand, ref)

        assert "rouge1" in scores
        assert "rouge2" in scores
        assert "rougeL" in scores

        # Verify formatting into metrics and 3-column container logic without raising NameError
        class DummyCol:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        r1, r2, r3 = DummyCol(), DummyCol(), DummyCol()
        with r1:
            m1 = f"{scores['rouge1']['fmeasure'] * 100:.1f}%"
        with r2:
            m2 = f"{scores['rouge2']['fmeasure'] * 100:.1f}%"
        with r3:
            mL = f"{scores['rougeL']['fmeasure'] * 100:.1f}%"

        assert m1 and m2 and mL
        assert float(m1.replace("%", "")) > 0
        assert float(mL.replace("%", "")) > 0

    def test_app_source_has_no_undefined_r3(self):
        """Regression test ensuring app.py has no undefined r3 references causing NameError."""
        import ast

        with open("app.py", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        with_r3_found = False
        r3_assigned = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name) and elt.id == "r3":
                                r3_assigned = True
                    elif isinstance(target, ast.Name) and target.id == "r3":
                        r3_assigned = True
            elif isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Name) and item.context_expr.id == "r3":
                        with_r3_found = True

        assert r3_assigned is True, "Expected 'r3' to be assigned in app.py (e.g. r1, r2, r3 = st.columns(3))"
        assert with_r3_found is True, "Expected 'with r3:' context to be present for ROUGE-L metric display"


class TestModelRegistryAndEnvironment:
    """Tests registry configurations, device detection, and Ollama check."""

    def test_model_registry_contains_bart_and_flan_t5(self):
        assert "BART (facebook/bart-large-cnn)" in MODEL_REGISTRY
        assert "FLAN-T5 (google/flan-t5-base)" in MODEL_REGISTRY

        bart_cfg = MODEL_REGISTRY["BART (facebook/bart-large-cnn)"]
        assert bart_cfg["max_input_tokens"] == 1024
        assert bart_cfg["recommended_chunk_tokens"] < 1024
        assert bart_cfg["model_id"] == "facebook/bart-large-cnn"

        t5_cfg = MODEL_REGISTRY["FLAN-T5 (google/flan-t5-base)"]
        assert t5_cfg["max_input_tokens"] == 512
        assert t5_cfg["prefix"] == "summarize: "

    def test_get_device_returns_valid_tuple(self):
        device, desc = get_device()
        assert device in ["cpu", "cuda"]
        assert isinstance(desc, str) and len(desc) > 0

    def test_check_ollama_status_signature(self):
        is_avail, models, msg = check_ollama_status("http://127.0.0.1:9999")
        # Invalid port should cleanly return False without unhandled exception
        assert is_avail is False
        assert isinstance(models, list)
        assert isinstance(msg, str)

    def test_summarization_result_dataclass(self):
        res = SummarizationResult(
            summary_text="Summarized text",
            model_name="TestModel",
            inference_time_sec=1.23,
            original_words=100,
            original_chars=600,
            summary_words=20,
            summary_chars=120,
            compression_ratio=5.0,
            reduction_percent=80.0,
            was_chunked=True,
            num_chunks=2,
            hierarchy_depth=1,
            device_used="cpu",
        )
        assert res.compression_ratio == 5.0
        assert res.was_chunked is True
        assert res.num_chunks == 2


class TestPDFExtraction:
    """Tests for PDF text extraction, encryption detection, and error handling."""

    @staticmethod
    def _create_sample_pdf(text: str = "Artificial Intelligence in NLP") -> bytes:
        """Generates a valid minimal in-memory PDF containing text."""
        import io
        import pypdf

        pdf_bytes = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length {len(text) + 40} >> stream
BT /F1 12 Tf 20 250 Td ({text}) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000350 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
424
%%EOF""".encode("latin1")

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        writer = pypdf.PdfWriter()
        writer.append(reader)
        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()

    def test_extract_text_from_valid_pdf_bytes(self):
        pdf_data = self._create_sample_pdf("Transformers revolutionize sequence summarization.")
        extracted, pages = extract_text_from_pdf(pdf_data)
        assert pages == 1
        assert "Transformers revolutionize sequence summarization." in extracted

    def test_extract_text_from_file_like_stream(self):
        import io

        pdf_data = self._create_sample_pdf("Deep learning methods distill academic texts.")
        stream = io.BytesIO(pdf_data)
        extracted, pages = extract_text_from_pdf(stream)
        assert pages == 1
        assert "Deep learning methods distill academic texts." in extracted

    def test_extract_text_from_encrypted_pdf_raises_error(self):
        import io
        import pypdf

        pdf_data = self._create_sample_pdf("Secret confidential document content.")
        reader = pypdf.PdfReader(io.BytesIO(pdf_data))
        writer = pypdf.PdfWriter()
        writer.append(reader)
        writer.encrypt("secure_password_123")
        enc_buf = io.BytesIO()
        writer.write(enc_buf)
        enc_bytes = enc_buf.getvalue()

        with pytest.raises(PDFEncryptedError) as exc_info:
            extract_text_from_pdf(enc_bytes)
        assert "encrypted or password-protected" in str(exc_info.value).lower()

    def test_extract_text_from_blank_or_scanned_pdf_raises_error(self):
        import io
        import pypdf

        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        writer.write(buf)
        blank_bytes = buf.getvalue()

        with pytest.raises(PDFScannedOrEmptyError) as exc_info:
            extract_text_from_pdf(blank_bytes)
        assert "no extractable digital text" in str(exc_info.value).lower()

    def test_extract_text_from_corrupted_pdf_raises_error(self):
        corrupted_bytes = b"This is completely arbitrary text, definitely not a PDF document."
        with pytest.raises(PDFExtractionError):
            extract_text_from_pdf(corrupted_bytes)

    def test_extract_text_from_unsupported_type_raises_error(self):
        with pytest.raises(PDFExtractionError):
            extract_text_from_pdf(12345)


class TestDocumentUploadWorkflow:
    """Tests for TXT extraction, unified upload processing, empty document handling, and summarization pipeline integration."""

    def test_extract_text_from_valid_txt_bytes(self):
        txt_bytes = b"Natural language processing allows computers to understand and generate text."
        extracted = extract_text_from_txt(txt_bytes)
        assert "Natural language processing allows computers" in extracted

    def test_extract_text_from_txt_stream(self):
        import io

        stream = io.BytesIO(
            "BART and FLAN-T5 are popular sequence-to-sequence transformer architectures.".encode("utf-8")
        )
        extracted = extract_text_from_txt(stream)
        assert "BART and FLAN-T5" in extracted

    def test_extract_text_from_txt_encodings(self):
        # UTF-8 with BOM
        utf8_sig_data = "Modern summarization models condense long papers.".encode("utf-8-sig")
        extracted_sig = extract_text_from_txt(utf8_sig_data)
        assert "Modern summarization models condense long papers." in extracted_sig

        # Latin-1 / CP1252 with special characters
        latin1_data = "Caf\xe9 and na\xefve summarization approaches are evaluated.".encode("latin-1")
        extracted_latin1 = extract_text_from_txt(latin1_data)
        assert "Caf" in extracted_latin1

    def test_extract_text_from_empty_txt_raises_error(self):
        with pytest.raises(TextFileExtractionError) as exc_info:
            extract_text_from_txt(b"")
        assert "empty" in str(exc_info.value).lower()

    def test_extract_text_from_whitespace_txt_raises_error(self):
        with pytest.raises(TextFileExtractionError) as exc_info:
            extract_text_from_txt(b"   \n\t   \n  ")
        assert "whitespace" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()

    def test_extract_text_from_unsupported_txt_type_raises_error(self):
        with pytest.raises(TextFileExtractionError):
            extract_text_from_txt(99999)

    def test_unified_extract_text_from_upload_pdf(self):
        import io

        pdf_bytes = TestPDFExtraction._create_sample_pdf("Automatic text summarization using neural attention.")
        stream = io.BytesIO(pdf_bytes)
        stream.name = "research_paper.pdf"
        text, info = extract_text_from_upload(stream)
        assert "Automatic text summarization" in text
        assert "PDF page" in info

    def test_unified_extract_text_from_upload_txt(self):
        import io

        txt_bytes = b"Abstractive summarization creates novel sentences reflecting the core message of the source."
        stream = io.BytesIO(txt_bytes)
        stream.name = "abstract.txt"
        text, info = extract_text_from_upload(stream)
        assert "Abstractive summarization creates novel sentences" in text
        assert "plain text" in info

    def test_summary_pipeline_with_extracted_upload_text(self):
        """Validates that extracted upload text passes correctly through the summarization pipeline."""
        import io

        pdf_bytes = TestPDFExtraction._create_sample_pdf(
            "State of the art transformer models like BART utilize bidirectional encoders and autoregressive decoders. "
            "They achieve competitive performance on benchmarks such as CNN/DailyMail and XSum."
        )
        stream = io.BytesIO(pdf_bytes)
        stream.name = "case_study.pdf"
        extracted_text, _ = extract_text_from_upload(stream)

        # Ensure extracted text is clean and non-empty
        cleaned_text = clean_text(extracted_text)
        words, chars = calculate_word_and_char_count(cleaned_text)
        assert words >= 15
        assert chars > 0

        # Simulate summarizer pipeline execution with the extracted text
        simulated_summary = (
            "BART transformers use bidirectional encoders and achieve competitive benchmark performance."
        )
        (
            orig_w,
            orig_c,
            sum_w,
            sum_c,
            ratio,
            reduction,
        ) = calculate_compression_metrics(cleaned_text, simulated_summary)

        result = SummarizationResult(
            summary_text=simulated_summary,
            model_name="BART (facebook/bart-large-cnn)",
            inference_time_sec=1.1,
            original_words=orig_w,
            original_chars=orig_c,
            summary_words=sum_w,
            summary_chars=sum_c,
            compression_ratio=ratio,
            reduction_percent=reduction,
        )

        assert result.original_words == words
        assert result.summary_words < result.original_words
        assert result.compression_ratio > 1.0
        assert 0.0 < result.reduction_percent <= 100.0
        assert "BART" in result.model_name


