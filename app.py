"""
Automatic Text Summarization Using Transformer-Based Models and LLMs
Streamlit Web Application: AI Text Summarization Studio
File: app.py
"""

from __future__ import annotations

import io
import time
from typing import Optional

import pandas as pd
import streamlit as st

from src.evaluation import (
    calculate_rouge_scores,
    format_rouge_to_dataframe,
    run_head_to_head_comparison,
)
from src.summarizer import (
    MODEL_REGISTRY,
    PDFEncryptedError,
    PDFExtractionError,
    PDFScannedOrEmptyError,
    SummarizationResult,
    TextFileExtractionError,
    TransformerSummarizer,
    calculate_compression_metrics,
    calculate_word_and_char_count,
    check_ollama_status,
    clean_text,
    extract_text_from_pdf,
    extract_text_from_txt,
    get_device,
    summarize_with_ollama,
)

# =========================================================================
# Page Configuration & Styling
# =========================================================================

st.set_page_config(
    page_title="AI Text Summarization Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a modern dashboard
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: #f8fafc;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.1rem;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    .stat-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .stat-badge-blue { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .stat-badge-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .stat-badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }

    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }

    .output-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }

    .disclaimer-banner {
        background: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 4px solid #f59e0b;
        color: #92400e;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================================
# Sample Demonstration Articles
# =========================================================================

SAMPLE_ARTICLES = {
    "AI & Transformer Models Case Study": {
        "text": (
            "The evolution of deep learning has revolutionized natural language processing (NLP), "
            "most notably through the breakthrough introduction of the Transformer architecture by Vaswani et al. in 2017. "
            "Traditional recurrent neural networks (RNNs) and Long Short-Term Memory networks (LSTMs) processed sequences sequentially, "
            "which created severe computational bottlenecks and made learning long-range contextual dependencies difficult. "
            "Transformers eliminated sequential recurrence entirely by relying on multi-head self-attention mechanisms. "
            "This self-attention mechanism enables the model to dynamically compute interaction weights between every pair of words "
            "in a sequence simultaneously, unlocking massive parallel computation on modern GPU accelerators.\n\n"
            "In automatic text summarization, two primary paradigms exist: extractive and abstractive summarization. "
            "Extractive summarization identifies and extracts key sentences directly from the source text without alteration. "
            "In contrast, abstractive summarization interprets the core concepts and synthesizes entirely new sentences, "
            "closely resembling human paraphrasing and synthesis. Leading seq2seq models like BART (Bidirectional and Auto-Regressive Transformers) "
            "and FLAN-T5 (Fine-tuned Language Net - Text-to-Text Transfer Transformer) excel at abstractive summarization. "
            "BART utilizes a bidirectional encoder paired with an autoregressive decoder, trained as a denoising autoencoder where documents "
            "are intentionally corrupted by masking tokens and shuffling sentences. Meanwhile, FLAN-T5 leverages instruction tuning across thousands "
            "of tasks, enabling high zero-shot performance across varied summarization prompts.\n\n"
            "Evaluating abstractive summaries presents unique challenges because different words can convey identical semantic meaning. "
            "The ROUGE (Recall-Oriented Understudy for Gisting Evaluation) framework serves as the standard academic benchmark. "
            "ROUGE-1 measures unigram lexical overlap, ROUGE-2 measures bigram syntactic fluency, and ROUGE-L measures the longest common "
            "subsequence to capture sentence-level structure. While ROUGE metrics require an authoritative ground-truth reference summary, "
            "they remain the empirical standard for evaluating summarization quality in scholarly research."
        ),
        "reference": (
            "Transformers revolutionized natural language processing by replacing recurrent networks with multi-head self-attention, "
            "enabling parallel processing and superior long-range dependency capture. Abstractive summarization synthesizes novel sentences "
            "to summarize content, exemplified by sequence-to-sequence architectures like BART and instruction-tuned FLAN-T5. "
            "Summarization quality is empirically benchmarked using ROUGE metrics against reference summaries."
        ),
    },
    "Climate Change and Renewable Energy Transition": {
        "text": (
            "Global energy infrastructure is undergoing a historic structural shift as nations transition away from fossil fuel combustion "
            "toward renewable energy sources. Driven by accelerating climate change and severe weather events, governments worldwide are "
            "investing trillions into solar photovoltaics, offshore wind installations, and next-generation battery storage systems. "
            "Solar and wind power have achieved dramatic reductions in levelized cost of energy (LCOE) over the last decade, often outcompeting "
            "coal and natural gas even without direct state subsidies.\n\n"
            "However, integrating intermittent renewable sources into legacy electrical grids introduces complex technical hurdles. "
            "Grid operators must balance supply and demand instantaneously. Without continuous inertia from spinning thermal generators, "
            "modern power grids require grid-forming inverters, utility-scale lithium-ion battery storage facilities, and pumped hydroelectric storage. "
            "Furthermore, cross-border high-voltage direct current (HVDC) transmission lines are crucial for wheeling power from remote generation sites "
            "to urban population centers. Successful decarbonization requires not only clean power generation, but also grid modernization, "
            "regulatory market reform, and long-duration energy storage technologies."
        ),
        "reference": (
            "The global transition to renewable energy is accelerating due to climate imperatives and falling costs for solar and wind. "
            "However, integrating variable clean energy requires modernizing legacy grids with energy storage systems, advanced inverters, "
            "and long-distance transmission infrastructure."
        ),
    },
}

# =========================================================================
# Model Caching via Streamlit
# =========================================================================

@st.cache_resource(show_spinner=False)
def get_cached_summarizer(model_key: str) -> TransformerSummarizer:
    """Loads and caches TransformerSummarizer instance per model key."""
    summarizer = TransformerSummarizer(model_key=model_key)
    summarizer.load_model()
    return summarizer


# =========================================================================
# Header & Navigation
# =========================================================================

device_str, device_desc = get_device()
ollama_avail, ollama_models, ollama_msg = check_ollama_status()

st.markdown(
    f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1>✨ AI Text Summarization Studio</h1>
                <p>Automatic Text Summarization using Pretrained Transformers (BART, FLAN-T5) & LLMs for Academic NLP Research</p>
            </div>
            <div style="margin-top: 0.5rem;">
                <span class="stat-badge stat-badge-blue">⚡ Hardware: {device_str.upper()}</span>
                <span class="stat-badge {'stat-badge-green' if ollama_avail else 'stat-badge-amber'}">
                    {'🦙 Ollama Online' if ollama_avail else '🦙 Ollama Offline'}
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_studio, tab_compare, tab_case_study = st.tabs(
    [
        "⚡ Summarization Studio",
        "⚖️ Model Benchmark & Evaluation",
        "📚 NLP Case Study & Viva Voce",
    ]
)

# =========================================================================
# Sidebar Controls & System Diagnostics
# =========================================================================

with st.sidebar:
    st.subheader("⚙️ Model Configuration")

    model_options = list(MODEL_REGISTRY.keys())
    if ollama_avail and ollama_models:
        model_options.append("Ollama LLM (Local)")

    selected_model = st.selectbox(
        "Select Summarization Engine:",
        options=model_options,
        index=0,
        help="Choose between sequence-to-sequence Transformers or local LLMs.",
    )

    is_ollama_selected = selected_model == "Ollama LLM (Local)"

    if is_ollama_selected:
        selected_ollama_model = st.selectbox(
            "Installed Ollama Model:",
            options=ollama_models,
            help="Models discovered locally from Ollama repository.",
        )
        ollama_temp = st.slider("Temperature:", 0.0, 1.0, 0.2, 0.05)
    else:
        cfg = MODEL_REGISTRY[selected_model]
        st.caption(f"**Model ID:** `{cfg['model_id']}`")
        st.caption(f"**Max Input Tokens:** `{cfg['max_input_tokens']}` | **Safe Chunk:** `{cfg['recommended_chunk_tokens']}`")

        # Summary Length Presets
        preset = st.radio(
            "Summary Length Preset:",
            options=["Concise / Executive", "Balanced (Recommended)", "Detailed / In-Depth", "Custom"],
            index=1,
            horizontal=True,
        )

        if preset == "Concise / Executive":
            min_len = 25
            max_len = 70
            n_beams = 3
            len_penalty = 1.8
        elif preset == "Balanced (Recommended)":
            min_len = cfg["default_min_length"]
            max_len = cfg["default_max_length"]
            n_beams = cfg["default_num_beams"]
            len_penalty = cfg["default_length_penalty"]
        elif preset == "Detailed / In-Depth":
            min_len = 90
            max_len = 220
            n_beams = 4
            len_penalty = 1.0
        else:  # Custom
            with st.expander("🛠️ Advanced Hyperparameters", expanded=True):
                min_len = st.slider("Minimum Length (Tokens):", 10, 150, cfg["default_min_length"])
                max_len = st.slider("Maximum Length (Tokens):", 40, 400, cfg["default_max_length"])
                n_beams = st.slider("Beam Search Size (num_beams):", 1, 6, cfg["default_num_beams"])
                len_penalty = st.slider("Length Penalty:", 0.5, 3.0, cfg["default_length_penalty"], 0.1)

    st.markdown("---")
    st.subheader("🛡️ Safe Long-Document Chunking")
    st.info(
        "**Hierarchical Chunking Active:**\n"
        "- Long documents are split into coherent token-aware chunks.\n"
        "- Intermediate summaries are recursively condensed (depth cap = 3).\n"
        "- Guarantees no token overflow or infinite recursion."
    )

    st.markdown("---")
    st.subheader("💻 System Diagnostics")
    st.caption(f"**Device Acceleration:** {device_desc}")
    st.caption(f"**PyTorch:** Active")
    st.caption(f"**Hugging Face Transformers:** Active")
    if not ollama_avail:
        with st.expander("ℹ️ How to enable Ollama LLM mode"):
            st.markdown(
                """
                1. Install Ollama from [ollama.ai](https://ollama.ai)
                2. Run `ollama serve` in a terminal
                3. Download a model: `ollama run llama3` or `ollama run mistral`
                4. Refresh this page to activate the LLM option!
                """
            )


# =========================================================================
# TAB 1: Summarization Studio
# =========================================================================

with tab_studio:
    col_input, col_output = st.columns([1, 1], gap="large")

    # Manage input state
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    if "text_area_input" not in st.session_state:
        st.session_state["text_area_input"] = ""
    if "ref_text" not in st.session_state:
        st.session_state["ref_text"] = ""
    if "text_area_ref" not in st.session_state:
        st.session_state["text_area_ref"] = ""
    if "_last_uploaded_file_id" not in st.session_state:
        st.session_state["_last_uploaded_file_id"] = None

    with col_input:
        st.subheader("📄 Input Document")

        # Sample Article Loader
        col_sample_sel, col_sample_btn = st.columns([3, 1])
        with col_sample_sel:
            selected_sample_title = st.selectbox(
                "Load academic sample article:",
                options=list(SAMPLE_ARTICLES.keys()),
                label_visibility="collapsed",
            )
        with col_sample_btn:
            if st.button("📥 Load Sample", use_container_width=True):
                sample_text = SAMPLE_ARTICLES[selected_sample_title]["text"]
                sample_ref = SAMPLE_ARTICLES[selected_sample_title]["reference"]
                st.session_state["input_text"] = sample_text
                st.session_state["text_area_input"] = sample_text
                st.session_state["ref_text"] = sample_ref
                st.session_state["text_area_ref"] = sample_ref
                st.session_state["_last_uploaded_file_id"] = None
                st.rerun()

        # Upload .txt or .pdf file
        uploaded_file = st.file_uploader(
            "Or upload a document (.txt or .pdf):",
            type=["txt", "pdf"],
            help="Upload an academic paper, news article, or research PDF.",
            key="file_uploader_input",
        )
        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("_last_uploaded_file_id") != file_id:
                try:
                    if uploaded_file.name.lower().endswith(".pdf"):
                        with st.spinner("Extracting text from PDF document..."):
                            extracted_txt, page_count = extract_text_from_pdf(uploaded_file)
                        st.session_state["input_text"] = extracted_txt
                        st.session_state["text_area_input"] = extracted_txt
                        st.session_state["_last_uploaded_file_id"] = file_id
                        st.toast(f"Extracted {len(extracted_txt.split())} words across {page_count} PDF page(s)!", icon="📄")
                        st.rerun()
                    else:
                        with st.spinner("Reading uploaded text document..."):
                            extracted_txt = extract_text_from_txt(uploaded_file)
                        st.session_state["input_text"] = extracted_txt
                        st.session_state["text_area_input"] = extracted_txt
                        st.session_state["_last_uploaded_file_id"] = file_id
                        st.toast(f"Loaded '{uploaded_file.name}' ({len(extracted_txt.split())} words) successfully!", icon="📝")
                        st.rerun()
                except PDFEncryptedError as enc_err:
                    st.error(f"🔒 {str(enc_err)}")
                except PDFScannedOrEmptyError as scan_err:
                    st.error(f"📷 Scanned or Image-Only PDF: {str(scan_err)}")
                except PDFExtractionError as pdf_err:
                    st.error(f"❌ PDF Processing Error: {str(pdf_err)}")
                except TextFileExtractionError as txt_err:
                    st.error(f"❌ Text File Error: {str(txt_err)}")
                except Exception as e:
                    st.error(f"❌ Error reading uploaded file: {str(e)}")
        else:
            if st.session_state.get("_last_uploaded_file_id") is not None:
                st.session_state["_last_uploaded_file_id"] = None

        # Text input area
        user_text = st.text_area(
            "Paste or edit document text below:",
            height=280,
            placeholder="Type or paste the document you wish to summarize here...",
            key="text_area_input",
        )
        st.session_state["input_text"] = user_text

        # Real-time input counts
        orig_words, orig_chars = calculate_word_and_char_count(user_text)
        est_tokens = int(orig_words * 1.3) if orig_words > 0 else 0

        st.markdown(
            f"""
            <div style="display: flex; gap: 0.5rem; margin-top: -0.5rem; margin-bottom: 1rem;">
                <span class="stat-badge stat-badge-blue">📝 Words: {orig_words}</span>
                <span class="stat-badge stat-badge-blue">🔤 Characters: {orig_chars}</span>
                <span class="stat-badge stat-badge-amber">🪙 Est. Tokens: ~{est_tokens}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Optional Reference Summary for ROUGE Evaluation
        with st.expander("🎯 Ground-Truth Reference Summary (Optional for ROUGE)", expanded=bool(st.session_state.get("ref_text"))):
            st.caption(
                "Enter a human-written or authoritative summary. If provided, ROUGE-1, ROUGE-2, "
                "and ROUGE-L will be calculated automatically."
            )
            ref_input = st.text_area(
                "Reference Summary:",
                height=110,
                placeholder="Paste reference summary here to compute empirical ROUGE scores...",
                key="text_area_ref",
            )
            st.session_state["ref_text"] = ref_input

        # Action Buttons
        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            summarize_clicked = st.button("🚀 Generate Abstractive Summary", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🧹 Clear Input", use_container_width=True):
                st.session_state["input_text"] = ""
                st.session_state["text_area_input"] = ""
                st.session_state["ref_text"] = ""
                st.session_state["text_area_ref"] = ""
                st.session_state["_last_uploaded_file_id"] = None
                if "last_result" in st.session_state:
                    del st.session_state["last_result"]
                st.rerun()

    # Summarization Execution & Output
    with col_output:
        st.subheader("✨ Generated Summary")

        if summarize_clicked:
            raw_input = st.session_state.get("text_area_input") or st.session_state.get("input_text", "")
            cleaned_input = clean_text(raw_input)
            if not cleaned_input:
                st.warning("⚠️ Please provide or paste input text before generating a summary.")
            elif len(cleaned_input.split()) < 15:
                st.warning("⚠️ Input document is very short (< 15 words). Please provide more context for abstractive summarization.")
            else:
                progress_bar = st.progress(0.1)
                status_text = st.empty()

                def update_progress(p: float, msg: str):
                    progress_bar.progress(min(1.0, max(0.0, p)))
                    status_text.text(f"⏳ {msg}")

                try:
                    if is_ollama_selected:
                        update_progress(0.4, f"Sending prompt to Ollama ({selected_ollama_model})...")
                        res = summarize_with_ollama(
                            cleaned_input,
                            model_name=selected_ollama_model,
                            temperature=ollama_temp,
                        )
                    else:
                        update_progress(0.3, f"Initializing {selected_model}...")
                        summarizer = get_cached_summarizer(selected_model)
                        res = summarizer.summarize(
                            cleaned_input,
                            min_length=min_len,
                            max_length=max_len,
                            num_beams=n_beams,
                            length_penalty=len_penalty,
                            progress_callback=update_progress,
                        )

                    update_progress(1.0, "Summary generation complete!")
                    time.sleep(0.3)
                    progress_bar.empty()
                    status_text.empty()
                    st.session_state["last_result"] = res

                except Exception as exc:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Summarization failed: {str(exc)}")
                    st.info("💡 Tip: If running on a system with limited RAM, try the 'Concise' preset or restart the app.")

        # Display Result if available
        if "last_result" in st.session_state:
            res: SummarizationResult = st.session_state["last_result"]

            # Output Text Card
            st.markdown(
                f"""
                <div class="output-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <span style="font-weight: 700; color: #1e293b; font-size: 1rem;">
                            Model: {res.model_name}
                        </span>
                        <span>
                            {'<span class="stat-badge stat-badge-amber">🧩 Chunked (' + str(res.num_chunks) + ' parts)</span>' if res.was_chunked else '<span class="stat-badge stat-badge-green">⚡ Direct Pass</span>'}
                        </span>
                    </div>
                    <p style="color: #334155; line-height: 1.65; font-size: 0.98rem; margin: 0;">
                        {res.summary_text}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Metrics Row
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(
                    f"""<div class="metric-card"><div class="metric-title">Summary Words</div><div class="metric-val">{res.summary_words}</div></div>""",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""<div class="metric-card"><div class="metric-title">Compression</div><div class="metric-val">{res.compression_ratio}x</div></div>""",
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""<div class="metric-card"><div class="metric-title">Reduction</div><div class="metric-val">{res.reduction_percent}%</div></div>""",
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""<div class="metric-card"><div class="metric-title">Inference Time</div><div class="metric-val">{res.inference_time_sec}s</div></div>""",
                    unsafe_allow_html=True,
                )
            with m5:
                st.markdown(
                    f"""<div class="metric-card"><div class="metric-title">Chunks</div><div class="metric-val">{res.num_chunks}</div></div>""",
                    unsafe_allow_html=True,
                )

            # Download Button
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            summary_bytes = io.BytesIO(res.summary_text.encode("utf-8"))
            st.download_button(
                label="💾 Download Summary as .txt",
                data=summary_bytes,
                file_name=f"summary_{res.model_name.split()[0].lower()}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            # ROUGE Calculation if reference is supplied
            ref_clean = clean_text(st.session_state.get("ref_text", ""))
            if ref_clean:
                st.markdown("---")
                st.subheader("🎯 Empirical ROUGE Evaluation (vs Reference)")
                try:
                    rouge_scores = calculate_rouge_scores(res.summary_text, ref_clean)
                    df_rouge = format_rouge_to_dataframe(rouge_scores, model_label=res.model_name)
                    st.dataframe(df_rouge, use_container_width=True, hide_index=True)

                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.metric("ROUGE-1 (Unigram F1)", f"{rouge_scores['rouge1']['fmeasure']*100:.1f}%")
                    with r2:
                        st.metric("ROUGE-2 (Bigram F1)", f"{rouge_scores['rouge2']['fmeasure']*100:.1f}%")
                    with r3:
                        st.metric("ROUGE-L (LCS F1)", f"{rouge_scores['rougeL']['fmeasure']*100:.1f}%")
                except Exception as e:
                    st.warning(f"Could not compute ROUGE: {str(e)}")
            else:
                st.info(
                    "💡 **Note on ROUGE:** To calculate ROUGE-1, ROUGE-2, and ROUGE-L scores, "
                    "expand the 'Ground-Truth Reference Summary' section on the left and enter an authoritative summary."
                )

        else:
            st.markdown(
                """
                <div style="background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 3rem 1.5rem; text-align: center; color: #64748b;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                    <div style="font-weight: 600; color: #334155;">No summary generated yet</div>
                    <div style="font-size: 0.9rem; margin-top: 0.25rem;">Enter text or load a sample article on the left, then click <b>'Generate Abstractive Summary'</b>.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Academic Human Review Disclaimer
    st.markdown(
        """
        <div class="disclaimer-banner">
            ⚠️ <b>Academic NLP Disclaimer:</b> Pretrained neural sequence-to-sequence models generate abstractive summaries probabilistically.
            While highly coherent, generated outputs may occasionally omit key nuances, hallucinate minor facts, or reflect dataset biases.
            Always cross-verify summaries with original primary sources for critical, legal, or medical applications.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================
# TAB 2: Model Benchmark & Evaluation
# =========================================================================

with tab_compare:
    st.subheader("⚖️ Head-to-Head Transformer Benchmark")
    st.markdown(
        "Run actual inference simultaneously using **BART (facebook/bart-large-cnn)** and "
        "**FLAN-T5 (google/flan-t5-base)** on the same input text to compare empirical performance, "
        "inference latency, compression efficiency, and ROUGE scores."
    )

    comp_input = st.text_area(
        "Benchmark Document:",
        value=st.session_state.get("input_text", SAMPLE_ARTICLES["AI & Transformer Models Case Study"]["text"]),
        height=180,
        key="benchmark_input_area",
    )

    comp_ref = st.text_area(
        "Ground Truth Reference Summary (Optional, required for ROUGE):",
        value=st.session_state.get("ref_text", SAMPLE_ARTICLES["AI & Transformer Models Case Study"]["reference"]),
        height=90,
        key="benchmark_ref_area",
    )

    if st.button("🏁 Run Head-to-Head Benchmark", type="primary"):
        cleaned_b_input = clean_text(comp_input)
        if not cleaned_b_input:
            st.warning("Please provide benchmark document text.")
        else:
            prog_bench = st.progress(0.1)
            status_bench = st.empty()

            def update_bench_progress(p: float, msg: str):
                prog_bench.progress(min(1.0, max(0.0, p)))
                status_bench.text(f"⏳ {msg}")

            try:
                benchmark_data = run_head_to_head_comparison(
                    cleaned_b_input,
                    models=["BART (facebook/bart-large-cnn)", "FLAN-T5 (google/flan-t5-base)"],
                    reference_summary=clean_text(comp_ref) if comp_ref else None,
                    progress_callback=update_bench_progress,
                )
                prog_bench.empty()
                status_bench.empty()

                st.success("✅ Head-to-head benchmark completed successfully!")

                # Benchmark Summary Table
                st.subheader("📊 Empirical Performance Comparison Table")
                st.dataframe(benchmark_data["dataframe"], use_container_width=True, hide_index=True)

                # Side-by-side Summaries
                st.subheader("📑 Side-by-Side Generated Outputs")
                c_bart, c_t5 = st.columns(2)
                with c_bart:
                    b_res = benchmark_data["results"]["BART (facebook/bart-large-cnn)"]
                    st.markdown(
                        f"""
                        <div class="output-card" style="border-left-color: #3b82f6;">
                            <b>BART (facebook/bart-large-cnn)</b>
                            <p style="margin-top: 0.5rem; color: #334155; font-size: 0.95rem;">{b_res.summary_text}</p>
                            <small style="color: #64748b;">Time: {b_res.inference_time_sec}s | Words: {b_res.summary_words} | Reduction: {b_res.reduction_percent}%</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with c_t5:
                    t_res = benchmark_data["results"]["FLAN-T5 (google/flan-t5-base)"]
                    st.markdown(
                        f"""
                        <div class="output-card" style="border-left-color: #10b981;">
                            <b>FLAN-T5 (google/flan-t5-base)</b>
                            <p style="margin-top: 0.5rem; color: #334155; font-size: 0.95rem;">{t_res.summary_text}</p>
                            <small style="color: #64748b;">Time: {t_res.inference_time_sec}s | Words: {t_res.summary_words} | Reduction: {t_res.reduction_percent}%</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # ROUGE Visual Comparison if Reference Provided
                if benchmark_data["has_reference"]:
                    st.subheader("📈 ROUGE Scores Visual Benchmark")
                    r_evals = benchmark_data["rouge_evaluations"]

                    chart_rows = []
                    for m_name, scores in r_evals.items():
                        short_name = "BART" if "bart" in m_name.lower() else "FLAN-T5"
                        chart_rows.append({"Model": short_name, "Metric": "ROUGE-1 F1", "Score (%)": scores["rouge1"]["fmeasure"] * 100})
                        chart_rows.append({"Model": short_name, "Metric": "ROUGE-2 F1", "Score (%)": scores["rouge2"]["fmeasure"] * 100})
                        chart_rows.append({"Model": short_name, "Metric": "ROUGE-L F1", "Score (%)": scores["rougeL"]["fmeasure"] * 100})

                    df_chart = pd.DataFrame(chart_rows)
                    st.bar_chart(df_chart, x="Metric", y="Score (%)", color="Model")
                else:
                    st.info(
                        "ℹ️ **Ground truth reference summary was not supplied.** ROUGE metrics were not calculated. "
                        "Supply a reference summary above to benchmark lexical and syntactic overlap."
                    )

            except Exception as ex:
                prog_bench.empty()
                status_bench.empty()
                st.error(f"Benchmark error: {str(ex)}")


# =========================================================================
# TAB 3: NLP Case Study & Viva Voce
# =========================================================================

with tab_case_study:
    st.subheader("📚 Academic Case Study Documentation & Viva Preparation")

    st.markdown(
        """
        ### 1. Abstract & Problem Statement
        In the era of information explosion, processing and distilling high-volume unstructured text into actionable insights
        is critical across academia, law, medicine, and business. Traditional text summarization methods were predominantly
        **extractive**, selecting verbatim sentences based on term frequencies (TF-IDF, TextRank). However, extractive summaries
        frequently suffer from lack of coherence, redundant phrasing, and disjointed transitions.

        This case study explores **abstractive text summarization** powered by sequence-to-sequence Transformer architectures
        (**BART** and **FLAN-T5**) alongside instruction-tuned Large Language Models (**LLMs**). We evaluate these architectures
        on generation quality, inference latency, compression capability, token handling, and empirical lexical overlap via **ROUGE**.

        ---

        ### 2. Theoretical Architecture Comparison

        | Architectural Dimension | BART (`facebook/bart-large-cnn`) | FLAN-T5 (`google/flan-t5-base`) | Local LLM via Ollama (e.g. Llama 3) |
        | :--- | :--- | :--- | :--- |
        | **Underlying Architecture** | Bidirectional Encoder + Autoregressive Decoder | Encoder-Decoder Transformer | Decoder-only Autoregressive LLM |
        | **Pretraining Paradigm** | Denoising Autoencoder (corrupted tokens, sentence permutation) | Multi-task instruction tuning on 1,800+ tasks | Causal Language Modeling on massive web corpora |
        | **Fine-Tuning Objective** | CNN / DailyMail abstractive summarization dataset | Instruction-conditioned task prompts (`summarize: `) | Instruction-following via RLHF / DPO |
        | **Input Token Context** | 1,024 Tokens | 512 Tokens | 4,096 - 8,192+ Tokens |
        | **Strengths** | Outstanding journalistic flow, polished narrative synthesis | High instruction compliance, parameter-efficient | Zero-shot reasoning, highly customizable style |
        | **Primary Limitation** | Strict 1024 token window requiring chunking | Shorter default context (512 tokens), sensitive to prompts | Higher compute and memory requirements |

        ---

        ### 3. Long Document Chunking & Hierarchical Summarization
        Neural sequence-to-sequence models have fixed positional embeddings. Passing documents that exceed this input context
        results in unrecoverable token truncation and loss of vital information.

        To solve this, our system implements **Recursive Hierarchical Summarization**:
        1. **Token-Aware Sentence Segmentation**: Punctuation-aware regex partitions the document into sentences.
        2. **Safe Chunk Assembly**: Sentences are accumulated into chunks strictly below the model's safe token budget (750 tokens for BART, 380 for FLAN-T5).
        3. **Pass 1 Chunk Summarization**: Each chunk is summarized independently to distill localized facts.
        4. **Recursive Condensation**: If combined chunk summaries still exceed model capacity, the system recurses with a depth guard (`max_depth = 3`) to eliminate infinite loops.
        5. **Final Cohesive Synthesis**: A final pass across condensed summaries synthesizes a unified narrative.

        ---

        ### 4. Evaluation Metrics Formulation
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the n-gram overlap between candidate summaries ($C$)
        and human reference summaries ($R$):

        $$\\text{ROUGE-N} = \\frac{\\sum_{S \\in \\{R\\}} \\sum_{gram_n \\in S} \\text{Count}_{match}(gram_n)}{\\sum_{S \\in \\{R\\}} \\sum_{gram_n \\in S} \\text{Count}(gram_n)}$$

        - **ROUGE-1**: Evaluates unigram (single word) overlap, measuring information coverage.
        - **ROUGE-2**: Evaluates bigram (word pair) overlap, measuring phrase fluency and syntactic fidelity.
        - **ROUGE-L**: Measures the Longest Common Subsequence (LCS), assessing sentence structure without requiring consecutive n-grams.

        ---

        ### 5. Academic Viva Voce (Oral Defense) Q&A Guide

        **Q1: What is the fundamental difference between Extractive and Abstractive Summarization?**
        > *Answer:* Extractive summarization selects and concatenates pre-existing sentences directly from the source text based on statistical importance (e.g., TextRank, LexRank). Abstractive summarization interprets semantic meaning and generates entirely new sentences using natural language generation (NLG), paraphrasing and compressing concepts like a human writer.

        **Q2: Why do Transformers outperform RNNs/LSTMs in sequence summarization?**
        > *Answer:* LSTMs process tokens sequentially, creating gradient vanishing/exploding problems over long documents and preventing GPU parallelization. Transformers use multi-head self-attention, computing pairwise word relationships across the entire sequence simultaneously in $\\mathcal{O}(1)$ sequential operations.

        **Q3: How does BART's pretraining differ from standard BERT or GPT?**
        > *Answer:* BERT is encoder-only (good for classification), and GPT is decoder-only (good for causal generation). BART is a full encoder-decoder model trained as a denoising autoencoder. The encoder ingests corrupted text (random token masking, token deletion, document rotation, and sentence permutation), while the decoder learns to reconstruct the original uncorrupted document.

        **Q4: Why does FLAN-T5 require the prefix `"summarize: "`?**
        > *Answer:* FLAN-T5 is a "Text-to-Text" model trained on hundreds of tasks simultaneously (translation, QA, classification, summarization). The prompt prefix conditions the decoder to activate summarization task weights rather than answering questions or translating.

        **Q5: Can ROUGE be calculated without a ground truth reference summary?**
        > *Answer:* No. ROUGE is fundamentally a reference-dependent metric that requires an authoritative ground truth summary to calculate recall, precision, and F1. Evaluating summaries without a reference requires reference-free metrics (e.g., semantic cosine similarity with the source text, perplexity, or LLM-as-a-judge).
        """
    )
