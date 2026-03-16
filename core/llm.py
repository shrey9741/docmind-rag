"""
core/llm.py
===========
Handles loading and caching of the Flan-T5 language model.
Wrapped in a LangChain HuggingFacePipeline for easy invocation.
Cached with @st.cache_resource so the model loads only once
per session regardless of how many questions are asked.
"""
import streamlit as st
from config.settings import LLM_MODEL, MAX_NEW_TOKENS, DO_SAMPLE, TEMPERATURE

@st.cache_resource(show_spinner=False)
def load_llm():
    """
    Load the Flan-T5 model and tokenizer from HuggingFace.
    Wraps them in a LangChain HuggingFacePipeline.
    Model: google/flan-t5-base (~250MB, runs on CPU)
    Task : text2text-generation (encoder-decoder / seq2seq)
    Returns:
        LangChain HuggingFacePipeline object ready for .invoke()
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from langchain_community.llms import HuggingFacePipeline

    # Load tokenizer and model weights from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    # Build pipeline — only pass temperature if do_sample is True
    pipe_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
    }
    if DO_SAMPLE:
        pipe_kwargs["temperature"] = TEMPERATURE

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        **pipe_kwargs,
    )

    # Wrap in LangChain for easy .invoke() calls
    return HuggingFacePipeline(pipeline=pipe)