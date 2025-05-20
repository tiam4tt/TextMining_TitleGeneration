# streamlit
# transformers>=4.45.0
# torch>=2.6.0
# bcrypt

import datetime

import requests

# import torch
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_model(model_name):
    # Placeholder for model loading logic

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    # .to(torch.device(device))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    st.session_state["model_name"] = model_name

    return model, tokenizer


def generate_title(abstract, num_generation=1, creativity=1.2):
    if not abstract:
        return

    model, tokenizer = st.session_state["model"], st.session_state["tokenizer"]

    inputs = tokenizer(abstract, return_tensors="pt", padding=True, truncation=True)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=48,
        do_sample=True,  # Enable sampling
        top_k=50,  # Sample from top 50 tokens
        top_p=0.9,  # Nucleus sampling with 90% probability mass
        temperature=creativity,
        num_return_sequences=num_generation,
    )
    title = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return title


def getinfo():
    model_name = st.session_state["model_name"]
    titles = st.session_state["titles"]
    abstract = st.session_state["abstract"]
    temperature = st.session_state["temperature"]
    timestamp = datetime.datetime.now()

    return {
        "model_name": model_name,
        "abstract": abstract,
        "temperature": temperature,
        "titles": titles,
        "timestamp": timestamp,
    }


def load_entry(info):
    with st.expander(str(info["timestamp"])):
        st.markdown(
            f"Created by *{info["model_name"]}* at temperature **{info["temperature"]}**"
        )
        st.markdown("#### Abstract")
        with st.container(border=True):
            st.markdown(info["abstract"])
        st.markdown("#### Generated Titles")
        with st.container(border=True):
            for i, title in enumerate(info["titles"]):
                st.write(f"{i+1}. {title}")


if "history" not in st.session_state:
    st.session_state["history"] = []


def home():
    st.header("Welcome!")

    # Initialize session state for model and tokenizer if not present
    if "model" not in st.session_state:
        st.session_state["model"] = None
    if "tokenizer" not in st.session_state:
        st.session_state["tokenizer"] = None
    if "model_name" not in st.session_state:
        st.session_state["model_name"] = None
    if "titles" not in st.session_state:
        st.session_state["titles"] = None
    if "abstract" not in st.session_state:
        st.session_state["abstract"] = None
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = None

    options = [
        "tiam4tt/flan-t5-titlegen-springer",
        "HTThuanHcmus/bart-finetune-scientific-improve",
    ]
    model_name = st.selectbox(
        "Current model", options=options, placeholder="No model selected"
    )

    # Load model only if the selected model name has changed
    if model_name != st.session_state["model_name"]:
        with st.spinner("Loading model..."):
            st.session_state["model"], st.session_state["tokenizer"] = load_model(
                model_name
            )
            st.session_state["model_name"] = model_name

    st.subheader("Your Abstract")
    abstract = st.text_area(
        label="label not displayed", label_visibility="collapsed", height=180
    )

    col1, col2 = st.columns(2)

    with col1:
        num_gen = int(
            st.number_input(
                "Number of generations (1-10)", min_value=1, max_value=10, step=1
            )
        )
    with col2:
        creativity = st.slider(
            "Choose the level of creativity for the generated title",
            min_value=1.0,
            max_value=2.0,
            step=0.1,
            value=1.2,
            help="Reduce the level of creativity if the generated titles are getting unrelevant.",
        )
    _, button_col, _ = st.columns(3)
    with button_col:
        if st.button(
            "**Generate Title**",
            type="primary",
            use_container_width=True,
        ):
            if len(abstract) != 0:
                with st.spinner("Generating results"):
                    st.session_state["titles"] = generate_title(
                        abstract, num_generation=num_gen, creativity=creativity
                    )
                st.session_state["abstract"] = abstract
                st.session_state["temperature"] = creativity

                st.session_state["history"].append(getinfo())
            else:
                st.toast(
                    "Please input abstract to see generated title.",
                    icon=":material/info:",
                )

    st.subheader("Generated Titles")
    with st.container():
        titles = st.session_state["titles"]
        if titles:
            for title in titles:
                with st.container(border=True):
                    st.markdown(f"**{title.strip()}**")


def about():
    url = "https://raw.githubusercontent.com/tiam4tt/TextMining_TitleGeneration/refs/heads/main/README.md"
    response = requests.get(url)
    if response.status_code == 200:
        content = response.text
        st.markdown(content)
    else:
        st.markdown("*Oops, an error occurred while loading this page*")


def history():
    st.subheader("History")
    st.warning(
        "Please note that any recorded interactions **WILL BE CLEARED** on page refresh!"
    )
    st.markdown("---")
    if not st.session_state["history"]:
        st.info("No interaction recorded!")
    else:
        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()
        for info in st.session_state["history"]:
            load_entry(info)


st.set_page_config(page_title="TitleFormer", page_icon=":material/token:")

title = "This is TitleFormer"
description = "A simple tool that generates **titles** from your scientific publication's **abstract** using seq2seq **transformer**-based models."

# Simplify session state initialization
if "page" not in st.session_state:
    st.session_state["page"] = "home"

with st.sidebar:
    st.title(title)
    st.markdown(f"*{description}*")
    st.divider()

    disableHistory, disableHome, disableAbout = False, False, False

    if st.session_state["page"] == "home":
        disableHome = True

    if st.session_state["page"] == "history":
        disableHistory = True

    if st.session_state["page"] == "about":
        disableAbout = True

    if st.button(
        "**Home**", use_container_width=True, type="primary", disabled=disableHome
    ):
        st.session_state["page"] = "home"
        st.rerun()

    if st.button(
        "Recent history",
        use_container_width=True,
        type="secondary",
        disabled=disableHistory,
    ):
        st.session_state["page"] = "history"
        st.rerun()

    if st.button(
        "About this project",
        use_container_width=True,
        type="tertiary",
        disabled=disableAbout,
    ):
        st.session_state["page"] = "about"
        st.rerun()


if st.session_state["page"] == "home":
    home()
elif st.session_state["page"] == "about":
    about()
elif st.session_state["page"] == "history":
    history()
