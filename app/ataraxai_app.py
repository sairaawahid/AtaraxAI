import streamlit as st
import datetime
import socket
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from utils.ollama_prompt import generate_dynamic_prompt
from utils.ollama_runner import run_ollama_prompt
from utils.onnx_infer import predict_crisis
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit.components.v1 as components

# --- Environment Detection ---
def is_local():
    hostname = socket.gethostname()
    return not hostname.startswith("cloud") and "streamlit" not in hostname

# --- Load Summarizer ---
@st.cache_resource(show_spinner=False)
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return tokenizer, model

summarizer_tokenizer, summarizer_model = load_summarizer()

def summarize_journal_entry(entry_text):
    input_text = "summarize: " + entry_text.strip().replace("\n", " ")
    inputs = summarizer_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarizer_model.generate(
        inputs, max_length=100, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- Initial Setup ---
st.set_page_config(page_title="AtaraxAI: Private Mental Health Companion")
st.title("AtaraxAI: CBT + Crisis Detection")
st.subheader("Your Private, Offline Mental Health Assistant")

# --- Memory & User Directory Setup ---
os.makedirs("user_data", exist_ok=True)
USER_MEMORY_FILE = "user_data/user_memory.json"

if not os.path.exists(USER_MEMORY_FILE):
    with open(USER_MEMORY_FILE, "w") as f:
        json.dump({"entries": [], "summary_text": ""}, f)

def load_user_memory():
    with open(USER_MEMORY_FILE, "r") as f:
        return json.load(f)

def save_user_memory(data):
    with open(USER_MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

# --- Session State Setup ---
for key, default in {
    "history": [],
    "developer_mode": False,
    "USE_OLLAMA": is_local(),
    "conversation_active": True,
    "memory_context": "",
    "use_memory": True,
    "user_input": "",
    "clear_input": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar Controls ---
st.sidebar.header("Customize Your Support")
st.session_state.use_memory = st.sidebar.checkbox("Use Memory for Context", value=True)

if is_local():
    st.sidebar.markdown("---")
    st.session_state.developer_mode = st.sidebar.toggle("</> Developer Mode", value=False)
    if st.session_state.developer_mode:
        st.session_state.USE_OLLAMA = st.sidebar.radio(
            "Toggle between real model (Ollama) and mock demo mode.", [True, False], format_func=lambda x: "Ollama" if x else "Mock"
        )
        st.sidebar.markdown("---")

language = st.sidebar.selectbox("Language", ["English", "Urdu", "Spanish", "French"])
tone = st.sidebar.selectbox("Tone", ["Supportive", "Calming", "Motivational", "Professional"])
cbt_technique = st.sidebar.selectbox("CBT Technique", ["None", "Cognitive Reframing", "Socratic Questioning", "Gratitude Focus", "Self-Compassion"])
resilience_focus = st.sidebar.checkbox("Highlight Strengths")
response_length = st.sidebar.selectbox("Response Length", ["Short", "Detailed"])
multi_turn = st.sidebar.checkbox("Ask Reflective Question")
include_resources = st.sidebar.checkbox("Suggest Self-Help Resources")
user_name = st.sidebar.text_input("Your Name (optional)")
cultural_sensitivity = st.sidebar.selectbox("Cultural Context", ["None", "South Asian", "Western", "Middle Eastern", "East Asian"])
coping_preferences = st.sidebar.multiselect("Preferred Coping Methods", ["Journaling", "Spirituality", "Exercise", "Social Support", "Grounding Techniques"])
trauma_tag = st.sidebar.text_input("Trauma Note (optional)")
emotion_tag = st.sidebar.text_input("Detected Emotion (optional)")
threshold = st.sidebar.slider("Crisis Sensitivity", 0.1, 0.9, 0.5, step=0.05)
st.sidebar.markdown("Set how sensitive the system is when identifying a crisis.")

# --- Info ---
with st.expander("ℹ️ **What does 'Crisis Detection' mean?**"):
    st.markdown("""
    We use a fine-tuned model to evaluate if your entry may indicate emotional distress, suicidal thoughts, or psychological crisis.

    **Confidence Score:** Reflects the model’s certainty.

    If high risk is detected, resources will be included, but your data never leaves your device (offline & private).

    _**Note:** This response is AI-generated and not a substitute for professional help._
    """)

# --- Chat Interface ---
st.markdown("---")
st.subheader("AtaraxAI Chat")

for entry in st.session_state.history:
    st.chat_message("user").markdown(entry["user"])
    st.chat_message("assistant").markdown(entry["response"])

if st.session_state.conversation_active:
    with st.chat_message("user"):
        col1, col2 = st.columns([0.97, 0.03])  # 0.97 for input, 0.03 for spacing

        # --- Auto-clear input logic ---
        if st.session_state.clear_input:
            st.session_state.user_input = ""
            st.session_state.clear_input = False

        with col1:
            user_input = st.text_area(
                "Your Journal Entry",
                key="user_input",
                placeholder="Continue...",
                height=80,
                label_visibility="collapsed"
            )
            st.markdown("""
                <style>
                textarea[data-testid="stTextArea"] {
                    width: 99% !important;
                    min-width: 350px !important;
                    max-width: 1000px !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
        submitted = st.button("Generate CBT Response")

    if submitted:
        if not user_input.strip():
            st.warning("Please enter your journal entry first.")
        else:
            journal_input = user_input

            st.markdown("---")
            st.subheader("🛡️ Crisis Risk Analysis (Offline)")
            predicted_label, confidence = predict_crisis(journal_input, return_confidence=True)
            is_crisis = confidence >= threshold

            risk_level = int(confidence * 100)
            bar_color = "red" if is_crisis else "green"
            st.progress(risk_level, text=f"{risk_level}% Confidence")
            st.markdown(f"<span style='color:{bar_color}; font-weight:bold;'>Risk Level: {risk_level}%</span>", unsafe_allow_html=True)
            st.caption("🧪 Powered by fine-tuned ONNX model")

            if is_crisis:
                st.error(f"⚠️ Crisis Detected — Confidence: {confidence:.2f}")
                st.info("Consider seeking immediate support. Resources will be included in the response.")
                include_resources = True
            else:
                st.success(f"✅ No Immediate Crisis Detected — Confidence: {confidence:.2f}")

            # --- MEMORY CONTEXT ---
            if st.session_state.use_memory:
                memory_data = load_user_memory()
                memory_para = memory_data.get("summary_text", "")
                context_text = f"### Memory Summary:\n{memory_para}\n\n" if memory_para else ""
            else:
                context_text = ""

            prompt = generate_dynamic_prompt(
                user_input=journal_input,
                emotion_tag=emotion_tag,
                language=language,
                trauma_tag=trauma_tag,
                tone=tone,
                cbt_technique=cbt_technique if cbt_technique != "None" else None,
                resilience_focus=resilience_focus,
                context_note=None,
                ethical_disclaimer=True,
                include_resources=include_resources,
                multi_turn=multi_turn,
                response_length=response_length,
                cultural_sensitivity=cultural_sensitivity if cultural_sensitivity != "None" else None,
                user_name=user_name if user_name else None,
                coping_preferences=", ".join(coping_preferences) if coping_preferences else None
            )

            if context_text:
                prompt = context_text + "\n" + prompt

            with st.spinner("Thinking..."):
                if st.session_state.USE_OLLAMA:
                    response = run_ollama_prompt(prompt)
                else:
                    response = "Mock response here."

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            token_count = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
            response_words = len(response.split())

            new_entry = {
                "time": timestamp,
                "user": journal_input,
                "response": response,
                "prompt": prompt,
                "tokens": token_count,
                "response_words": response_words,
                "crisis": is_crisis,
                "confidence": confidence,
                "emotion_tag": emotion_tag,
                "cbt_technique": cbt_technique,
                "language": language,
                "tone": tone,
                "trauma_tag": trauma_tag,
                "cultural_sensitivity": cultural_sensitivity,
                "coping_preferences": coping_preferences,
                "resilience_focus": resilience_focus,
                "context_note": None,
                "response_length": response_length,
                "multi_turn": multi_turn,
                "include_resources": include_resources,
                "user_name": user_name
            }

            st.session_state.history.append(new_entry)

            # Save memory summary (expanded version)
            memory_data = load_user_memory()
            clean_summary = summarize_journal_entry(journal_input)
            summary_record = {
                "time": timestamp,
                "emotion": emotion_tag or "N/A",
                "journal": journal_input,
                "summary": clean_summary
            }
            # append the new per-entry summary
            memory_data["entries"].append(summary_record)

            # Only include entries that have a summary!
            all_summaries = " ".join([e["summary"] for e in memory_data["entries"] if "summary" in e and e["summary"].strip()])
            if all_summaries.strip():
                memory_data["summary_text"] = summarize_journal_entry(all_summaries)
            else:
                memory_data["summary_text"] = ""

            # save back to disk
            save_user_memory(memory_data)
            st.success("✓ Memory updated!")

            # --- Clear input on next rerun ---
            st.session_state.clear_input = True
            st.rerun()

# --- End Conversation ---
if st.session_state.conversation_active:
    if st.button("🛑 End Conversation"):
        st.session_state.conversation_active = False
        st.success("Conversation ended. You can download the full transcript below.")

    if st.button("🧹 Clear Conversation"):
        st.session_state.history = []
        st.success("Conversation cleared. Memory still retained.")

# --- Developer Diagnostics ---
if st.session_state.developer_mode and st.session_state.history:
    st.subheader("</> Developer Mode Diagnostics")
    last_entry = st.session_state.history[-1]
    st.markdown("Final Prompt Sent to Ollama:")
    # Custom CSS to enable word wrap in code block
    st.markdown("""
    <style>
    pre code {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.code(last_entry['prompt'], language="markdown")
    st.markdown(f"**Model Used:** {'Gemma:2b (Ollama)' if st.session_state.USE_OLLAMA else 'Mock'}")
    st.markdown(f"**Prompt Token Count:** {last_entry['tokens']}")
    st.markdown(f"**Response Length:** {last_entry['response_length']}")
    st.markdown(f"**Response Word Count:** {last_entry['response_words']}")

    df = pd.DataFrame(st.session_state.history)
    fig, ax = plt.subplots()
    ax.plot(df['time'], df['tokens'], label='Prompt Tokens', marker='o')
    ax.plot(df['time'], df['response_words'], label='Response Words', marker='s')
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Count")
    ax.set_title("Token Usage & Response Length Over Time")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --- Export Chat History ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("📂 Export Chat History")

    def format_entry(entry):
        def get(field, default="N/A"):
            return entry.get(field) if entry.get(field) not in [None, "", [], "None"] else default

        settings = [
            f"Time: {get('time')}",
            f"User: {get('user_name', 'Anonymous') or 'Anonymous'}",
            f"Emotion: {get('emotion_tag')}",
            f"CBT Technique: {get('cbt_technique', 'None')}",
            f"Language: {get('language', 'English')}",
            f"Tone: {get('tone', 'Supportive')}",
            f"Crisis Risk: {'Crisis' if entry.get('crisis') else 'Non-Crisis'} ({entry.get('confidence', 0.0):.2f})",
            f"Trauma Tag: {get('trauma_tag')}",
            f"Cultural Sensitivity: {get('cultural_sensitivity')}",
            f"Coping Preferences: {get('coping_preferences') if isinstance(get('coping_preferences'), str) else ', '.join(get('coping_preferences') or []) or 'N/A'}",
            f"Resilience Focus: {'Yes' if entry.get('resilience_focus') else 'No'}",
            f"Context Note: {get('context_note')}",
            f"Response Length: {get('response_length', 'Detailed')}",
            f"Multi-turn: {'Yes' if entry.get('multi_turn') else 'No'}",
            f"Include Resources: {'Yes' if entry.get('include_resources') else 'No'}"
        ]

        return (
            "\n".join(settings)
            + "\n\nJournal Entry:\n"
            + entry.get("user", "")
            + "\n\nAtaraxAI Response:\n"
            + entry.get("response", "")
        )

    chat_transcript = ("\n\n" + "=" * 60 + "\n\n").join(format_entry(e) for e in st.session_state.history)

    st.download_button("📄 Download Full Conversation (.txt)", chat_transcript, file_name="ataraxai_chat_history.txt")

# --- Export Weekly PDF ---
if st.sidebar.button("📄 Download Weekly Summary PDF"):
    memory = load_user_memory()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Weekly Emotional Summary", ln=1, align="C")
    for e in memory["entries"][-7:]:
        pdf.multi_cell(0, 10, txt=f"{e['time']} - {e['emotion']}\n{e.get('insight', e.get('summary', ''))}\n")
    pdf.output("user_data/weekly_summary.pdf")
    st.success("✅ PDF Exported! Check `user_data/weekly_summary.pdf`")

# --- Delete Memory ---
if st.sidebar.button("🗑️ Clear Memory"):
    save_user_memory({"entries": [], "summary_text": ""})
    st.success("Memory cleared.")
