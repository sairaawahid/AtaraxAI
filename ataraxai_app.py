import streamlit as st
import datetime
import socket
import os
import json
import pandas as pd
from fpdf import FPDF
import plotly.graph_objs as go
import plotly.express as px
from utils.ollama_prompt import generate_dynamic_prompt
from utils.ollama_runner import run_ollama_prompt
from utils.onnx_infer import predict_crisis
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import base64
# from streamlit_extras.st_tooltip import st_tooltip

# --- Mental Health Color Palette ---
mental_health_colors = [
    "#6041f7",  # lavender white
    "#2a98f3",  # blue-white
    "#dc14e3",  # pinky lavender
    "#2252cd",  # sky blue-lavender
    "#7e16ba",  # soft purple
    "#0f959e",  # soft blue
    "#cc258c",  # blush
    "#0b9a2a",  # mint
]

def is_local():
    hostname = socket.gethostname()
    return not hostname.startswith("cloud") and "streamlit" not in hostname

@st.cache_resource(show_spinner=False)
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return tokenizer, model

summarizer_tokenizer, summarizer_model = load_summarizer()

def summarize_text_long(text, max_len=120):
    input_text = "summarize: " + text.replace("\n", " ")
    inputs = summarizer_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summarizer_model.generate(
        inputs, max_length=max_len, min_length=15, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def required_label(text):
    return f"{text} <span style='color:red;font-weight:bold;'>*</span>"

# --- Initial Setup ---
st.set_page_config(page_title="AtaraxAI: Private, Off-line Mental Health Companion")
st.markdown("""
<div style="
    margin: 30px auto 20px auto;
    padding: 28px 18px 18px 18px;
    border-radius: 26px;
    max-width: 1500px;
    background: linear-gradient(90deg, #ece8ff 20%, #DDE6FF 30%, #ffe8f6 90%);
    box-shadow: 0 6px 32px 0 rgba(150,200,190,0.13);
    display: flex;
    flex-direction: column;
    align-items: center;
    border: none;
">
    <!-- Icon Example: replace src with your logo if you have -->
    <!-- <img src="your_logo.png" style="height:46px; margin-bottom:6px;" /> -->
    <h1 style="
        font-size:2.7rem;
        font-weight: 700;
        margin-bottom: 0.36rem;
        margin-top: 0;
        color: #101820;
        letter-spacing: 1.2px;
        text-align: center;
    ">
        AtaraxAI: Cognitive Behavioral Therapy + Crisis Detection
    </h1>
    <h3 style="
        font-size:1.35rem;
        font-weight: 600;
        color: #101820;
        margin-bottom: 0.45rem;
        margin-top: 0;
        text-align: center;
    ">
        Your Private, Offline Mental Health Companion ⋆˙⟡
    </h3>
    <p style="
        font-size:1.09rem;
        font-weight: 400;
        color: #222;
        margin-top: 0;
        margin-bottom: 0.1rem;
        text-align: center;
    ">
        AtaraxAI helps you journal, reflect, and build emotional resilience. Utilize privacy-first CBT and real-time crisis detection—no internet needed.<br>
        Your data stays only with you.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ---- Custom Button Styling for All Main Buttons ---- */
.stButton button, .stDownloadButton button {
    background-color: #ddb3fc !important;
    color: #101820 !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.08rem !important;
    box-shadow: 0 2px 9px 0 rgba(221,179,252,0.10);
    transition: background 0.18s, color 0.18s;
    padding: 0.65rem 1.65rem !important;
}
.stButton button:hover, .stDownloadButton button:hover {
    background-color: #c994e8 !important;  /* a slightly deeper lavender */
    color: #fff !important;
    border: none !important;
}
/* --- AtaraxBot Send button in floating chat --- */
button[data-testid="chatbot_send_btn"] {
    background-color: #ddb3fc !important;
    color: #101820 !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.08rem !important;
    padding: 0.5rem 1.5rem !important;
}
button[data-testid="chatbot_send_btn"]:hover {
    background-color: #c994e8 !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)


# st.title("AtaraxAI: CBT + Crisis Detection")
# st.subheader("Your Private, Offline Mental Health Assistant")

# --- Memory & User Directory Setup ---
os.makedirs("user_data", exist_ok=True)
USER_MEMORY_FILE = "user_data/user_memory.json"

if not os.path.exists(USER_MEMORY_FILE):
    with open(USER_MEMORY_FILE, "w") as f:
        json.dump({"entries": [], "summary_text": ""}, f)

def load_user_memory():
    if is_local() and os.path.exists(USER_MEMORY_FILE):
        with open(USER_MEMORY_FILE, "r") as f:
            return json.load(f)
    else:
        return st.session_state.get("cloud_memory", {"entries": [], "summary_text": ""})

def save_user_memory(data):
    if is_local():
        with open(USER_MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
    else:
        st.session_state["cloud_memory"] = data

# --- Session State Setup ---
for key, default in {
    "history": [],
    "developer_mode": False,
    "USE_OLLAMA": is_local(),
    "memory_context": "",
    "use_memory": True,
    "user_input": "",
    "clear_input": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
    
    if "chatbot_open" not in st.session_state:
        st.session_state.chatbot_open = False

    if "chatbot_history" not in st.session_state:
        st.session_state.chatbot_history = []


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

user_name = st.sidebar.text_input("Your Name (optional)")

st.sidebar.markdown(required_label("Tone"), unsafe_allow_html=True)
tone = st.sidebar.selectbox(
    "",
    ["Supportive", "Calming", "Motivational", "Professional"],
    key="tone"
)

# --- CBT Techniques Dictionary with Descriptions ---
cbt_techniques = {
    "Cognitive Reframing": "Learn to identify and change unhelpful thoughts to more balanced ones.",
    "Socratic Questioning": "Use gentle questions to examine and challenge your thoughts and beliefs.",
    "Gratitude Focus": "Practice recognizing and appreciating positive things in your life.",
    "Self-Compassion": "Treat yourself with kindness, especially during difficult moments.",
    "Behavioral Activation": "Increase positive activities to improve mood and motivation.",
    "Problem-Solving": "Break down challenges and find practical, step-by-step solutions.",
    "Exposure Therapy": "Gradually face situations or feelings you avoid to reduce fear and distress.",
    "Mindfulness Practice": "Pay attention to the present moment without judgment.",
    "Thought Records": "Write down your thoughts, feelings, and actions to spot patterns and make changes.",
    "Activity Scheduling": "Plan enjoyable or meaningful activities to boost well-being and structure.",
    "Cognitive Restructuring": "Identify negative thinking patterns and shift to healthier perspectives.",
    "Stress Inoculation": "Learn coping strategies to handle stress before it becomes overwhelming.",
    "Relaxation Training": "Use techniques like deep breathing or muscle relaxation to calm your mind and body.",
    "Values Clarification": "Explore what truly matters to you and let those values guide your choices.",
    "Goal Setting": "Set realistic, achievable goals to create a sense of purpose and direction.",
    "Assertiveness Training": "Learn to express your needs and opinions confidently and respectfully."
}

# --- CBT Technique Selection ---
st.sidebar.markdown('<label style="font-weight:600;">CBT Technique <span style="color:red;">*</span></label>', unsafe_allow_html=True)

cbt_technique = st.sidebar.selectbox(
    "",
    list(cbt_techniques.keys()),
    key="cbt_technique"
)

# --- Description Display with Info Icon ---
desc = cbt_techniques.get(cbt_technique, "")
st.sidebar.markdown(
    f'<span style="font-size:1.02rem;font-weight:500;">🛈</span> '
    f'<span style="font-size:1.06rem;color:#6f3fd3;font-weight:700;">{cbt_technique}</span> '
    f'<span style="color:#5e3ba7;">{desc}</span>',
    unsafe_allow_html=True
)

st.sidebar.markdown(required_label("Response Length"), unsafe_allow_html=True)
response_length = st.sidebar.selectbox(
    "",
    ["Short", "Detailed"],
    key="response_length"
)

coping_options = [
    "Journaling",
    "Spirituality/Faith Practices",
    "Exercise/Physical Activity",
    "Social Support (friends, family)",
    "Grounding Techniques",
    "Mindfulness Meditation",
    "Breathing Exercises",
    "Creative Arts (music, drawing, crafts)",
    "Nature Time (walks, gardening, etc.)",
    "Listening to Music",
    "Progressive Muscle Relaxation",
    "Healthy Eating/Nutrition",
    "Sleep Hygiene",
    "Problem-Solving",
    "Time Management/Planning",
    "Gratitude Practices",
    "Humor/Laughter",
    "Volunteering/Helping Others",
    "Pet/Animal Interaction",
    "Seeking Professional Support",
    "Reading",
    "Yoga",
    "Affirmations/Self-Talk",
    "Digital Detox (reducing screen time)",
    "Hobbies (cooking, puzzles, etc.)",
    "Visualization/Imagery",
    "Body Scan Relaxation",
    "Positive Distraction (movies, games, etc.)",
    "Support Groups (peer or online)",
    "Spending time with loved ones",
    "Assertiveness Practice"
]

st.sidebar.markdown(required_label("Preferred Coping Methods"), unsafe_allow_html=True)
coping_preferences = st.sidebar.multiselect(
    "",
    coping_options,
    key="coping_preferences"
)

resilience_focus = st.sidebar.checkbox("Highlight Strengths")
include_resources = st.sidebar.checkbox("Suggest Self-Help Resources")
trauma_tag = st.sidebar.text_input("Trauma Note (optional)")

st.sidebar.markdown(required_label("Detected Emotion"), unsafe_allow_html=True)
emotion_tag = st.sidebar.text_input(
    "",
    key="emotion_tag"
)

threshold = st.sidebar.slider("Crisis Sensitivity", 0.1, 0.9, 0.5, step=0.05)
st.sidebar.markdown("Set how sensitive the system is when identifying a crisis.")

# --- Sidebar: Weekly Summary PDF and Clear Memory (Always Available) ---
if st.sidebar.button("💾 Download Weekly Summary"):
    memory = load_user_memory()
    font_regular = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
    font_bold = os.path.join(os.path.dirname(__file__), "DejaVuSans-Bold.ttf")
    class PDF(FPDF):
        def header(self):
            self.set_font("DejaVu", "B", 14)
            self.cell(0, 10, "Weekly Emotional Summary", ln=1, align="C")
            self.ln(6)
        def chapter_title(self, num, entry_title):
            self.set_font("DejaVu", "B", 12)
            self.cell(0, 8, f"Day {num}: {entry_title}", ln=1, align="L")
        def chapter_body(self, body):
            self.set_font("DejaVu", "", 12)
            self.multi_cell(0, 8, body)
            self.ln()
    pdf = PDF()
    pdf.add_font("DejaVu", "", font_regular, uni=True)
    pdf.add_font("DejaVu", "B", font_bold, uni=True)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    for i, e in enumerate(memory["entries"][-7:], 1):
        pdf.chapter_title(i, e.get("time", "Unknown"))
        body = f"{e.get('emotion', '')}\n{e.get('journal', '')}\n"
        pdf.chapter_body(body)
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    st.success("✅ PDF Exported!")
    st.download_button(
        "📄 Download Weekly Summary (.pdf)",
        data=pdf_bytes,
        file_name="weekly_summary.pdf",
        mime="application/pdf"
    )

if not is_local():
    st.sidebar.download_button(
        label="⬇️ Download My Data (JSON)",
        data=json.dumps(st.session_state.get("cloud_memory", {}), indent=2),
        file_name="ataraxai_session.json",
        mime="application/json"
    )

if st.sidebar.button("❌ Clear Memory"):
    save_user_memory({"entries": [], "summary_text": ""})
    st.success("Memory cleared.")



# --- Info ---
with st.expander("ℹ️ **What does 'Crisis Detection' mean?**"):
    st.markdown("""
    <div style="
        margin: 0.5rem auto 0.5rem auto;
        padding: 20px 18px 16px 18px;
        border-radius: 18px;
        max-width: 1500px;
        background: linear-gradient(90deg, #ece8ff 20%, #DDE6FF 30%, #ffe8f6 90%);
        box-shadow: 0 4px 20px 0 rgba(150,200,190,0.10);
        color: #191b1c;
        font-size: 1.07rem;
        font-weight: 400;
    ">
        We use a fine-tuned model to evaluate if your entry may indicate emotional distress, suicidal thoughts, or psychological crisis.
        <br><br>
        <b>Confidence Score:</b> Reflects the model’s certainty.
        <br><br>
        If high risk is detected, resources will be included, but your data never leaves your device (offline & private).
        <br><br>
        <span style="color: #000000; font-weight:600;">❗Note:</span> These responses are AI-generated and not a substitute for professional help.
    </div>
    """, unsafe_allow_html=True)


# --- Memory+History Context Builder ---
def build_context_block(memory, history, n_recent=5):
    memory_entries = memory.get("entries", [])
    if len(memory_entries) > n_recent:
        to_summarize = " ".join(
            [e.get("journal", "") for e in memory_entries[:-n_recent] if e.get("journal")]
        )
        memory_summary = summarize_text_long(to_summarize) if to_summarize.strip() else ""
    else:
        memory_summary = ""
    memory_block = f"### Memory Summary:\n{memory_summary}\n" if memory_summary else ""
    recent_blocks = []
    for entry in history[-n_recent:]:
        user = entry.get("user", "")
        ai = entry.get("response", "")
        recent_blocks.append(f"User: {user}\nAI: {ai}")
    recent_dialogue = "\n\n".join(recent_blocks)
    recent_block = f"\n### Recent Conversation:\n{recent_dialogue}\n" if recent_blocks else ""
    return memory_block + recent_block

# --- Chat Interface ---

# Gradient Card for Chat Section (centered)
st.markdown("""
<div style="
    margin: 28px auto 10px auto;
    padding: 26px 14px 8px 14px;
    border-radius: 22px;
    max-width: 1500px;
    background: linear-gradient(90deg, #ece8ff 20%, #DDE6FF 30%, #ffe8f6 90%);
    box-shadow: 0 3px 20px 0 rgba(150,200,190,0.11);
    display: flex;
    flex-direction: column;
    align-items: center;
    border: none;
">
    <h2 style="
        font-size:2rem;
        font-weight: 800;
        margin-bottom: 0.18rem;
        color: #101820;
        letter-spacing: 1.1px;
        text-align: center;
    ">
        AtaraxAI Chat💭
    </h2>
    <div style="color:#000000;font-size:1.07rem;margin-bottom:12px;text-align:center;">
        Journal, reflect, or ask anything. Your session is always private & offline.
    </div>
</div>
""", unsafe_allow_html=True)

# Custom 0011 (user: #b1aaff, assistant: #98c8e9, black font)
st.markdown("""
<style>
.st-chat-message.user {
    background: #b1aaff !important;
    color: #111 !important;
    border-radius: 16px 16px 4px 16px !important;
    margin-bottom: 10px !important;
}
.st-chat-message.assistant {
    background: #98c8e9 !important;
    color: #111 !important;
    border-radius: 16px 16px 16px 4px !important;
    margin-bottom: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# Messages (user/assistant chat history)
for entry in st.session_state.history:
    st.chat_message("user").markdown(entry["user"])
    st.chat_message("assistant").markdown(entry["response"])

# Input Area (remains the same, inside the gradient card visually)
with st.chat_message("user"):
    col1, col2 = st.columns([0.97, 0.03])
    if st.session_state.clear_input:
        st.session_state.user_input = ""
        st.session_state.clear_input = False
    with col1:
        user_input = st.text_area(
            "",
            key="user_input",
            placeholder="Write your journal entry",
            height=100,
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
    missing_fields = []
    if not user_input.strip():
        missing_fields.append("journal entry")
    if not tone:
        missing_fields.append("tone")
    if not cbt_technique:
        missing_fields.append("CBT technique")
    if not response_length:
        missing_fields.append("response length")
    if not coping_preferences:
        missing_fields.append("at least one coping preference")
    if not emotion_tag:
        missing_fields.append("emotion")
    if missing_fields:
        st.warning("Please provide: " + ", ".join(missing_fields) + ".")
    else:
        journal_input = user_input
        predicted_label, confidence = predict_crisis(journal_input, return_confidence=True)
        is_crisis = confidence >= threshold
        if st.session_state.use_memory:
            memory_data = load_user_memory()
            history = st.session_state.history
            context_text = build_context_block(memory_data, history, n_recent=5)
        else:
            context_text = ""
        prompt = generate_dynamic_prompt(
            user_input=journal_input,
            context_block=context_text,
            language="English",
            emotion_tag=emotion_tag,
            cbt_technique=cbt_technique,
            tone=tone,
            coping_preferences=", ".join(coping_preferences),
            response_length=response_length,
            trauma_tag=trauma_tag,
            resilience_focus=resilience_focus,
            context_note=None,
            include_resources=include_resources,
            user_name=user_name
        )
    
        with st.spinner("AtaraxAI is Responding..."):
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
            "crisis_score": confidence,
            "confidence": confidence,
            "emotion_tag": emotion_tag,
            "cbt_technique": cbt_technique,
            "language": "English",
            "tone": tone,
            "trauma_tag": trauma_tag,
            "coping_preferences": coping_preferences,
            "resilience_focus": resilience_focus,
            "context_note": None,
            "response_length": response_length,
            "include_resources": include_resources,
            "user_name": user_name
        }
        st.session_state.history.append(new_entry)
        memory_data = load_user_memory()
        memory_data["entries"].append({
            "time": timestamp,
            "emotion": emotion_tag or "N/A",
            "journal": journal_input
        })
        save_user_memory(memory_data)
        st.success("✓ Memory updated!")
        st.session_state.clear_input = True
        st.rerun()

# --- Crisis Detection Analytics & Session Visualizations (Collapsible: all charts together) ---
with st.expander("📈 **Analytics & Session Visualizations**", expanded=False):
    st.markdown(
        "Explore your emotional and crisis risk trends over time. "
        "These charts help you spot patterns in your wellbeing and journal activity. "
        "All data remains offline and private."
        )
    if len(st.session_state.history) >= 1:
        df = pd.DataFrame(st.session_state.history)
        df['time'] = pd.to_datetime(df['time'], errors='coerce')

        # 1. Session Crisis Risk
        st.markdown("#### 1. Crisis Risk (This Session)")
        st.caption("**How was your crisis risk this session?** This chart shows your crisis risk scores for each journal entry in the current conversation.")
        crisis_scores = [entry["crisis_score"] for entry in st.session_state.history if "crisis_score" in entry]
        times = [entry["time"] for entry in st.session_state.history if "crisis_score" in entry]
        if crisis_scores and times:
            avg_crisis = sum(crisis_scores) / len(crisis_scores)
            st.markdown(
                f"**Average Crisis Risk:** <span style='color:{'red' if avg_crisis >= 0.5 else 'green'}; font-weight:bold;'>{avg_crisis*100:.1f}%</span>",
                unsafe_allow_html=True
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times,
                y=[score * 100 for score in crisis_scores],
                mode='lines+markers+text',
                name='Crisis Risk (%)',
                marker=dict(color=mental_health_colors[2]),
                text=[f"{score*100:.1f}%" for score in crisis_scores],
                textposition="top center"
            ))
            fig.update_layout(
                title="Crisis Risk Level Over This Session",
                xaxis_title="Timestamp",
                yaxis_title="Crisis Risk (%)",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig, use_container_width=True)

        # 2. Emotion Trend (over time, as smooth line)
        st.markdown("#### 2. Emotion Trend Over Time")
        st.caption("How have your reported emotions changed across sessions?")
        if "emotion_tag" in df.columns and df["emotion_tag"].notnull().any():
            df['emotion_tag'] = df['emotion_tag'].replace("", "Unknown").fillna("Unknown")
            fig = px.scatter(
                df, x="time", y="emotion_tag", color="emotion_tag",
                color_discrete_sequence=mental_health_colors,
                title="Detected Emotion Trend Over Time", labels={"emotion_tag": "Emotion"}
            )
            fig.update_traces(marker=dict(size=18, opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No emotion tags found for visualization.")

        # 3. Emotional Trend Heatmap
        st.markdown("#### 3. Emotional Trend Heatmap")
        st.caption("Frequency of each emotion per day/week. Darker means more frequent.")
        if "emotion_tag" in df.columns and df["emotion_tag"].notnull().any():
            df['date'] = df['time'].dt.date
            heatmap_data = df.groupby(['date', 'emotion_tag']).size().unstack(fill_value=0)
            if not heatmap_data.empty:
                fig = px.imshow(
                    heatmap_data.T,
                    labels=dict(x="Date", y="Emotion", color="Count"),
                    aspect="auto",
                    title="Emotional Frequency Heatmap",
                    color_continuous_scale=mental_health_colors
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough emotion data for heatmap.")
        else:
            st.info("No emotion tags for heatmap.")

        # 4. Crisis Frequency Chart
        st.markdown("#### 4. Crisis Frequency Over Time")
        st.caption("Shows how often high-risk entries occur each week or day.")
        if "crisis" in df.columns:
            crisis_df = df.copy()
            crisis_df['crisis_flag'] = crisis_df['crisis'].apply(lambda x: "Crisis" if x else "No Crisis")
            crisis_counts = crisis_df.groupby([crisis_df['time'].dt.date, 'crisis_flag']).size().unstack(fill_value=0)
            fig = go.Figure()
            if "Crisis" in crisis_counts.columns:
                fig.add_trace(go.Bar(
                    x=crisis_counts.index,
                    y=crisis_counts["Crisis"],
                    name="Crisis Episodes",
                    marker_color=mental_health_colors[3]  # pink
                ))
            if "No Crisis" in crisis_counts.columns:
                fig.add_trace(go.Bar(
                    x=crisis_counts.index,
                    y=crisis_counts["No Crisis"],
                    name="Non-Crisis Episodes",
                    marker_color=mental_health_colors[0]  # teal
                ))
            fig.update_layout(
                barmode='stack',
                title="Crisis Frequency (All Time)",
                xaxis_title="Date",
                yaxis_title="Entry Count",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No crisis data available.")

        # 5. Timeline Visualization
        st.markdown("#### 5. Journal Timeline Visualization")
        st.caption("Timeline of your journaling activity.")
        df_timeline = df.copy()
        df_timeline['date'] = df_timeline['time'].dt.date
        activity = df_timeline.groupby('date').size()
        fig = go.Figure(data=go.Scatter(
            x=activity.index, y=activity.values, mode='lines+markers',
            line=dict(color=mental_health_colors[1])
        ))
        fig.update_layout(
            title="Journal Entries Timeline",
            xaxis_title="Date",
            yaxis_title="Entries",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("At least one journal entry needed for analytics.")

# --- Export Chat History (PDF with Unicode Font Support) ---
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
            f"Coping Preferences: {get('coping_preferences') if isinstance(get('coping_preferences'), str) else ', '.join(get('coping_preferences') or []) or 'N/A'}",
            f"Resilience Focus: {'Yes' if entry.get('resilience_focus') else 'No'}",
            f"Context Note: {get('context_note')}",
            f"Response Length: {get('response_length', 'Detailed')}",
            f"Include Resources: {'Yes' if entry.get('include_resources') else 'No'}"
        ]
        return (
            "\n".join(settings)
            + "\n\nJournal Entry:\n"
            + entry.get("user", "")
            + "\n\nAtaraxAI Response:\n"
            + entry.get("response", "")
        )
    font_regular = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
    font_bold = os.path.join(os.path.dirname(__file__), "DejaVuSans-Bold.ttf")
    class PDF(FPDF):
        def header(self):
            self.set_font("DejaVu", "B", 14)
            self.cell(0, 10, "AtaraxAI Chat History", ln=1, align="C")
            self.ln(6)
        def chapter_title(self, num, entry_title):
            self.set_font("DejaVu", "B", 12)
            self.cell(0, 8, f"Entry {num}: {entry_title}", ln=1, align="L")
        def chapter_body(self, body):
            self.set_font("DejaVu", "", 12)
            self.multi_cell(0, 8, body)
            self.ln()
    pdf = PDF()
    pdf.add_font("DejaVu", "", font_regular, uni=True)
    pdf.add_font("DejaVu", "B", font_bold, uni=True)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    for i, e in enumerate(st.session_state.history, 1):
        pdf.chapter_title(i, e.get("time", "Unknown Time"))
        pdf.chapter_body(format_entry(e))
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    st.download_button(
        "💾 Download Full Conversation",
        data=pdf_bytes,
        file_name="ataraxai_chat_history.pdf",
        mime="application/pdf"
    )

# --- Developer Diagnostics (Optional) ---
if st.session_state.developer_mode and st.session_state.history:
    st.subheader("</> Developer Mode Diagnostics")
    last_entry = st.session_state.history[-1]
    st.markdown("Final Prompt Sent to Ollama:")
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
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df['time'],
        y=df['tokens'],
        mode='lines+markers',
        name='Prompt Tokens',
        marker=dict(color=mental_health_colors[5])
    ))
    fig2.add_trace(go.Scatter(
        x=df['time'],
        y=df['response_words'],
        mode='lines+markers',
        name='Response Words',
        marker=dict(color=mental_health_colors[4])
    ))
    fig2.update_layout(
        title="Token Usage & Response Length Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Count",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig2, use_container_width=True)


# ---------- AtaraxBot ----------

# ---------- Icon loader ----------
def get_image_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

icon_b64 = get_image_base64("chatbot_icon.png")

# ---------- Ensure all required session state variables ----------
if "chatbot_open" not in st.session_state:
    st.session_state.chatbot_open = False
if "chatbot_history" not in st.session_state:
    st.session_state.chatbot_history = []
if "bot_typing" not in st.session_state:
    st.session_state.bot_typing = False
if "chatbot_clear_input" not in st.session_state:
    st.session_state.chatbot_clear_input = False
if "chatbot_pending_question" not in st.session_state:
    st.session_state.chatbot_pending_question = None

# ---------- CSS for styling ----------
st.markdown("""
<style>
#chatbot-button {
    position: fixed;
    bottom: 20px;
    left: 20px;
    background-color: #69c9b0;
    border-radius: 50%;
    width: 56px;
    height: 56px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    cursor: pointer;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color 0.3s ease;
    font-size: 2rem;
}
#chatbot-button:hover {
    background-color: #4ca891;
}
#chatbot-header {
    background: linear-gradient(90deg, #ece8ff 20%, #DDE6FF 30%, #ffe8f6 90%);
    color: #8223a5;
    font-weight: bold;
    padding: 18px 24px 10px 24px;
    border-top-left-radius: 16px;
    border-top-right-radius: 16px;
    display: flex;
    align-items: center;
    font-size: 1.19rem;
    margin-bottom: 0;
}
#chatbot-header-title {
    font-weight: bold;
    font-size: 1.18rem;
    vertical-align: middle;
}
#chatbot-close {
    cursor: pointer;
    font-weight: bold;
    font-size: 18px;
    background: #e6c5fa;
    color: #6913b3;
    border-radius: 12px;
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.msg-pair {
    margin-bottom: 18px;
}
.msg-row { display: flex; margin-bottom: 10px; }
.message.user {
    background-color: #ffe8f6;
    color: #141414;
    padding: 8px 14px;
    border-radius: 15px 15px 0 15px;
    margin: 0 0 0 auto;
    max-width: 80%;
    align-self: flex-end;
    font-size: 1rem;
    box-shadow: 0 1px 4px 0 #e0e0e0;
}
.message.assistant {
    background-color: #e8eefc;
    color: #141414;
    padding: 8px 14px;
    border-radius: 15px 15px 15px 0;
    margin: 0 auto 0 0;
    max-width: 80%;
    align-self: flex-start;
    font-size: 1rem;
    box-shadow: 0 1px 4px 0 #e0e0e0;
}
.typing-indicator {
    display: flex;
    gap: 6px;
    align-items: center;
    margin-left: 8px;
    margin-top: 3px;
    margin-bottom: 4px;
}
.typing-dot {
    width: 7px;
    height: 7px;
    background: #b79eff;
    border-radius: 50%;
    animation: typing-bounce 1.1s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.14s;}
.typing-dot:nth-child(3) { animation-delay: 0.28s;}
@keyframes typing-bounce {
    0%, 80%, 100% { transform: translateY(0);}
    40% { transform: translateY(-8px);}
}
#chatbot-input {
    width: 100%;
    padding: 9px 13px;
    border-radius: 8px;
    border: 1px solid #ccc;
    font-size: 15px;
    resize: none;
    box-sizing: border-box;
    background: #f1f3fa;
}
#char-limit-note {
    font-size: 14px;
    color: #666;
    padding: 4px 0px 2px 0px;
}
#chatbot-send-btn {
    background: #d7bfff;
    color: #6913b3;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1.06rem;
    padding: 6px 28px;
    float: right;
    cursor: pointer;
    transition: background 0.18s;
    margin-top: 10px;
}
#chatbot-send-btn:hover { background: #f0c7ff; color: #1d0055;}
</style>
""", unsafe_allow_html=True)

# ---------- Floating button (NO columns, real floating) ----------
if st.button("🗫", key="chatbot_button", help="Need Help Understanding? Ask AtaraxBot"):
    st.session_state.chatbot_open = not st.session_state.chatbot_open

# ---------- Inline Chatbot (header, messages, input, etc.) ----------
if st.session_state.chatbot_open:
    # Header with icon and close button
    headcol1, headcol2 = st.columns([0.93, 0.07])
    with headcol1:
        st.markdown(
            f'''
            <div id="chatbot-header">
                <img src="data:image/png;base64,{icon_b64}" alt="AtaraxBot" style="height:36px;width:35px;margin-right:13px;vertical-align:middle;display:inline;">
                <span id="chatbot-header-title">I'm AtaraxBot, Your Mental Health Assistant</span>
            </div>
            ''',
            unsafe_allow_html=True
        )
    with headcol2:
        if st.button("✖", key="chatbot_close_btn"):
            st.session_state.chatbot_open = False

    st.markdown('<div id="char-limit-note">Max 250 characters per question.</div>', unsafe_allow_html=True)

    # Show chat history
    for chat_msg in st.session_state.chatbot_history:
        st.markdown('<div class="msg-pair">', unsafe_allow_html=True)
        user_msg = chat_msg.get("user")
        bot_msg = chat_msg.get("assistant")
        st.markdown(f'<div class="msg-row"><div class="message user">{user_msg}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-row"><div class="message assistant">{bot_msg}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Typing indicator (if bot_typing True)
    if st.session_state.bot_typing and st.session_state.chatbot_pending_question:
        st.markdown(
            '<div class="msg-row"><div class="message assistant">'
            '<span class="typing-indicator"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></span>'
            '</div></div>',
            unsafe_allow_html=True
        )

    # Chat input area (NO #chatbot-input-container)
    if st.session_state.chatbot_clear_input:
        st.session_state["chatbot_input"] = ""
        st.session_state.chatbot_clear_input = False

    # Place input field directly, without wrapper
    question = st.text_area(
        "",
        max_chars=250,
        height=50,
        key="chatbot_input",
        placeholder="Type your message here..."
    )
    send_btn = st.button("Send", key="chatbot_send_btn")

    # Handle send button
    if send_btn:
        if question.strip() == "":
            st.warning("Please enter a question before sending.")
        elif len(question) > 250:
            st.warning("Your question exceeds the 250 character limit.")
        else:
            st.session_state.bot_typing = True
            st.session_state.chatbot_pending_question = question  # Save user question
            st.session_state.chatbot_clear_input = True
            st.rerun()

    # Generate response after rerun
    if st.session_state.bot_typing and st.session_state.chatbot_pending_question:
        import time
        time.sleep(1.1)  # Show the typing dots for 1s

        clarify_prompt = f"""
You are AtaraxBot, a compassionate mental health assistant.  
Answer the user's question clearly and simply.

Rules:
- Do NOT generate information outside the scope of the user's question.
- Keep your response strictly relevant to the question asked.
- Do NOT guess or provide unrelated details.
- If you do not know the answer, politely state that you cannot provide information on that.
- Keep the explanation short, simple, and easy to understand.
- Avoid jargon, complex terms, or metaphors.

Question: {st.session_state.chatbot_pending_question}
"""
        # Replace this with your actual model call
        if st.session_state.get("USE_OLLAMA"):
            clarify_response = run_ollama_prompt(clarify_prompt)
        else:
            clarify_response = "This is a mock explanation. Replace with your model response."

        st.session_state.chatbot_history.append({
            "user": st.session_state.chatbot_pending_question,
            "assistant": clarify_response
        })
        st.session_state.bot_typing = False
        st.session_state.chatbot_pending_question = None
        st.rerun()