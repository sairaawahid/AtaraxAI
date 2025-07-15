def generate_dynamic_prompt(
    user_input,
    emotion_tag=None,
    language="English",
    trauma_tag=None,
    tone="supportive",
    cbt_technique=None,
    resilience_focus=False,
    context_note=None,
    ethical_disclaimer=True,
    include_resources=False,
    multi_turn=False,
    response_length="detailed",
    cultural_sensitivity=None,
    user_name=None,
    coping_preferences=None,
    template_mode="structured"
):
    prompt = ""

    # --- Template Style ---
    if template_mode == "conversational":
        prompt += "You are a kind and thoughtful mental health coach having a conversation with the user."
    elif template_mode == "narrative":
        prompt += "You are writing a narrative reflection for the user's journal entry as a supportive AI companion."
    else:
        prompt += "\n### Instruction:\nYou are a compassionate AI mental health coach."

    # --- Attribute Integration ---
    if user_name and user_name.lower() != "None":
        prompt += f" The user is named {user_name}."
    if language and language.lower() != "english":
        prompt += f" Respond in {language}."
    if trauma_tag and trauma_tag.lower() != "None":
        prompt += f" The user has a history of {trauma_tag}-related trauma. Respond with extra sensitivity."
    if cultural_sensitivity and cultural_sensitivity.lower() != "none":
        prompt += f" Tailor your response with awareness of {cultural_sensitivity} cultural norms."
    if tone and tone.lower() != "None":
        prompt += f" Use a {tone} tone."
    if cbt_technique and cbt_technique.lower() != "None":
        prompt += f" Apply the {cbt_technique} technique from Cognitive Behavioral Therapy."
    if resilience_focus:
        prompt += " Emphasize the user's strengths and past coping strategies."
    if context_note and context_note.lower() != "None":
        prompt += f" This entry was written {context_note}."
    if coping_preferences and coping_preferences.lower() != "None":
        prompt += f" The user prefers coping through: {coping_preferences}."
    if response_length == "short":
        prompt += " Keep the response brief and to the point."
    elif response_length == "detailed":
        prompt += " Provide a thorough and reflective response."
    if multi_turn:
        prompt += " End the response with an open-ended reflective question."

    # --- Core Sections ---
    prompt += f"\n\n### Journal Entry:\n{user_input}"

    if emotion_tag:
        prompt += f"\n\n### Detected Emotion:\n{emotion_tag}"
    
    prompt += "\n\n### CBT Response:\n"
    prompt += "Provide a supportive CBT-based reflection addressing the journal entry above."

    if include_resources:
        prompt += "\n\n### Resources:\nSuggest offline self-help techniques."
    if ethical_disclaimer:
        prompt += ("\n\n_Note: This response is AI-generated and not a substitute for professional help._")

    return prompt
