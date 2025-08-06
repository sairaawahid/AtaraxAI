def generate_dynamic_prompt(
    user_input,
    context_block=None,
    emotion_tag=None,
    language="English",
    trauma_tag=None,
    tone="supportive",
    cbt_technique=None,
    resilience_focus=False,
    context_note=None,
    include_resources=False,
    response_length="detailed",
    user_name=None,
    coping_preferences=None
):
    """
    Generate a context-aware, modular prompt for AtaraxAI (CBT Mental Health Assistant).
    
    This function uses the Hybrid Prompt Builder pattern:
    - Prompt sections are stored as multi-line string blocks for readability.
    - Only the blocks relevant to the user/session are included (using if-condition logic).
    
    Sections:
        - Core instructions for the LLM
        - (Optional) User memory & recent conversation (context)
        - (Optional) User personalization (name, trauma, tone, technique, etc.)
        - Journal entry and emotion tag
        - Structure for CBT response, resources, and disclaimers
    """

    # --- Prompt Section Templates (Hybrid Pattern) ---
    CORE_INSTRUCTION_BLOCK = (
        "You are AtaraxAI: a private, offline, compassionate AI mental health companion specializing in Cognitive Behavioral Therapy (CBT). "
        "Your task is to provide a human-centered, emotionally-aware, supportive CBT response to the user's journal entry."
    )
    RESPONSE_RULES_BLOCK = (
    "\n\nRules for this response:"
    "\n- Use simple and easy-to-understand language."
    "\n- Avoid jargon, complex terms, or metaphors."
    "\n- Respond using the user's words and themes naturally where possible."
    "\n- Do NOT generate information outside the context of the user's journal entry."
    "\n- Keep your response strictly relevant to the journal entry."
    "\n- Do NOT guess or provide unrelated details."
    "\n- If you do not know the answer, politely state that you cannot provide information on that."
    "\n- Keep the explanation short, clear, and easy to understand."
    )
    CONTEXT_BLOCK = (
        "\n\nYou have access to the following user memory and recent conversation. "
        "Leverage this information to personalize your response, maintain continuity, and avoid repetition. "
        "\n\n==== CONTEXT ====\n"
        "{context}"
        "\n==== END CONTEXT ====\n"
    )
    USER_NAME_BLOCK = " The user's name is {user_name}."
    TRAUMA_BLOCK = " The user has a history of {trauma_tag}-related trauma. Respond with extra sensitivity."
    TONE_BLOCK = " Use a {tone} tone."
    CBT_BLOCK = " Apply the {cbt_technique} technique from Cognitive Behavioral Therapy."
    RESILIENCE_BLOCK = " Emphasize the user's strengths and past coping strategies."
    CONTEXT_NOTE_BLOCK = " This entry was written {context_note}."
    COPING_BLOCK = " The user prefers coping through: {coping_preferences}."
    LENGTH_BLOCK_SHORT = " Keep the response brief and to the point."
    LENGTH_BLOCK_DETAILED = " Provide a thorough and reflective response."
    MULTITURN_BLOCK = (
    "\n\nAlways end your response with an open-ended, reflective question directly relevant to the user's journal entry and the context that encourages the user to explore their thoughts and feelings further."
    "\nMake sure the question is directly related to the user's journal entry and any provided context. Do NOT ask generic or repetitive questions."
    "\nExample: After providing support, ask something like, 'What are some small steps you could try next?' or 'How did this experience affect your outlook?'"
    )
    JOURNAL_BLOCK = "\n\n### Journal Entry:\n{user_input}"
    EMOTION_BLOCK = "\n\n### Detected Emotion:\n{emotion_tag}"
    CBT_RESPONSE_HEADER = "\n\n### CBT Response:\nProvide a supportive CBT-based reflection addressing the journal entry above."
    RESOURCES_HEADER = "\n\n### Resources:\nSuggest offline self-help techniques."
    DISCLAIMER_BLOCK = (
        "\n\nDisclaimer: AtaraxAI is an AI assistant, not a licensed mental health professional.  \n"
        "If you are in crisis or need urgent help, please contact a professional or local emergency services."
    )

    # --- 1. Start with Core Instruction ---
    prompt = CORE_INSTRUCTION_BLOCK
    prompt += RESPONSE_RULES_BLOCK

    # --- 2. Add Context Block if Available ---
    if context_block and context_block.strip():
        prompt += CONTEXT_BLOCK.format(context=context_block.strip())

    # --- 3. Add User Personalization/Settings (conditionally) ---
    if user_name and user_name.lower() != "none":
        prompt += USER_NAME_BLOCK.format(user_name=user_name)
    if trauma_tag and trauma_tag.lower() != "none":
        prompt += TRAUMA_BLOCK.format(trauma_tag=trauma_tag)
    if tone and tone.lower() != "none":
        prompt += TONE_BLOCK.format(tone=tone)
    if cbt_technique and cbt_technique.lower() != "none":
        prompt += CBT_BLOCK.format(cbt_technique=cbt_technique)
    if resilience_focus:
        prompt += RESILIENCE_BLOCK
    if context_note and context_note.lower() != "none":
        prompt += CONTEXT_NOTE_BLOCK.format(context_note=context_note)
    if coping_preferences and coping_preferences.lower() != "none":
        prompt += COPING_BLOCK.format(coping_preferences=coping_preferences)
    if response_length == "short":
        prompt += LENGTH_BLOCK_SHORT
    elif response_length == "detailed":
        prompt += LENGTH_BLOCK_DETAILED

    # --- 4. Add Journal Entry and Emotion Tag ---
    prompt += JOURNAL_BLOCK.format(user_input=user_input)
    if emotion_tag:
        prompt += EMOTION_BLOCK.format(emotion_tag=emotion_tag)

    # --- 5. Add CBT Response Header ---
    prompt += CBT_RESPONSE_HEADER

    # --- 6. Optional: Add Resources and Disclaimer ---
    if include_resources:
        prompt += RESOURCES_HEADER

    # --- 7. Always Add Reflective Question Instruction ---
    prompt += MULTITURN_BLOCK

    # --- 8. Always Add Disclaimer ---
    prompt += DISCLAIMER_BLOCK

    return prompt










#     prompt = f"""
# You are AtaraxAI: a private, offline, compassionate AI mental health companion specializing in Cognitive Behavioral Therapy (CBT).
# Your task is to provide a *human-centered, supportive, emotionally-aware* Cognitive Behavioral Therapy response to the user's journal entry.
# Use the user's information and emotional cues to tailor your response, strictly following these instructions:

# ---

# **Formatting & Structure**
# - Write your response in paragraphs and give appropriate headings for each paragraph (no lists or bullet points unless instructed).
# - Your response must always reflect a one-on-one conversation between a mental health therapist and client. 
# - End with a gentle, open-ended reflective question.
# - List 4–5 simple, *offline*, practical self-help techniques and tips related to the user's situation.
# - Always include the following at the end:  
#   _Note: This response is AI-generated and not a substitute for professional help._

# **Stay in Context**
# - Your response must always stay directly relevant to the user's journal entry and emotional context.
# - Never introduce unrelated topics, stories, or generic advice not connected to the user's words.
# - Only include suggestions or observations that are natural extensions of what the user shared.

# **Word Choice & Language**
# - Use simple, clear, and direct language that feels natural in daily life.
# - Respond using the user's words and themes naturally where possible.
# - Avoid technical jargon, poetic or metaphorical language, and never use rhymes.
# - Use everyday words that anyone can understand.
# - Make the response human, not robotic.

# **Personalization Rules**
# - Use the user's name if provided ({user_name if user_name else "none"}).
# - Show extra care if trauma or sensitive topics are mentioned ({trauma_tag if trauma_tag else "none"}).
# - The emotional tone must be {tone.lower()}.
# - Always acknowledge and reflect the user's stated emotion: {emotion_tag if emotion_tag else "Not specified"}.

# **CBT-Specific Rules**
# - If a CBT technique is specified ({cbt_technique if cbt_technique else "none"}), clearly use it as the main approach.  
#   For example, if "Cognitive Reframing" is chosen, help the user challenge negative thoughts and suggest alternative perspectives.  
#   If "Socratic Questioning," use gentle probing questions about the user's beliefs.
# - If resilience focus is enabled, highlight personal strengths, past successes, or effective coping strategies.
# - If context notes are included ({context_note if context_note else "none"}), consider them when personalizing the response.
# - If coping preferences are given ({coping_preferences if coping_preferences else "none"}), naturally weave them into your suggestion or support.
# - Response length should match user preference: "{response_length.lower()}".

# **Additional Features**
# - If `include_resources` is enabled, after the main paragraph, add a new heading:  
#   Helpful Techniques  
#   - List 4–5 simple, *offline*, practical self-help techniques and tips related to the user's situation.


# ---

# **User Journal Entry:**  
# {user_input}

# {"Context Note: " + context_note if context_note else ""}

# ---

# **Format Example:**  
# Hey {user_name if user_name else "none"}. It's understandable that you’re feeling {emotion_tag if emotion_tag else 'overwhelmed'} after what you shared. Let's use {cbt_technique if cbt_technique else 'a CBT approach'}: take a moment to notice your thoughts and ask, "Is there another way to look at this situation?" Remember, it's okay to have these feelings, and you've managed tough days before. {('You might find writing in a gratitude journal or reaching out to a friend helpful.' if resilience_focus else '')}
# {"What is one small thing you could do today to care for yourself?" if multi_turn else ""}
# {f"\nHelpful Techniques\n- Try writing down one thing you managed today, however small.\n- Practice a brief breathing exercise when stress feels high." if include_resources else ""}
# {"\n_Note: This response is AI-generated and not a substitute for professional help._" if ethical_disclaimer else ""}

# """
#     return prompt
