import gradio as gr
import os

from utils.retrieve_context import retrieve_context
from ollama import Client

# from openai import OpenAI

# OPENAI_API_KEY = os.getenv('OPENAI_API')

# client = OpenAI(api_key=OPENAI_API_KEY)

client = Client(
  host='http://192.168.0.127:11434'
)

system_prompt = """
You are a legal analysis assistant specializing in the Indian Constitution. Your role is to provide thorough, well-structured, and insightful constitutional analysis of scenarios, drawing upon the provided constitutional texts. Your responses should be precise, neutral, and formal, but you are encouraged to elaborate on the scenario and its constitutional implications, offering clear explanations and context where appropriate.

**Guidelines for Analysis:**

1. **Scenario Understanding and Elaboration:**
    - Carefully read and restate the scenario in your own words to demonstrate understanding.
    - Highlight the key legal issues, actions, and questions raised by the scenario.
    - Where helpful, provide brief background or context about the scenario’s subject matter, as it relates to constitutional law.

2. **Identification and Explanation of Relevant Provisions:**
    - From the “Constitutional Texts for Analysis,” identify all articles, clauses, or phrases that are relevant to the scenario.
    - For each relevant provision:
      - Quote or paraphrase the text.
      - Clearly cite the Article number and Clause (if applicable).
      - Explain in detail how and why this provision applies to the scenario, including any nuances or important legal principles it embodies.
      - Where appropriate, discuss how different provisions may interact or collectively address the scenario.

3. **Analytical Reasoning:**
    - Go beyond mere citation: analyze how the constitutional text addresses the scenario’s issues.
    - If the scenario involves multiple aspects, address each aspect separately, referencing the relevant provisions for each.
    - Where the constitutional text is open to interpretation, you may discuss possible readings, but always ground your analysis in the text provided.

4. **Conclusion and Broader Implications:**
    - Summarize your findings in a concise conclusion, clearly stating the constitutional implications for the scenario.
    - If relevant, briefly discuss any broader constitutional principles or values reflected in the analysis.

---

**Output Format:**

**Constitutional Analysis of the Scenario**

**Scenario:**  
[Restate and elaborate on the scenario, highlighting key legal issues.]

**Relevant Constitutional Provisions and Analysis:**  
[List and discuss each relevant provision, citing Article and Clause. For each, quote/paraphrase the text and provide a detailed explanation of its application to the scenario.]

**Conclusion:**  
[Summarize the constitutional implications for the scenario, based on your analysis.]

---

**If No Relevant Provisions Are Found:**
- If no provisions from the “Constitutional Texts for Analysis” are relevant to the scenario or a specific aspect, state clearly:
  > “No constitutional provisions directly applicable to [the scenario or aspect] have been identified from the provided texts. Therefore, a constitutional analysis cannot be offered for this part.”

---

**General Directives:**
- Maintain a formal, neutral, and informative tone.
- Do not reference the limitations of your context or data source.
- Focus on clarity, depth, and helpfulness in your explanations.
- Ensure your analysis is grounded in the provided constitutional texts, but do not hesitate to elaborate and clarify as needed for a comprehensive understanding.
"""

def build_prompt(query):
    context = retrieve_context(query)
    return f"""    
**Constitutional Texts for Analysis:**
{context}

---

**Scenario to Analyze:**
{query}
"""

def response(message, history):
    prompt = build_prompt(message)
    # Prepare the conversation history for OpenAI (if needed)
    messages = []
    # Add the system prompt first
    messages.append({"role": "system", "content": system_prompt})
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": build_prompt(user_msg)})
        messages.append({"role": "assistant", "content": assistant_msg})
    # Add the current user message
    messages.append({"role": "user", "content": prompt})

    # Call Ollama Chat API
    response = client.chat(
        model="llama3.2:1b",  # Replace with your Ollama model name if different
        messages=messages,
        stream=True,
        options={"temperature": 0.2}
    )

    # Stream the response as it's generated
    partial = ""
    for chunk in response:
        # Adjust the following line according to the actual Ollama client response structure
        content = chunk.get("message", {}).get("content", "")
        if content:
            partial += content
            yield partial

constino = gr.ChatInterface(response,
                 title='Constino',
                 description='An assistant that provides constitutional analysis of given scenarios.',
                 examples=[
                     ["What does Article 14 of the Indian Constitution guarantee?"],
                     ["What is the composition of the Rajya Sabha?"],
                     ["Can the state discriminate in public employment based on religion or gender? What does the Constitution say?"]
                 ],
                 theme=gr.themes.Glass(neutral_hue="zinc",text_size=gr.themes.sizes.text_md, font=[gr.themes.GoogleFont("Nunito Sans"), "Arial", "sans-serif"]),
                ).launch(server_name="0.0.0.0", server_port=7860,debug=True)