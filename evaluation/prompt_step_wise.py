import json
import os

# ============================================================
# 1. Base system prompt (without experiences, used for main model)
# ============================================================
BASE_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

Before each of your responses, you will be provided with a set of **RELEVANT EXPERIENCES** selected specifically for the current step. These are curated insights from past successful research sessions. You should actively leverage these experiences to guide your strategy, tool usage, verification approach, and answer formulation for the current step.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:\n1. The 'arguments' JSON object must be empty: {}.\n2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.\n\nIMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.\n\nExample of a correct call:\n<tool_call>\n{\"name\": \"PythonInterpreter\", \"arguments\": {}}\n<code>\nimport numpy as np\n# Your code here\nprint(f\"The result is: {np.mean([1,2,3])}\")\n</code>\n</tool_call>", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries for Google Scholar."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "parse_file", "description": "This is a tool that can be used to parse multiple user uploaded local files such as PDF, DOCX, PPTX, TXT, CSV, XLSX, DOC, ZIP, MP4, MP3.", "parameters": {"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}, "description": "The file name of the user uploaded local files to be parsed."}}, "required": ["files"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: """


# ============================================================
# 2. Experience retrieval prompt (for the 72B retrieval model)
# ============================================================
EXP_RETRIEVAL_PROMPT = """You are an experience retrieval assistant for a deep research system. Your task is to analyze the current state of a multi-turn research conversation and select the most relevant experiences from a library that would best guide the researcher's next step.

## Available Experience Library
{experience_library}

## Current Research Conversation
{conversation_context}

## Your Task
Based on the conversation above, analyze:
1. What is the core research question?
2. What has been done so far? What information has been gathered?
3. What is the model likely to do next (search, visit, verify, synthesize)?
4. Which experiences from the library are most relevant for guiding the NEXT step?

Select 3-8 experiences that are most relevant to the current situation. Consider:
- The current phase of research (initial exploration vs. verification vs. synthesis)
- Any challenges or ambiguities encountered
- The type of information being sought
- Potential pitfalls the experiences can help avoid

## Output Format
You must output valid JSON only, no other text:
{{"selected_ids": ["G1", "G5", ...], "rationale": "Brief explanation of why these experiences are relevant for the current step"}}"""


# ============================================================
# 3. Experience injection format (inserted into user messages)
# ============================================================
EXP_INJECTION_TEMPLATE = """
**RELEVANT EXPERIENCES FOR THIS STEP:**
{selected_experiences}
"""


# ============================================================
# 4. Load experiences from file
# ============================================================
def load_experiences(exp_path="experiences.json"):
    with open(exp_path, "r", encoding="utf-8") as f:
        experiences = json.load(f)
    return experiences


def format_experience_library(experiences):
    """Format all experiences into a string for the retrieval model."""
    lines = []
    for exp in experiences:
        lines.append(exp["full_text"])
    return "\n".join(lines)


def format_selected_experiences(experiences, selected_ids):
    """Format only the selected experiences for injection into user messages."""
    selected = [exp for exp in experiences if exp["id"] in selected_ids]
    lines = []
    for exp in selected:
        lines.append(exp["full_text"])
    return "\n".join(lines)


def format_conversation_for_retrieval(messages):
    """Format the conversation history for the retrieval model to analyze."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            # Don't include full system prompt, just note it exists
            parts.append("[System prompt with tool definitions]")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n\n".join(parts)


# ============================================================
# 5. Reuse existing EXTRACTOR_PROMPT
# ============================================================
EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""
