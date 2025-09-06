# This module will contain all business logic for model API calls and response handling.

from typing import List, Dict, Any, Callable
import google.generativeai as genai

# Provider dispatcher registry
# Signature: (input_text, pdf_mime, pdf_data, prompt, groq_client, pplx_client, model_name, max_output_tokens) -> str
ProviderHandler = Callable[[str, str, str, str, Any, Any, str, int], str]

DEFAULT_MAX_OUTPUT_TOKENS = 4096  # Default maximum output tokens

def _handle_google(input_text: str, pdf_mime: str, pdf_data: str, prompt: str, groq_client, pplx_client, model_name: str, max_output_tokens: int) -> str:
    """Call Gemini with resilience:
    - Retry once
    - On 429 quota: downgrade model & reduce tokens
    - If still failing: graceful explanatory message (not raw stack)
    """
    def _call(model_id: str, max_tokens: int):
        m = genai.GenerativeModel(model_id)
        return m.generate_content([
            input_text,
            {"mime_type": pdf_mime, "data": pdf_data},
            prompt
        ], generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=0.4
        ))

    last_error = None
    # Primary attempts
    for attempt in range(2):
        try:
            resp = _call(model_name, max_output_tokens)
            return resp.text
        except Exception as e:
            last_error = e
            # If not quota related, continue retry once
            if "429" not in str(e):
                continue
            # Quota: break to fallback path immediately
            break

    # Quota / failure fallback sequence
    quota_flag = last_error and "429" in str(last_error)
    fallback_models = [
        "gemini-1.5-flash",  # smallest cheap model
        "gemini-2.5-flash" if model_name != "gemini-2.5-flash" else "gemini-1.5-flash"
    ]
    for fb in fallback_models:
        try:
            resp = _call(fb, min(1024, max_output_tokens // 2))
            return resp.text
        except Exception as e2:
            last_error = e2
            if "429" not in str(e2):
                continue

    # Final graceful message (do NOT surface raw giant exception text to UI)
    hint_lines = [
        "The selected Google Gemini model could not be used right now.",
        "Reason: quota or rate limit exceeded." if quota_flag else "Reason: repeated API invocation errors.",
        "Recommended actions:",
        "1. Switch to another provider (Groq or Perplexity) from the Model dropdown.",
        "2. Reduce Max Tokens slider and retry.",
        "3. Upgrade or verify Google AI Studio quota (billing & rate limits).",
        "4. For immediate continuity, re-run using a non-Google model." ,
    ]
    # If Groq available provide inline suggestion keyword
    if groq_client is None and pplx_client is None:
        hint_lines.append("Tip: Add GROQ_API_KEY or PPLX_API_KEY in your environment for automatic fallback.")
    return "\n".join(hint_lines)

def _handle_groq(input_text: str, pdf_mime: str, pdf_data: str, prompt: str, groq_client, pplx_client, model_name: str, max_output_tokens: int) -> str:
    if groq_client is None:
        return "GROQ client not initialized."
    full_input = f"{input_text}\n\nResume Content: {pdf_data[:200]}...\n\n{prompt}"
    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": full_input}],
            max_tokens=max_output_tokens,
            temperature=0.4,  # Slightly increased for more nuanced responses
            timeout=15
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing resume with Groq: {str(e)}"

def _handle_perplexity(input_text: str, pdf_mime: str, pdf_data: str, prompt: str, groq_client, pplx_client, model_name: str, max_output_tokens: int) -> str:
    if pplx_client is None:
        return "Perplexity client not initialized or API key missing."
    full_input = f"{input_text}\n\nResume Content: {pdf_data[:200]}...\n\n{prompt}"
    try:
        response = pplx_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": full_input}],
            temperature=0.4,  # Slightly increased for more nuanced responses
            max_tokens=max_output_tokens,
            timeout=15
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing resume with Perplexity: {str(e)}"

_PROVIDER_DISPATCH = {
    "google": _handle_google,
    "groq": _handle_groq,
    "perplexity": _handle_perplexity,
}

# Model API clients will be initialized in app.py and passed here if needed

def get_model_response(input_text: str, pdf_content: List[Dict[str, Any]], prompt: str, model_info: Dict[str, str], groq_client=None, pplx_client=None, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> str:
    if not pdf_content:
        return "Error: No resume content provided."
    pdf_data = pdf_content[0]["data"]
    pdf_mime = pdf_content[0]["mime_type"]
    provider = model_info.get("provider")
    model_name = model_info.get("model")
    handler = _PROVIDER_DISPATCH.get(provider)
    if not handler:
        return f"Error: Unknown model provider '{provider}'"
    return handler(input_text, pdf_mime, pdf_data, prompt, groq_client, pplx_client, model_name, max_output_tokens)

def general_query_response(query: str, resume_content: List[Dict[str, Any]], job_description: str, prior: list[str], model_info: Dict[str,str], groq_client=None, pplx_client=None, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> str:
    """Generate a contextual answer for a free-form query referencing resume & job description.
    prior: previous response snippets to ground follow-ups.
    """
    if not resume_content:
        return "No resume provided. Upload a resume PDF to enable contextual answers."
    pdf_data = resume_content[0]["data"]
    pdf_mime = resume_content[0]["mime_type"]
    provider = model_info.get("provider")
    model_name = model_info.get("model")
    handler = _PROVIDER_DISPATCH.get(provider)
    if not handler:
        return f"Error: Unknown model provider '{provider}'"
    context_snippets = "\n---\n".join(prior[-3:]) if prior else ""
    prompt = f"You are an AI career assistant. Use ONLY the provided resume, job description, and chat context. If unsure, say you are unsure.\n\nResume (excerpt):\n{pdf_data}\n\nJob Description:\n{job_description}\n\nConversation Context:\n{context_snippets}\n\nUser Query:\n{query}\n\nConstraints:\n- Be concise but helpful.\n- Avoid hallucinations; cite 'Based on resume' or 'Not present in resume' explicitly.\n- If user asks for analysis categories, you may produce structured bullets.\nProvide answer now:"
    return handler(job_description, pdf_mime, pdf_data, prompt, groq_client, pplx_client, model_name, max_output_tokens)

def refine_voice_query(raw_transcript: str, job_description: str, resume_excerpt: str, model_info: Dict[str,str], groq_client=None, pplx_client=None) -> str:
    """Use the selected model provider to turn a raw voice transcript into a concise, well-structured query.
    Output format MUST be JSON with keys: optimized_query, probable_intent (one of analysis|interview|suggestions|job_fit|freeform), rationale.
    """
    if not raw_transcript:
        return '{"optimized_query":"","probable_intent":"freeform","rationale":"Empty transcript"}'
    provider = model_info.get("provider")
    model_name = model_info.get("model")
    handler = _PROVIDER_DISPATCH.get(provider)
    if not handler:
        return '{"optimized_query":"'+raw_transcript+'","probable_intent":"freeform","rationale":"Unknown provider"}'
    # Craft a short system-style prompt (we pass via user content due to handler signature)
    guiding_prompt = (
        "You are an intent normalizer for a career assistant. Given a raw spoken user transcript, job description context, and partial resume excerpt, produce a minimal JSON.\n"
        "Rules:\n- optimized_query: cleaned, grammar-fixed, single sentence if possible.\n"
        "- probable_intent: choose strictly among analysis|interview|suggestions|job_fit|freeform.\n"
        "- rationale: one short sentence why you chose that intent.\n"
        "Return ONLY JSON. No markdown.\n"
        f"Raw Transcript: {raw_transcript}\nJob Description (trimmed): {job_description[:800]}\nResume Excerpt: {resume_excerpt[:800]}\nJSON:"
    )
    # Minimal shim: pdf bytes not needed; pass placeholders
    result = handler(job_description[:50], 'text/plain', resume_excerpt[:50], guiding_prompt, groq_client, pplx_client, model_name, 512)
    # Basic validation fallback
    if '{' not in result:
        return '{"optimized_query":"'+raw_transcript+'","probable_intent":"freeform","rationale":"Model returned non-JSON"}'
    return result.strip()

## Removed benchmark_models: no longer used in streamlined UI
