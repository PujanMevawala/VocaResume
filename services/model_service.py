# This module will contain all business logic for model API calls and response handling.

from typing import List, Dict, Any, Callable, Union
import google.generativeai as genai

# Provider dispatcher registry
# Signature: (input_text, pdf_mime, pdf_data, prompt, groq_client, pplx_client, model_name, max_output_tokens) -> str
ProviderHandler = Callable[[str, str, str, str, Any, Any, str, int], str]

DEFAULT_MAX_OUTPUT_TOKENS = 4096  # Default maximum output tokens

def _handle_google(input_text: str, pdf_mime: str, pdf_data: str, prompt: str, groq_client, pplx_client, model_name: str, max_output_tokens: int) -> str:
    model = genai.GenerativeModel(model_name)
    last_error = None
    for attempt in range(2):
        try:
            response = model.generate_content([
                input_text,
                {"mime_type": pdf_mime, "data": pdf_data},
                prompt
            ], generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=0.4  # Slightly increased for more nuanced responses
            ))
            return response.text
        except Exception as e:
            last_error = e
                # simple retry without delay (removed time dependency)
    # fallback
    try:
        fallback_model = genai.GenerativeModel("gemini-1.5-flash")
        response = fallback_model.generate_content([
            input_text,
            {"mime_type": pdf_mime, "data": pdf_data},
            prompt
        ], generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.4  # Consistent temperature for fallback
        ))
        return response.text
    except Exception as e2:
        return f"Error processing resume with Gemini: {str(last_error)} | Fallback failed: {str(e2)}"

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

from utils.text_utils import normalize_for_tts


def get_model_response(input_text: str, pdf_content: List[Dict[str, Any]], prompt: str, model_info: Dict[str, str], groq_client=None, pplx_client=None, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> Dict[str, str]:
    if not pdf_content:
        return {"display_md": "Error: No resume content provided.", "tts_text": "Error: No resume content provided."}
    pdf_data = pdf_content[0]["data"]
    pdf_mime = pdf_content[0]["mime_type"]
    provider = model_info.get("provider")
    model_name = model_info.get("model")
    if not provider or not model_name:
        err = f"Error: Model config missing provider or model (provider={provider}, model={model_name})"
        return {"display_md": err, "tts_text": err}
    handler = _PROVIDER_DISPATCH.get(provider)
    if not handler:
        err = f"Error: Unknown model provider '{provider}'"
        return {"display_md": err, "tts_text": err}
    # Debug print for provider/model
    print(f"[DEBUG] get_model_response: provider={provider}, model={model_name}")
    raw_md = handler(input_text, pdf_mime, pdf_data, prompt, groq_client, pplx_client, model_name, max_output_tokens)
    sanitized = normalize_for_tts(raw_md)
    return {"display_md": raw_md, "tts_text": sanitized}

## Removed benchmark_models: no longer used in streamlined UI
