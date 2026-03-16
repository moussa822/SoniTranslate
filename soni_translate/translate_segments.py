from tqdm import tqdm
from deep_translator import GoogleTranslator
from itertools import chain
import copy
from .language_configuration import fix_code_language, INVERTED_LANGUAGES
from .logging_setup import logger
import re
import time
import os

# --- IMPORTS COMPATIBLES SDK V2 (2026) ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    pass
try:
    from openai import OpenAI
except ImportError:
    pass
try:
    from huggingface_hub import InferenceClient
except ImportError:
    pass
try:
    import httpx
except ImportError:
    pass
# -----------------------------------------

TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "google_translator",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "gemini_flash",
    "gemini_pro",
    "groq_llama3",
    "hf_zephyr_7b_beta",
    "disable_translation",
]

DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "google_translator",
    "gemini_flash",
    "groq_llama3",
    "hf_zephyr_7b_beta",
    "disable_translation",
]

# ==============================================================================
# PROMPT FINAL - Conversation naturelle (sans numéros forcés)
# ==============================================================================
CONTEXT_GOLD_DIGGER_PROMPT = """Tu es un traducteur expert en doublage français pour vidéos YouTube "Gold Digger Prank".

RÈGLES OBLIGATOIRES :
1. LONGUEUR : Le français doit être AUSSI COURT ou PLUS COURT que l'anglais original.
2. STYLE : Français naturel de jeunes (22-28 ans) en conversation réelle.
   - Tutoiement fluide et naturel.
   - Langage courant : mec, frère, vas-y, sérieux ?, c'est ouf, grave, etc. (seulement quand ça sonne vrai).
3. ADAPTATION : Transforme le slang américain en français courant et naturel.
4. CONTEXTE : Tiens compte des lignes précédentes pour que la conversation coule naturellement.

Réponds UNIQUEMENT avec les traductions, une par ligne, sans numéros, sans explications."""

# ==============================================================================
# FONCTION BATCH + CONTEXTE GLOBAL (utilisée par tous les modèles)
# ==============================================================================
def _batch_with_context(segments, batch_size, translate_func, desc):
    translated = copy.deepcopy(segments)
    progress = tqdm(total=len(segments), desc=desc)
    context = []  # garde les 3 dernières lignes pour le contexte

    for start in range(0, len(segments), batch_size):
        end = min(start + batch_size, len(segments))
        batch = translated[start:end]
        batch_len = len(batch)

        previous = "\n".join([f"Précédent {i+1}: {c}" for i, c in enumerate(context[-3:])])
        lines_text = "\n".join([f"{i+1}. {seg['text'].strip()}" for i, seg in enumerate(batch)])

        full_prompt = f"{CONTEXT_GOLD_DIGGER_PROMPT}\n\nContexte précédent :\n{previous}\n\nTraduis maintenant ces lignes :\n{lines_text}"

        translated_lines = translate_func(full_prompt, batch_len)

        if translated_lines and len(translated_lines) == batch_len:
            for j, trans in enumerate(translated_lines):
                # Nettoyage automatique des numéros (1. , 1) , 1- etc.)
                clean = re.sub(r'^\s*[\d]+[\.\)\-\s]+', '', trans).strip()
                translated[start + j]["text"] = clean
            context.extend(translated_lines)
        else:
            # Fallback Google
            tr = GoogleTranslator(source='auto', target=fix_code_language(target))
            for seg in batch:
                try:
                    seg["text"] = tr.translate(seg["text"].strip())
                except:
                    pass

        progress.update(batch_len)
        time.sleep(1.8)

    progress.close()
    return translated

# ==============================================================================
# GEMINI - Batch + Contexte
# ==============================================================================
def gemini_translate(segments, target, source=None, mode="flash"):
    api_key = "AIzaSyTaCléGeminiIciColleLaVraieCléComplète"
    if not api_key or len(api_key) < 30:
        logger.error("❌ GEMINI: Clé manquante !")
        return translate_iterative(segments, target, source)

    model_id = "gemini-3.1-pro-preview" if mode == "pro" else "gemini-flash-latest"
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=0.25, max_output_tokens=1500, system_instruction=CONTEXT_GOLD_DIGGER_PROMPT)

    def call_gemini(full_prompt, batch_len):
        try:
            response = client.models.generate_content(model=model_id, contents=full_prompt, config=config)
            lines = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            return lines[:batch_len]
        except:
            return None

    return _batch_with_context(segments, 15, call_gemini, f"Translating (Gemini {mode.upper()} BATCH CONTEXT)")

# ==============================================================================
# GROQ - Batch + Contexte
# ==============================================================================
def groq_translate(segments, target, source=None):
    api_key = "gsk_taCléGroqIciColleLaVraieClé"
    if not api_key:
        logger.error("❌ GROQ: Clé manquante !")
        return translate_iterative(segments, target, source)

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, http_client=httpx.Client(timeout=60))

    def call_groq(full_prompt, batch_len):
        try:
            chat = client.chat.completions.create(
                messages=[{"role": "system", "content": CONTEXT_GOLD_DIGGER_PROMPT},
                          {"role": "user", "content": full_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
            )
            lines = [line.strip() for line in chat.choices[0].message.content.strip().split('\n') if line.strip()]
            return lines[:batch_len]
        except:
            return None

    return _batch_with_context(segments, 18, call_groq, "Translating (Groq BATCH CONTEXT)")

# ==============================================================================
# ZEPHYR - Batch + Contexte (ultra rapide)
# ==============================================================================
def hf_zephyr_translate(segments, target, source=None, batch_size=18):
    hf_token = "hf_taCléHuggingFaceIciColleLaVraieToken"
    if not hf_token or not hf_token.startswith("hf_"):
        logger.error("❌ ZEPHYR: Clé manquante !")
        return translate_iterative(segments, target, source)

    client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_token)

    def call_zephyr(full_prompt, batch_len):
        try:
            response = client.text_generation(full_prompt, max_new_tokens=1400, temperature=0.35, top_p=0.9, return_full_text=False)
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            return lines[:batch_len]
        except:
            return None

    return _batch_with_context(segments, batch_size, call_zephyr, "Translating (Zephyr BATCH CONTEXT)")

# ==============================================================================
# FONCTIONS RESTANTES (inchangées)
# ==============================================================================
def translate_iterative(segments, target, source=None):
    segments_ = copy.deepcopy(segments)
    if not source: source = "auto"
    translator = GoogleTranslator(source=source, target=target)
    for line in tqdm(range(len(segments_))):
        text = segments_[line]["text"]
        try:
            translated_line = translator.translate(text.strip())
            segments_[line]["text"] = translated_line
        except Exception as e:
            logger.error(f"Error google iterative: {e}")
    return segments_

def translate_batch(segments, target, chunk_size=2000, source=None):
    segments_copy = copy.deepcopy(segments)
    if not source: source = "auto"
    text_lines = [seg["text"].strip() for seg in segments_copy]
    text_merge = []
    actual_chunk = ""
    global_text_list = []
    actual_text_list = []
    for one_line in text_lines:
        one_line = " " if not one_line else one_line
        if (len(actual_chunk) + len(one_line)) <= chunk_size:
            if actual_chunk: actual_chunk += " ||||| "
            actual_chunk += one_line
            actual_text_list.append(one_line)
        else:
            text_merge.append(actual_chunk)
            actual_chunk = one_line
            global_text_list.append(actual_text_list)
            actual_text_list = [one_line]
    if actual_chunk:
        text_merge.append(actual_chunk)
        global_text_list.append(actual_text_list)

    progress_bar = tqdm(total=len(segments), desc="Translating (Google Batch)")
    translator = GoogleTranslator(source=source, target=target)
    split_list = []
   
    try:
        for text, text_iterable in zip(text_merge, global_text_list):
            translated_line = translator.translate(text.strip())
            split_text = translated_line.split("|||||")
            if len(split_text) == len(text_iterable):
                progress_bar.update(len(split_text))
            else:
                split_text = []
                for txt_iter in text_iterable:
                    translated_txt = translator.translate(txt_iter.strip())
                    split_text.append(translated_txt)
                    progress_bar.update(1)
            split_list.append(split_text)
        progress_bar.close()
    except Exception:
        progress_bar.close()
        return translate_iterative(segments, target, source)
    translated_lines = list(chain.from_iterable(split_list))
    return verify_translate(segments, segments_copy, translated_lines, target, source)

def verify_translate(segments, segments_copy, translated_lines, target, source):
    if len(segments) == len(translated_lines):
        for line in range(len(segments_copy)):
            segments_copy[line]["text"] = translated_lines[line].replace("\t", "").replace("\n", "").strip()
        return segments_copy
    else:
        return translate_iterative(segments, target, source)

def gpt_sequential(segments, model, target, source=None):
    return translate_iterative(segments, target, source)

def gpt_batch(segments, model, target, token_batch_limit=900, source=None):
    return translate_iterative(segments, target, source)

def translate_text(
    segments,
    target,
    translation_process="google_translator_batch",
    chunk_size=4500,
    source=None,
    token_batch_limit=1000,
):
    target_clean = fix_code_language(target)
    source_clean = fix_code_language(source) if source else "auto"
    match translation_process:
        case "google_translator_batch":
            return translate_batch(segments, target_clean, chunk_size, source_clean)
        case "google_translator":
            return translate_iterative(segments, target_clean, source_clean)
        case "gemini_flash":
            return gemini_translate(segments, target, source, mode="flash")
        case "gemini_pro":
            return gemini_translate(segments, target, source, mode="pro")
        case "groq_llama3":
            return groq_translate(segments, target, source)
        case "hf_zephyr_7b_beta":
            return hf_zephyr_translate(segments, target, source)
        case model if "gpt" in model:
            return translate_iterative(segments, target_clean, source_clean)
        case "disable_translation":
            return segments
        case _:
            return translate_iterative(segments, target_clean, source_clean)
