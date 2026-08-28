from tqdm import tqdm
from deep_translator import GoogleTranslator
from itertools import chain
import copy
from .language_configuration import fix_code_language, INVERTED_LANGUAGES
from .logging_setup import logger
import re
import json
import time
import os

# --- IMPORTS ---
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

TRANSLATION_PROCESS_OPTIONS = [
    "gemini_flash",
    "gemini_pro",
    "groq_llama3",
    "hf_zephyr_7b_beta",
    "google_translator_batch",
    "google_translator",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "disable_translation",
]

DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "gemini_flash",
    "gemini_pro",
    "groq_llama3",
    "hf_zephyr_7b_beta",
    "google_translator",
    "disable_translation",
]

# ==============================================================================
# PROMPT CONTEXTUEL - Style Naturel YouTube
# ==============================================================================
CONTEXT_GOLD_DIGGER_PROMPT = """Tu es un traducteur expert en doublage français pour vidéos YouTube.

RÈGLES OBLIGATOIRES :
1. LONGUEUR : Le français doit être AUSSI COURT ou PLUS COURT que le texte original.
2. STYLE : Français naturel, fluide et percutant de jeunes (20-28 ans).
   - Tutoiement fluide et moderne.
   - Langage parlé dynamique (ex: mec, vas-y, c'est ouf, grave, etc. quand ça sonne naturel).
3. ADAPTATION : Traduis le sens et le ton des dialogues sans faire de traduction mot-à-mot robotique.
4. FORMAT : Renvoie STRICTEMENT un tableau JSON de chaînes de caractères contenant les traductions dans le même ordre exact.
Exemple attendu :
["Traduction phrase 1", "Traduction phrase 2", "Traduction phrase 3"]"""

# ==============================================================================
# FILTRE ANTI-PAGE D'ERREUR SERVEUR
# ==============================================================================
def is_corrupted_translation(text):
    """Détecte si un traducteur a renvoyé une page d'erreur HTTP au lieu d'une traduction."""
    if not text or not isinstance(text, str):
        return True
    bad_patterns = [
        "error 500", "server error", "that’s an error", "that's an error",
        "please try again later", "that's all we know", "html", "<body", "500."
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in bad_patterns)

# ==============================================================================
# BATCHING CONTEXTUEL UNIFIÉ
# ==============================================================================
def _batch_with_context(segments, batch_size, translate_func, desc, target_lang):
    translated = copy.deepcopy(segments)
    progress = tqdm(total=len(segments), desc=desc)
    context = []

    for start in range(0, len(segments), batch_size):
        end = min(start + batch_size, len(segments))
        batch = translated[start:end]
        batch_len = len(batch)

        previous = "\n".join([f"Précédent {i+1}: {c}" for i, c in enumerate(context[-3:])])
        lines_text = "\n".join([f"{i+1}. {seg['text'].strip()}" for i, seg in enumerate(batch)])

        full_prompt = (
            f"Contexte précédent :\n{previous}\n\n"
            f"Lignes à traduire ({batch_len} phrases) :\n{lines_text}\n\n"
            f"Renvoie UNIQUEMENT le tableau JSON de {batch_len} chaînes de caractères :"
        )

        translated_lines = None
        # Boucle de réessai automatique (3 tentatives avec pause)
        for attempt in range(3):
            try:
                translated_lines = translate_func(full_prompt, batch_len)
                if translated_lines and len(translated_lines) == batch_len:
                    if not any(is_corrupted_translation(line) for line in translated_lines):
                        break
            except Exception as e:
                logger.warning(f"Tentative {attempt+1} échouée pour le batch {start}-{end}: {e}")
            time.sleep(1.5 * (attempt + 1))

        if translated_lines and len(translated_lines) == batch_len:
            for j, trans in enumerate(translated_lines):
                clean = re.sub(r'^\s*[\d]+[\.\)\-\s]+', '', str(trans)).strip()
                clean = re.sub(r'^["\']|["\']$', '', clean).strip()
                translated[start + j]["text"] = clean
            context.extend(translated_lines)
        else:
            # Fallback sécurisé : Google Translate avec filtre anti-erreur 500
            logger.warning(f"Batch {start}-{end} basculé sur Google Translate de secours.")
            tr = GoogleTranslator(source='auto', target=fix_code_language(target_lang))
            for seg in batch:
                orig_text = seg["text"].strip()
                try:
                    res = tr.translate(orig_text)
                    if res and not is_corrupted_translation(res):
                        seg["text"] = res
                    else:
                        seg["text"] = orig_text
                except Exception:
                    seg["text"] = orig_text

        progress.update(batch_len)
        time.sleep(0.5)

    progress.close()
    return translated

# ==============================================================================
# GEMINI (3.1 Pro & 2.0 Flash) - API Google GenAI
# ==============================================================================
def gemini_translate(segments, target, source=None, mode="flash"):
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key") or ""
    if not api_key:
        logger.error("❌ GEMINI: Clé GOOGLE_API_KEY manquante dans l'environnement !")
        return translate_iterative(segments, target, source)

    # Modèle 3.1 Pro pour le mode pro, et Flash 2.0/3.x pour le mode flash
    model_id = "gemini-3.1-pro-preview" if mode == "pro" else "gemini-2.0-flash"
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=2048,
        system_instruction=CONTEXT_GOLD_DIGGER_PROMPT,
        response_mime_type="application/json"
    )

    def call_gemini(full_prompt, batch_len):
        response = client.models.generate_content(model=model_id, contents=full_prompt, config=config)
        raw_text = response.text.strip()
        
        try:
            data = json.loads(raw_text)
            if isinstance(data, list):
                return [str(x).strip() for x in data][:batch_len]
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return [str(x).strip() for x in v][:batch_len]
        except Exception:
            pass

        lines = [re.sub(r'^\s*[\d]+[\.\)\-\s]+', '', l).strip() for l in raw_text.split('\n') if l.strip()]
        lines = [l for l in lines if not l.startswith('[') and not l.startswith(']')]
        return lines[:batch_len]

    return _batch_with_context(segments, 20, call_gemini, f"Translating (Gemini {mode.upper()} Batch)", target)

# ==============================================================================
# GROQ (LLaMA-3.3 70B)
# ==============================================================================
def groq_translate(segments, target, source=None):
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key") or ""
    if not api_key:
        logger.error("❌ GROQ: Clé GROQ_API_KEY manquante !")
        return translate_iterative(segments, target, source)

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, http_client=httpx.Client(timeout=60))

    def call_groq(full_prompt, batch_len):
        chat = client.chat.completions.create(
            messages=[
                {"role": "system", "content": CONTEXT_GOLD_DIGGER_PROMPT},
                {"role": "user", "content": full_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        content = chat.choices[0].message.content.strip()
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [str(x).strip() for x in data][:batch_len]
            for v in data.values():
                if isinstance(v, list):
                    return [str(x).strip() for x in v][:batch_len]
        except Exception:
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            return lines[:batch_len]
        return None

    return _batch_with_context(segments, 20, call_groq, "Translating (Groq LLaMA-3.3 Batch)", target)

# ==============================================================================
# ZEPHYR (Hugging Face Inference)
# ==============================================================================
def hf_zephyr_translate(segments, target, source=None, batch_size=15):
    hf_token = os.getenv("HF_TOKEN") or os.getenv("YOUR_HF_TOKEN") or ""
    if not hf_token or not hf_token.startswith("hf_"):
        logger.error("❌ ZEPHYR: HF_TOKEN manquant !")
        return translate_iterative(segments, target, source)

    client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_token)

    def call_zephyr(full_prompt, batch_len):
        prompt = f"<|system|>\n{CONTEXT_GOLD_DIGGER_PROMPT}</s>\n<|user|>\n{full_prompt}</s>\n<|assistant|>"
        response = client.text_generation(prompt, max_new_tokens=1500, temperature=0.3, return_full_text=False)
        lines = [re.sub(r'^\s*[\d]+[\.\)\-\s]+', '', l).strip() for l in response.strip().split('\n') if l.strip()]
        return lines[:batch_len]

    return _batch_with_context(segments, batch_size, call_zephyr, "Translating (Zephyr Batch)", target)

# ==============================================================================
# GOOGLE TRANSLATE FALLBACK
# ==============================================================================
def translate_iterative(segments, target, source=None):
    segments_ = copy.deepcopy(segments)
    if not source: source = "auto"
    target_clean = fix_code_language(target)
    translator = GoogleTranslator(source=source, target=target_clean)
    for line in tqdm(range(len(segments_)), desc="Translating (Iterative)"):
        text = segments_[line]["text"]
        try:
            res = translator.translate(text.strip())
            if res and not is_corrupted_translation(res):
                segments_[line]["text"] = res
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
            if is_corrupted_translation(translated_line):
                raise ValueError("Page d'erreur détectée dans Google Translate")
            split_text = translated_line.split("|||||")
            if len(split_text) == len(text_iterable):
                progress_bar.update(len(split_text))
            else:
                split_text = []
                for txt_iter in text_iterable:
                    translated_txt = translator.translate(txt_iter.strip())
                    if is_corrupted_translation(translated_txt):
                        translated_txt = txt_iter
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
            clean_text = translated_lines[line].replace("\t", "").replace("\n", "").strip()
            if not is_corrupted_translation(clean_text):
                segments_copy[line]["text"] = clean_text
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
    translation_process="gemini_flash",
    chunk_size=4500,
    source=None,
    token_batch_limit=1000,
):
    target_clean = fix_code_language(target)
    source_clean = fix_code_language(source) if source else "auto"
    match translation_process:
        case "gemini_flash":
            return gemini_translate(segments, target, source, mode="flash")
        case "gemini_pro":
            return gemini_translate(segments, target, source, mode="pro")
        case "groq_llama3":
            return groq_translate(segments, target, source)
        case "hf_zephyr_7b_beta":
            return hf_zephyr_translate(segments, target, source)
        case "google_translator_batch":
            return translate_batch(segments, target_clean, chunk_size, source_clean)
        case "google_translator":
            return translate_iterative(segments, target_clean, source_clean)
        case model if "gpt" in model:
            return translate_iterative(segments, target_clean, source_clean)
        case "disable_translation":
            return segments
        case _:
            return translate_iterative(segments, target_clean, source_clean)
