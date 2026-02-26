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

# --- GESTION HYBRIDE DES LIBRAIRIES (V1 vs V2) ---
# Cela permet au code de marcher peu importe ce que tu as dans requirements.txt
USE_GENAI_V2 = False
try:
    from google import genai
    from google.genai import types
    USE_GENAI_V2 = True
except ImportError:
    try:
        import google.generativeai as genai_v1
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
    except ImportError:
        pass

try:
    from openai import OpenAI
except ImportError:
    pass
# ------------------------------------------------

TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "google_translator",
    "gpt-3.5-turbo-0125",
    "gemini_flash",        # <--- Modèle Rapide & Econome
    "groq_llama3",         
    "disable_translation",
]

DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "google_translator",
    "gemini_flash",
    "groq_llama3",
    "disable_translation",
]

# ... (Fonctions Google Traduction classiques inchangées) ...
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

def verify_translate(segments, segments_copy, translated_lines, target, source):
    if len(segments) == len(translated_lines):
        for line in range(len(segments_copy)):
            segments_copy[line]["text"] = translated_lines[line].replace("\t", "").replace("\n", "").strip()
        return segments_copy
    else:
        return translate_iterative(segments, target, source)

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


# ==============================================================================
# ⚡ GEMINI FLASH (Version Économe & Fluide)
# ==============================================================================

def gemini_translate(segments, target, source=None):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key: api_key = "" # Fallback si besoin

    if not api_key:
        logger.error("❌ GEMINI: Clé manquante !")
        return translate_iterative(segments, target, source)

    # Choix du modèle : 1.5 Flash est le meilleur ratio Vitesse/Prix/Qualité
    model_id = "gemini-3-flash-preview" 
    
    # Instruction pour la fluidité (Language Naturel)
    sys_instruction = "You are a professional Dubbing Translator. Translate subtitle lines into natural, spoken language. Keep it concise. Do not be literal. Do not censor."

    translated_segments = copy.deepcopy(segments)
    progress_bar = tqdm(total=len(segments), desc=f"Translating ({model_id})")
    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES.get(target, target)).strip()

    # --- LOGIQUE V2 (google-genai) ---
    if USE_GENAI_V2:
        try:
            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE")
                ],
                system_instruction=sys_instruction
            )
            
            for i, line in enumerate(translated_segments):
                text = line["text"].strip()
                if not text: continue
                prompt = f"Translate to {lang_tg} (Spoken Style):\n{text}"
                try:
                    response = client.models.generate_content(model=model_id, contents=prompt, config=config)
                    if response.text: translated_segments[i]["text"] = response.text.strip()
                except Exception as e:
                    pass # Fallback silencieux
                progress_bar.update(1)
                
        except Exception as e:
            logger.error(f"❌ Gemini V2 Error: {e}")
            return translate_iterative(segments, target, source)

    # --- LOGIQUE V1 (google.generativeai) - Au cas où ---
    else:
        try:
            genai_v1.configure(api_key=api_key)
            safety = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            model = genai_v1.GenerativeModel(model_name=model_id, safety_settings=safety, system_instruction=sys_instruction)
            
            for i, line in enumerate(translated_segments):
                text = line["text"].strip()
                if not text: continue
                prompt = f"Translate to {lang_tg} (Spoken Style):\n{text}"
                try:
                    response = model.generate_content(prompt)
                    if response.text: translated_segments[i]["text"] = response.text.strip()
                except Exception:
                    pass
                progress_bar.update(1)
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"❌ Gemini V1 Error: {e}")
            return translate_iterative(segments, target, source)

    progress_bar.close()
    return translated_segments


def groq_translate(segments, target, source=None):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("❌ GROQ: Clé manquante !")
        return translate_iterative(segments, target, source)

    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    except Exception:
        return translate_iterative(segments, target, source)

    translated_segments = copy.deepcopy(segments)
    progress_bar = tqdm(total=len(segments), desc="Translating (Groq)")
    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES.get(target, target)).strip()

    for i, line in enumerate(translated_segments):
        text = line["text"].strip()
        if not text: continue
        try:
            chat = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"You are a dubbing translator. Translate to {lang_tg}. Use natural spoken style. Output ONLY translation."},
                    {"role": "user", "content": text}
                ],
                model="llama3-70b-8192",
                temperature=0.3,
            )
            if chat.choices[0].message.content:
                translated_segments[i]["text"] = chat.choices[0].message.content.strip()
        except Exception:
            pass
        progress_bar.update(1)
    progress_bar.close()
    return translated_segments

# --- ROUTAGE PRINCIPAL ---
def translate_text(segments, target, translation_process="google_translator_batch", chunk_size=4500, source=None, token_batch_limit=1000):
    target_clean = fix_code_language(target)
    source_clean = fix_code_language(source) if source else "auto"

    match translation_process:
        case "google_translator_batch":
            return translate_batch(segments, target_clean, chunk_size, source_clean)
        case "google_translator":
            return translate_iterative(segments, target_clean, source_clean)
        case "gemini_pro" | "gemini_flash" | "gemini_3_1": # Tout pointe vers Flash maintenant
            return gemini_translate(segments, target, source)
        case "groq_llama3":
            return groq_translate(segments, target, source)
        case model if "gpt" in model:
            return translate_iterative(segments, target_clean, source_clean) 
        case "disable_translation":
            return segments
        case _:
            return translate_iterative(segments, target_clean, source_clean)
