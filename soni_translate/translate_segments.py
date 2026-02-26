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

# --- GESTION DES IMPORTS ---
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    pass 

try:
    from openai import OpenAI
except ImportError:
    pass
# ---------------------------

TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "google_translator",
    "gpt-3.5-turbo-0125_batch",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview_batch",
    "gpt-4-turbo-preview",
    "gemini_pro",          # Notre Star
    "groq_llama3",         # Notre Turbo
    "disable_translation",
]

DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "google_translator",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "gemini_pro",
    "groq_llama3",
    "disable_translation",
]


def translate_iterative(segments, target, source=None):
    """Fallback de secours (Google Traduction classique)"""
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
    """Traduction par lot Google (Rapide mais basique)"""
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
    except Exception as error:
        progress_bar.close()
        return translate_iterative(segments, target, source)

    translated_lines = list(chain.from_iterable(split_list))
    return verify_translate(segments, segments_copy, translated_lines, target, source)


# ==============================================================================
# 🔥 GEMINI PRO 2.0 (DOUBLAGE FLUIDE & SANS CENSURE)
# ==============================================================================

def gemini_translate(segments, target, source=None):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = "AIzaSyDvzGjD5ZsvjUlfpd8CcY1ygm_N04OraXY"
    if not api_key:
        logger.error("❌ GEMINI: Clé manquante ! Passage à Google Trad.")
        return translate_iterative(segments, target, source)

    try:
        genai.configure(api_key=api_key)
        
        # 1. STOP CENSURE (Crucial pour les films)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # 2. CERVEAU DOUBLAGE
        # On utilise le modèle Flash Exp (2.0) pour la vitesse et l'intelligence
        # Si ça plante (accès refusé), remets 'gemini-1.5-pro'
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash-exp', 
            safety_settings=safety_settings,
            system_instruction="You are an expert Dubbing Translator. Your translations must be fluid, natural, and spoken-style. Do not be literal."
        )
    except Exception as e:
        logger.error(f"❌ GEMINI Config Error: {e}")
        return translate_iterative(segments, target, source)

    translated_segments = copy.deepcopy(segments)
    progress_bar = tqdm(total=len(segments), desc="Translating (Gemini Fluid)")
    
    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES.get(target, target)).strip()

    for i, line in enumerate(translated_segments):
        text = line["text"].strip()
        if not text: continue
        
        # 3. PROMPT DE FLUIDITÉ
        prompt = f"""Translate this line to {lang_tg} for audio dubbing.
        Make it sound natural and conversational.
        Original: "{text}"
        Translation:"""
        
        try:
            response = model.generate_content(prompt)
            
            if response and response.text:
                clean_text = response.text.replace("Translation:", "").replace('"', '').strip()
                translated_segments[i]["text"] = clean_text
            else:
                # Fallback immédiat si réponse vide
                logger.warning(f"⚠️ Gemini réponse vide pour : {text}")
                tr = GoogleTranslator(source='auto', target=fix_code_language(target))
                translated_segments[i]["text"] = tr.translate(text).strip()
                
        except Exception as e:
            # Fallback immédiat si erreur API
            logger.error(f"❌ Erreur Gemini : {e}")
            try:
                tr = GoogleTranslator(source='auto', target=fix_code_language(target))
                translated_segments[i]["text"] = tr.translate(text).strip()
            except:
                pass
        
        progress_bar.update(1)
        time.sleep(0.1) # Vitesse max

    progress_bar.close()
    return translated_segments


# ==============================================================================
# 🚀 GROQ (LLAMA 3) - RAPIDITÉ EXTRÊME
# ==============================================================================

def groq_translate(segments, target, source=None):
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        logger.error("❌ GROQ: Clé manquante !")
        return translate_iterative(segments, target, source)

    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
    except Exception as e:
        logger.error(f"❌ GROQ Client Error: {e}")
        return translate_iterative(segments, target, source)

    translated_segments = copy.deepcopy(segments)
    progress_bar = tqdm(total=len(segments), desc="Translating (Groq)")
    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES.get(target, target)).strip()

    for i, line in enumerate(translated_segments):
        text = line["text"].strip()
        if not text: continue

        try:
            # Prompt adapté aussi pour la fluidité
            chat = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"You are a dubbing translator. Translate to {lang_tg}. Use spoken, natural language. Output ONLY the translation."},
                    {"role": "user", "content": text}
                ],
                model="llama3-70b-8192",
                temperature=0.3,
            )
            res = chat.choices[0].message.content
            if res:
                translated_segments[i]["text"] = res.strip()
        except Exception as e:
            logger.error(f"❌ Groq Error segment {i}: {e}")
            try:
                tr = GoogleTranslator(source='auto', target=fix_code_language(target))
                translated_segments[i]["text"] = tr.translate(text).strip()
            except:
                pass

        progress_bar.update(1)

    progress_bar.close()
    return translated_segments


# --- FONCTIONS GPT (Simplifiées pour éviter les bugs) ---
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
    """Fonction principale"""
    target_clean = fix_code_language(target)
    source_clean = fix_code_language(source) if source else "auto"

    match translation_process:
        case "google_translator_batch":
            return translate_batch(segments, target_clean, chunk_size, source_clean)
        case "google_translator":
            return translate_iterative(segments, target_clean, source_clean)
        
        # --- NOS MODÈLES ---
        case "gemini_pro":
            return gemini_translate(segments, target, source)
        case "groq_llama3":
            return groq_translate(segments, target, source)
        # -------------------
        
        case model if "gpt" in model:
            # On redirige vers Google si quelqu'un choisit GPT par erreur
            return translate_iterative(segments, target_clean, source_clean)
            
        case "disable_translation":
            return segments
        case _:
            logger.warning(f"Unknown process {translation_process}, using Google.")
            return translate_iterative(segments, target_clean, source_clean)
