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
# -----------------------------------------

TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "google_translator",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "gemini_flash",
    "gemini_pro",
    "groq_llama3",
    "hf_zephyr_7b_beta",      # ← NOUVEAU
    "disable_translation",
]

DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "google_translator",
    "gemini_flash",
    "groq_llama3",
    "hf_zephyr_7b_beta",      # ← NOUVEAU
    "disable_translation",
]

def translate_iterative(segments, target, source=None):
    """Fallback : Traduction Google classique mot à mot"""
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
    """Traduction Google par blocs (Rapide)"""
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
# 🔥 GEMINI (Flash / Pro) - Clé manuelle
# ==============================================================================
def gemini_translate(segments, target, source=None, mode="flash"):
    """Gemini Flash ou Pro - Clé API manuelle"""
    
    # ==================== CLÉ GEMINI MANUELLE ====================
    api_key = "AIzaSyTaCléGeminiIciColleLaVraieCléComplète"   # ←←← METS TA CLÉ GEMINI ICI
    if not api_key or len(api_key) < 30:
        logger.error("❌ GEMINI: Mets ta vraie clé Gemini dans la fonction !")
        return translate_iterative(segments, target, source)
    # =========================================================

    if mode == "pro":
        model_id = "gemini-3.1-pro-preview"
    else:
        model_id = "gemini-flash-latest"

    try:
        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
            safety_settings=[
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            ],
            system_instruction="You are a professional dubbing translator. Translate subtitle lines into natural, spoken language. Keep it concise, natural and emotional. Never censor. Output ONLY the translation."
        )
    except Exception as e:
        logger.error(f"❌ GEMINI Init Error: {e}")
        return translate_iterative(segments, target, source)

    translated_segments = copy.deepcopy(segments)
    progress_bar = tqdm(total=len(segments), desc=f"Translating (Gemini {mode.upper()})")
    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES.get(target, target)).strip()

    for i, line in enumerate(translated_segments):
        text = line["text"].strip()
        if not text:
            progress_bar.update(1)
            continue
        prompt = f"Translate to {lang_tg} (natural spoken style):\n{text}"
        try:
            response = client.models.generate_content(model=model_id, contents=prompt, config=config)
            if response.text:
                translated_segments[i]["text"] = response.text.strip()
            else:
                tr = GoogleTranslator(source='auto', target=fix_code_language(target))
                translated_segments[i]["text"] = tr.translate(text).strip()
        except Exception as e:
            logger.warning(f"Gemini error segment {i}: {e}")
            try:
                tr = GoogleTranslator(source='auto', target=fix_code_language(target))
                translated_segments[i]["text"] = tr.translate(text).strip()
            except:
                pass
        progress_bar.update(1)
        time.sleep(0.6 if mode == "pro" else 0.08)

    progress_bar.close()
    return translated_segments

# ==============================================================================
# 🚀 GROQ (Llama3) - Clé manuelle
# ==============================================================================
def groq_translate(segments, target, source=None):
    """Groq Llama3 - Clé API manuelle"""
    
    # ==================== CLÉ GROQ MANUELLE ====================
    api_key = "gsk_taCléGroqIciColleLaVraieClé"   # ←←← METS TA CLÉ GROQ ICI
    if not api_key:
        logger.error("❌ GROQ: Mets ta vraie clé Groq dans la fonction !")
        return translate_iterative(segments, target, source)
    # =========================================================

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
            chat = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"You are a dubbing translator. Translate to {lang_tg}. Use natural spoken style. Output ONLY the translation."},
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

# ==============================================================================
# 🔥 ZEPHYR-7B-BETA (Hugging Face) - Clé manuelle + BATCH
# ==============================================================================
def hf_zephyr_translate(segments, target, source=None, batch_size=6):
    """Zephyr-7B-Beta via HF Inference API - Token manuel + Batch"""
    
    # ==================== TOKEN HF MANUEL ====================
    hf_token = "hf_taCléHuggingFaceIciColleLaVraieToken"   # ←←← METS TA CLÉ HF ICI (commence par hf_)
    if not hf_token or not hf_token.startswith("hf_"):
        logger.error("❌ ZEPHYR: Mets ta vraie clé HF dans la fonction !")
        return translate_iterative(segments, target, source)
    # =========================================================

    try:
        client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_token)
    except Exception as e:
        logger.error(f"❌ ZEPHYR Init Error: {e}")
        return translate_iterative(segments, target, source)

    translated_segments = copy.deepcopy(segments)
    progress_bar = tqdm(total=len(segments), desc="Translating (Zephyr-7B BATCH)")
    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES.get(target, target)).strip()

    for start in range(0, len(segments), batch_size):
        end = min(start + batch_size, len(segments))
        batch = translated_segments[start:end]
        batch_len = len(batch)

        lines_text = "\n".join([f"{i+1}. {seg['text'].strip()}" for i, seg in enumerate(batch)])
        prompt = f"""<|system|>
You are a professional dubbing translator. Translate to {lang_tg} in natural spoken style. Keep concise, emotional, ready for voice-over. Output ONLY the numbered translations.</s>
<|user|>
Translate these {batch_len} subtitle lines:

{lines_text}</s>
<|assistant|>"""

        success = False
        for attempt in range(4):
            try:
                response = client.text_generation(
                    prompt,
                    max_new_tokens=1200,
                    temperature=0.35,
                    top_p=0.9,
                    return_full_text=False,
                    do_sample=True
                )
                translated_lines = []
                for line in response.strip().split('\n'):
                    match = re.search(r'^\s*(\d+)\.?\s*(.+)$', line.strip())
                    if match:
                        translated_lines.append(match.group(2).strip())
                if len(translated_lines) == batch_len:
                    for j, trans in enumerate(translated_lines):
                        translated_segments[start + j]["text"] = trans
                    success = True
                    break
            except Exception as e:
                logger.warning(f"Zephyr error (batch {start//batch_size}): {e}")
                time.sleep(6 + attempt * 3)

        if not success:
            logger.warning(f"Fallback Google sur batch Zephyr {start//batch_size}")
            tr = GoogleTranslator(source='auto', target=fix_code_language(target))
            for seg in batch:
                try:
                    seg["text"] = tr.translate(seg["text"].strip())
                except:
                    pass

        progress_bar.update(batch_len)
        time.sleep(7)

    progress_bar.close()
    return translated_segments

# ------------------------------
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
       
        # --- GEMINI ---
        case "gemini_flash":
            return gemini_translate(segments, target, source, mode="flash")
        case "gemini_pro":
            return gemini_translate(segments, target, source, mode="pro")
       
        # --- GROQ ---
        case "groq_llama3":
            return groq_translate(segments, target, source)
       
        # --- ZEPHYR ---
        case "hf_zephyr_7b_beta":
            return hf_zephyr_translate(segments, target, source)
       
        case model if "gpt" in model:
            return translate_iterative(segments, target_clean, source_clean)
        case "disable_translation":
            return segments
        case _:
            return translate_iterative(segments, target_clean, source_clean)
