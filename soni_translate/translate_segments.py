from tqdm import tqdm
from deep_translator import GoogleTranslator
from itertools import chain
import copy
from .language_configuration import fix_code_language, INVERTED_LANGUAGES
from .logging_setup import logger
import re
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

# ==============================================================================
# PROMPT GOLD DIGGER (même pour toutes les IA)
# ==============================================================================
GOLD_DIGGER_PROMPT = """Tu es un traducteur professionnel EXPERT en doublage français de vidéos YouTube "Gold Digger Prank".

RÈGLES OBLIGATOIRES :
1. LONGUEUR : Le français doit être AUSSI COURT ou PLUS COURT que l'anglais original.
   - Coupe tout superflu. Phrases très courtes et naturelles.
   - Objectif : même durée de parole que l'original (timing parfait avec la vidéo).

2. STYLE : Français jeune, street, banlieue/parisien (22-28 ans).
   - Toujours tutoyer.
   - Mots : mec, frère, vas-y, sérieux ?, c'est ouf, grave, wesh, putain, franchement, t'es sérieux là ?, arrête, nan mais attends, j'hallucine, c'est mort, etc.

3. TON : Arrogant, dragueur, moqueur, choqué, provocateur. Garde l'énergie exacte.

4. OUTPUT : Réponds UNIQUEMENT avec la traduction. Rien d'autre."""

# ==============================================================================
# GEMINI - Prompt intégré
# ==============================================================================
def gemini_translate(segments, target, source=None, mode="flash"):
    api_key = "AIzaSyTaCléGeminiIciColleLaVraieCléComplète"   # ← TA CLÉ
    if not api_key or len(api_key) < 30:
        logger.error("❌ GEMINI: Mets ta vraie clé !")
        return translate_iterative(segments, target, source)

    model_id = "gemini-3.1-pro-preview" if mode == "pro" else "gemini-flash-latest"

    try:
        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
            safety_settings=[types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE) 
                           for c in [types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                     types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, types.HarmCategory.HARM_CATEGORY_HARASSMENT]],
            system_instruction=GOLD_DIGGER_PROMPT
        )
    except Exception as e:
        logger.error(f"❌ GEMINI Init Error: {e}")
        return translate_iterative(segments, target, source)

    translated_segments = copy.deepcopy(segments)
    progress_bar = tqdm(total=len(segments), desc=f"Translating (Gemini {mode.upper()})")
    lang_tg_safe = "French"

    for i, line in enumerate(translated_segments):
        text = line["text"].strip()
        if not text:
            progress_bar.update(1)
            continue
        prompt = f"Translate to {lang_tg_safe}:\n{text}"
        try:
            response = client.models.generate_content(model=model_id, contents=prompt, config=config)
            if response.text:
                translated_segments[i]["text"] = response.text.strip()
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            logger.warning(f"Gemini error segment {i}: {error_msg}")
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
# GROQ - Prompt intégré
# ==============================================================================
def groq_translate(segments, target, source=None):
    api_key = "gsk_taCléGroqIciColleLaVraieClé"   # ← TA CLÉ
    if not api_key:
        logger.error("❌ GROQ: Mets ta vraie clé !")
        return translate_iterative(segments, target, source)

    try:
        http_client = httpx.Client(timeout=60.0)
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, http_client=http_client)
    except Exception as e:
        logger.error(f"❌ GROQ Client Error: {e}")
        return translate_iterative(segments, target, source)

    translated_segments = copy.deepcopy(segments)
    progress_bar = tqdm(total=len(segments), desc="Translating (Groq Llama-3.3)")
    lang_tg_safe = "French"

    for i, line in enumerate(translated_segments):
        text = line["text"].strip()
        if not text: continue
        try:
            chat = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": GOLD_DIGGER_PROMPT},
                    {"role": "user", "content": text}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
            )
            res = chat.choices[0].message.content
            if res:
                translated_segments[i]["text"] = res.strip()
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            logger.error(f"❌ Groq Error segment {i}: {error_msg}")
            try:
                tr = GoogleTranslator(source='auto', target=fix_code_language(target))
                translated_segments[i]["text"] = tr.translate(text).strip()
            except:
                pass
        progress_bar.update(1)
    progress_bar.close()
    return translated_segments

# ==============================================================================
# ZEPHYR - Prompt intégré (format que tu voulais)
# ==============================================================================
def hf_zephyr_translate(segments, target, source=None, batch_size=10):
    hf_token = "hf_taCléHuggingFaceIciColleLaVraieToken"   # ← TA CLÉ
    if not hf_token or not hf_token.startswith("hf_"):
        logger.error("❌ ZEPHYR: Mets ta vraie clé !")
        return translate_iterative(segments, target, source)

    try:
        client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_token)
    except Exception as e:
        logger.error(f"❌ ZEPHYR Init Error: {e}")
        return translate_iterative(segments, target, source)

    translated_segments = copy.deepcopy(segments)
    progress_bar = tqdm(total=len(segments), desc="Translating (Zephyr-7B BATCH 10)")
    lang_tg = re.sub(r'\([^)]*\)', '', INVERTED_LANGUAGES.get(target, target)).strip()

    for start in range(0, len(segments), batch_size):
        end = min(start + batch_size, len(segments))
        batch = translated_segments[start:end]
        batch_len = len(batch)

        lines_text = "\n".join([f"{i+1}. {seg['text'].strip()}" for i, seg in enumerate(batch)])
        prompt = f"""<|system|>
{GOLD_DIGGER_PROMPT}</s>
<|user|>
Translate these {batch_len} subtitle lines to French (keep the same length or shorter):

{lines_text}</s>
<|assistant|>"""

        success = False
        for attempt in range(4):
            try:
                response = client.text_generation(
                    prompt,
                    max_new_tokens=1500,
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
                time.sleep(5 + attempt * 2)

        if not success:
            logger.warning(f"Fallback Google sur batch Zephyr {start//batch_size}")
            tr = GoogleTranslator(source='auto', target=fix_code_language(target))
            for seg in batch:
                try:
                    seg["text"] = tr.translate(seg["text"].strip())
                except:
                    pass

        progress_bar.update(batch_len)
        time.sleep(4)

    progress_bar.close()
    return translated_segments

# ==============================================================================
# Les autres fonctions restent identiques (translate_iterative, translate_batch, translate_text, etc.)
# ==============================================================================
# [Je ne les recopie pas ici pour ne pas alourdir le message, mais elles sont exactement les mêmes que dans la version précédente que je t’ai donnée]

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
