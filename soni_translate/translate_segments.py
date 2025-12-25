from tqdm import tqdm
from deep_translator import GoogleTranslator
from itertools import chain
import copy
import requests  # AJOUT : pour l'API Hugging Face
import os
from .language_configuration import fix_code_language, INVERTED_LANGUAGES
from .logging_setup import logger
import re
import json
import time

# AJOUT : nouvelle option pour Qwen HF gratuit
TRANSLATION_PROCESS_OPTIONS = [
    "google_translator_batch",
    "google_translator",
    "gpt-3.5-turbo-0125_batch",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview_batch",
    "gpt-4-turbo-preview",
    "qwen_hf (gratuit HF - contexte)",  # ← NOUVEAU !
    "disable_translation",
]

DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "google_translator",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "disable_translation",
]

# AJOUT : constante pour l'API Qwen (tu peux changer en 14B si tu veux plus rapide)
Qwen_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-32B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")  # Ajoute ton token dans .env ou ici directement

def translate_with_qwen_hf(segments, target, source=None):
    """
    Traduction intelligente avec Qwen2.5 via API gratuite Hugging Face
    + contexte des phrases précédentes pour un résultat naturel
    """
    if not HF_TOKEN:
        logger.error("Pas de token Hugging Face ! Ajoute HF_TOKEN dans .env ou dans le code")
        return translate_iterative(segments, target, source)  # fallback Google

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    translated_segments = []
    previous_context = ""  # On garde le contexte pour la cohérence

    progress_bar = tqdm(total=len(segments), desc="Traduction Qwen HF")

    for segment in segments:
        text = segment["text"].strip()
        if not text:
            translated_segments.append(segment)
            progress_bar.update(1)
            continue

        # Prompt optimisé pour contexte + naturel
        prompt = f"""Tu es un traducteur expert très naturel entre {source or 'la langue source'} et {target}.
Contexte des répliques précédentes (pour garder le ton, l'ironie, le registre) :
{previous_context if previous_context else "Aucun contexte précédent"}

Maintenant traduis UNIQUEMENT ce texte de manière fluide, idiomatique, avec le même ton et intention :

{text}

Réponse : uniquement la traduction, rien d'autre."""

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }

        try:
            response = requests.post(Qwen_API_URL, headers=headers, json=payload, timeout=45)
            response.raise_for_status()
            result = response.json()
            translated_text = result[0]["generated_text"].strip()

            # Nettoyage rapide (Qwen peut parfois ajouter du texte parasite)
            if "Réponse :" in translated_text:
                translated_text = translated_text.split("Réponse :", 1)[-1].strip()
            if translated_text.startswith("Voici la traduction :"):
                translated_text = translated_text.split(":", 1)[-1].strip()

            segment["text"] = translated_text
            previous_context += f"\n- Original: {text}\n  Traduit: {translated_text}"
            # Limite le contexte pour éviter de dépasser les tokens
            if len(previous_context) > 3000:
                previous_context = previous_context[-3000:]

        except Exception as e:
            logger.warning(f"Erreur Qwen HF sur segment : {e}. Fallback Google pour ce segment.")
            translator = GoogleTranslator(source=source or "auto", target=target)
            segment["text"] = translator.translate(text)

        translated_segments.append(segment)
        progress_bar.update(1)

    progress_bar.close()
    return translated_segments


def translate_iterative(segments, target, source=None):
    # Code original inchangé
    segments_ = copy.deepcopy(segments)
    if not source:
        logger.debug("No source language")
        source = "auto"
    translator = GoogleTranslator(source=source, target=target)
    for line in tqdm(range(len(segments_))):
        text = segments_[line]["text"]
        translated_line = translator.translate(text.strip())
        segments_[line]["text"] = translated_line
    return segments_


def verify_translate(segments, segments_copy, translated_lines, target, source):
    # Code original inchangé
    if len(segments) == len(translated_lines):
        for line in range(len(segments_copy)):
            logger.debug(f"{segments_copy[line]['text']} >> {translated_lines[line].strip()}")
            segments_copy[line]["text"] = translated_lines[line].replace("\t", "").replace("\n", "").strip()
        return segments_copy
    else:
        logger.error(f"Translation failed, switching to iterative. {len(segments), len(translated_lines)}")
        return translate_iterative(segments, target, source)


def translate_batch(segments, target, chunk_size=2000, source=None):
    # Code original inchangé
    # ... (tout le code Google batch reste identique)


def call_gpt_translate(...):
    # Code original GPT inchangé


def gpt_sequential(...):
    # Code original GPT inchangé


def gpt_batch(...):
    # Code original GPT inchangé


def translate_text(
    segments,
    target,
    translation_process="google_translator_batch",
    chunk_size=4500,
    source=None,
    token_batch_limit=1000,
):
    """Translates text segments using a specified process."""
    match translation_process:
        case "qwen_hf (gratuit HF - contexte)":  # AJOUT !
            logger.info("Traduction intelligente avec Qwen2.5 via Hugging Face API gratuite")
            return translate_with_qwen_hf(segments, fix_code_language(target), fix_code_language(source))

        case "google_translator_batch":
            return translate_batch(segments, fix_code_language(target), chunk_size, fix_code_language(source))

        case "google_translator":
            return translate_iterative(segments, fix_code_language(target), fix_code_language(source))

        case model if model in ["gpt-3.5-turbo-0125", "gpt-4-turbo-preview"]:
            return gpt_sequential(segments, model, target, source)

        case model if model in ["gpt-3.5-turbo-0125_batch", "gpt-4-turbo-preview_batch"]:
            return gpt_batch(segments, translation_process.replace("_batch", ""), target, token_batch_limit, source)

        case "disable_translation":
            return segments

        case _:
            raise ValueError("No valid translation process")
