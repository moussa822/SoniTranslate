import gradio as gr
from soni_translate.logging_setup import (
    logger,
    set_logging_level,
    configure_logging_libs,
); configure_logging_libs() # noqa
import whisperx
import torch
import os
from soni_translate.audio_segments import create_translated_audio
from soni_translate.text_to_speech import (
    audio_segmentation_to_voice,
    edge_tts_voices_list,
    coqui_xtts_voices_list,
    piper_tts_voices_list,
    create_wav_file_vc,
    accelerate_segments,
)
from soni_translate.translate_segments import (
    translate_text,
    TRANSLATION_PROCESS_OPTIONS,
    DOCS_TRANSLATION_PROCESS_OPTIONS
)
from soni_translate.preprocessor import (
    audio_video_preprocessor,
    audio_preprocessor,
)
from soni_translate.postprocessor import (
    OUTPUT_TYPE_OPTIONS,
    DOCS_OUTPUT_TYPE_OPTIONS,
    sound_separate,
    get_no_ext_filename,
    media_out,
    get_subtitle_speaker,
)
from soni_translate.language_configuration import (
    LANGUAGES,
    UNIDIRECTIONAL_L_LIST,
    LANGUAGES_LIST,
    BARK_VOICES_LIST,
    VITS_VOICES_LIST,
    OPENAI_TTS_MODELS,
)
from soni_translate.utils import (
    remove_files,
    download_list,
    upload_model_list,
    download_manager,
    run_command,
    is_audio_file,
    is_subtitle_file,
    copy_files,
    get_valid_files,
    get_link_list,
    remove_directory_contents,
)
from soni_translate.mdx_net import (
    UVR_MODELS,
    MDX_DOWNLOAD_LINK,
    mdxnet_models_dir,
)
from soni_translate.speech_segmentation import (
    ASR_MODEL_OPTIONS,
    COMPUTE_TYPE_GPU,
    COMPUTE_TYPE_CPU,
    find_whisper_models,
    transcribe_speech,
    align_speech,
    diarize_speech,
    diarization_models,
)
from soni_translate.text_multiformat_processor import (
    BORDER_COLORS,
    srt_file_to_segments,
    document_preprocessor,
    determine_chunk_size,
    plain_text_to_segments,
    segments_to_plain_text,
    process_subtitles,
    linguistic_level_segments,
    break_aling_segments,
    doc_to_txtximg_pages,
    page_data_to_segments,
    update_page_data,
    fix_timestamps_docs,
    create_video_from_images,
    merge_video_and_audio,
)
from soni_translate.languages_gui import language_data, news
import copy
import logging
import json
from pydub import AudioSegment
from voice_main import ClassVoices
import argparse
import time
import hashlib
import sys

directories = [
    "downloads", "logs", "weights", "clean_song_output", "_XTTS_",
    f"audio2{os.sep}audio", "audio", "outputs",
]
[os.makedirs(directory) for directory in directories if not os.path.exists(directory)]

class TTS_Info:
    def __init__(self, piper_enabled, xtts_enabled):
        self.list_edge = edge_tts_voices_list()
        self.list_bark = list(BARK_VOICES_LIST.keys())
        self.list_vits = list(VITS_VOICES_LIST.keys())
        self.list_openai_tts = OPENAI_TTS_MODELS
        self.piper_enabled = piper_enabled
        self.list_vits_onnx = piper_tts_voices_list() if self.piper_enabled else []
        self.xtts_enabled = xtts_enabled

    def tts_list(self):
        self.list_coqui_xtts = coqui_xtts_voices_list() if self.xtts_enabled else []
        return self.list_coqui_xtts + sorted(
            self.list_edge + self.list_bark + self.list_vits +
            self.list_openai_tts + self.list_vits_onnx
        )

def prog_disp(msg, percent, is_gui, progress=None):
    logger.info(msg)
    if is_gui:
        progress(percent, desc=msg)

def warn_disp(wrn_lang, is_gui):
    logger.warning(wrn_lang)
    if is_gui:
        gr.Warning(wrn_lang)

class SoniTrCache:
    # ... (le code de la classe SoniTrCache reste 100% identique, je ne le répète pas ici pour brevité)

class SoniTranslate(SoniTrCache):
    # ... (le __init__ et get_tts_voice_list restent identiques)

    def multilingual_media_conversion(
        self,
        media_file=None,
        link_media="",
        directory_input="",
        YOUR_HF_TOKEN="",
        preview=False,
        transcriber_model="large-v3",
        batch_size=4,
        compute_type="auto",
        origin_language="Automatic detection",
        target_language="English (en)",
        min_speakers=1,
        max_speakers=1,
        tts_voice00="en-US-EmmaMultilingualNeural-Female",
        # ... (tous les autres paramètres TTS et autres restent identiques)
        diarization_model="pyannote_2.1",
        translate_process="google_translator_batch",
        # ... (tous les autres paramètres jusqu'à is_gui=False, progress=gr.Progress()),
        diar_model_choice="pyannote (défaut - stable)",  # AJOUT NEVO : paramètre reçu du dropdown
    ):
        # AJOUT NEVO : décision simple si on utilise NeMo ou pyannote
        use_nemo = "nemo" in str(diar_model_choice).lower()

        # ... (tout le code jusqu'à l'appel à diarize_speech reste identique)

        if not self.task_in_cache("diarize", [
            min_speakers,
            max_speakers,
            YOUR_HF_TOKEN[:len(YOUR_HF_TOKEN)//2],
            diarization_model
        ], {"result": self.result}):
            prog_disp("Diarizing...", 0.60, is_gui, progress=progress)
            diarize_model_select = diarization_models[diarization_model]
            self.result_diarize = diarize_speech(
                base_audio_wav if not self.vocals else self.vocals,
                self.result,
                min_speakers,
                max_speakers,
                YOUR_HF_TOKEN,
                diarize_model_select,
                use_nemo=use_nemo  # AJOUT NEVO : on passe le flag ici !
            )
            logger.debug("Diarize complete")

        # ... (le reste de la fonction reste 100% identique jusqu'à la fin)

    # ... (les autres méthodes comme hook_beta_processor et multilingual_docs_conversion restent identiques)

# AJOUT NEVO : choix pour le dropdown
DIARIZATION_CHOICES = [
    "pyannote (défaut - stable)",
    "nemo (plus puissant sur overlaps & longs audios)"
]

def create_gui(theme, logs_in_gui=False):
    with gr.Blocks(theme=theme) as app:
        # ... (markdown et tabs existants)

        with gr.Tab(lg_conf["tab_translate"]):
            with gr.Row():
                with gr.Column():
                    # ... (tous les inputs vidéo, langues, speakers, TTS, etc. restent identiques)

                    with gr.Accordion(lg_conf["extra_setting"], open=False):
                        # ... (les sliders et options existantes)

                        # AJOUT NEVO : dropdown pour choisir pyannote ou NeMo
                        diar_model_choice = gr.Dropdown(
                            choices=DIARIZATION_CHOICES,
                            value=DIARIZATION_CHOICES[0],  # pyannote par défaut
                            label="Modèle de diarisation (locuteurs)",
                            info="pyannote = rapide et stable\nNeMo = meilleur pour voix qui se chevauchent",
                            interactive=True,
                        )

                        # ... (le reste de l'accordéon : text_segmentation, etc.)

                    # ... (edit_sub_check, boutons, etc.)

        # ... (les autres tabs : docs, custom voice, help restent identiques)

        # Mise à jour des .click() pour inclure le nouveau paramètre
        video_button.click(
            SoniTr.batch_multilingual_media_conversion,
            inputs=[
                # ... tous les inputs précédents (video_input, HFKEY, etc.)
                enable_custom_voice,
                workers_custom_voice,
                is_gui_dummy_check,
                diar_model_choice,           # AJOUT NEVO : dernier input !
            ],
            outputs=video_output,
            trigger_mode="multiple",
        ).then(
            play_sound_alert, [play_sound_gui], [sound_alert_notification]
        )

        subs_button.click(
            SoniTr.batch_multilingual_media_conversion,
            inputs=[
                # ... même liste que ci-dessus, plus le nouveau
                diar_model_choice,
            ],
            outputs=subs_edit_space,
        ).then(
            play_sound_alert, [play_sound_gui], [sound_alert_notification]
        )

        # ... (docs_button et le reste restent identiques)

    return app

# ... (le reste du fichier : get_language_config, create_parser, if __name__ == "__main__" reste identique)
