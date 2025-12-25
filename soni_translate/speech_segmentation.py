from whisperx.alignment import (
    DEFAULT_ALIGN_MODELS_TORCH as DAMT,
    DEFAULT_ALIGN_MODELS_HF as DAMHF,
)
from whisperx.utils import TO_LANGUAGE_CODE
import whisperx
import torch
import gc
import os
import soundfile as sf
from IPython.utils import capture  # noqa
from .language_configuration import EXTRA_ALIGN, INVERTED_LANGUAGES
from .logging_setup import logger
from .postprocessor import sanitize_file_name
from .utils import remove_directory_contents, run_command

# Imports pour NeMo (ajoutés ici)
import json
import yaml
import shutil
import glob
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ClusteringDiarizer


# Options existantes (inchangées)
ASR_MODEL_OPTIONS = [
    "tiny", "base", "small", "medium", "large",
    "large-v1", "large-v2", "large-v3",
    "distil-large-v2", "Systran/faster-distil-whisper-large-v3",
    "tiny.en", "base.en", "small.en", "medium.en",
    "distil-small.en", "distil-medium.en",
    "OpenAI_API_Whisper",
]

COMPUTE_TYPE_GPU = [
    "default", "auto", "int8", "int8_float32", "int8_float16",
    "int8_bfloat16", "float16", "bfloat16", "float32"
]

COMPUTE_TYPE_CPU = [
    "default", "auto", "int8", "int8_float32", "int16", "float32"
]

WHISPER_MODELS_PATH = './WHISPER_MODELS'


def openai_api_whisper(input_audio_file, source_lang=None, chunk_duration=1800):
    info = sf.info(input_audio_file)
    duration = info.duration

    output_directory = "./whisper_api_audio_parts"
    os.makedirs(output_directory, exist_ok=True)
    remove_directory_contents(output_directory)

    if duration > chunk_duration:
        cm = f'ffmpeg -i "{input_audio_file}" -f segment -segment_time {chunk_duration} -c:a libvorbis "{output_directory}/output%03d.ogg"'
        run_command(cm)
        chunk_files = sorted(
            [f"{output_directory}/{f}" for f in os.listdir(output_directory) if f.endswith('.ogg')]
        )
    else:
        one_file = f"{output_directory}/output000.ogg"
        cm = f'ffmpeg -i "{input_audio_file}" -c:a libvorbis {one_file}'
        run_command(cm)
        chunk_files = [one_file]

    segments = []
    language = source_lang if source_lang else None

    for i, chunk in enumerate(chunk_files):
        from openai import OpenAI
        client = OpenAI()
        audio_file = open(chunk, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

        try:
            transcript_dict = transcription.model_dump()
        except:  # noqa
            transcript_dict = transcription.to_dict()

        if language is None:
            logger.info(f'Language detected: {transcript_dict["language"]}')
            language = TO_LANGUAGE_CODE[transcript_dict["language"]]

        chunk_time = chunk_duration * i
        for seg in transcript_dict["segments"]:
            if "start" in seg:
                segments.append({
                    "text": seg["text"],
                    "start": seg["start"] + chunk_time,
                    "end": seg["end"] + chunk_time,
                })

    audio = whisperx.load_audio(input_audio_file)
    result = {"segments": segments, "language": language}
    return audio, result


def find_whisper_models():
    path = WHISPER_MODELS_PATH
    folders = []
    if os.path.exists(path):
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path) and 'model.bin' in os.listdir(folder_path):
                folders.append(folder)
    return folders


def transcribe_speech(
    audio_wav,
    asr_model,
    compute_type,
    batch_size,
    SOURCE_LANGUAGE,
    literalize_numbers=True,
    segment_duration_limit=15,
):
    if asr_model == "OpenAI_API_Whisper":
        if literalize_numbers:
            logger.info("OpenAI's API Whisper does not support the literalization of numbers.")
        return openai_api_whisper(audio_wav, SOURCE_LANGUAGE)

    prompt = "以下是普通话的句子。" if SOURCE_LANGUAGE == "zh" else None
    SOURCE_LANGUAGE = SOURCE_LANGUAGE if SOURCE_LANGUAGE != "zh-TW" else "zh"

    asr_options = {
        "initial_prompt": prompt,
        "suppress_numerals": literalize_numbers
    }

    if asr_model not in ASR_MODEL_OPTIONS:
        base_dir = WHISPER_MODELS_PATH
        os.makedirs(base_dir, exist_ok=True)
        model_dir = os.path.join(base_dir, sanitize_file_name(asr_model))

        if not os.path.exists(model_dir):
            from ctranslate2.converters import TransformersConverter
            quantization = "float32"
            try:
                converter = TransformersConverter(
                    asr_model,
                    low_cpu_mem_usage=True,
                    copy_files=["tokenizer_config.json", "preprocessor_config.json"]
                )
                converter.convert(model_dir, quantization=quantization, force=False)
            except Exception as error:
                if "File tokenizer_config.json does not exist" in str(error):
                    converter._copy_files = ["tokenizer.json", "preprocessor_config.json"]
                    converter.convert(model_dir, quantization=quantization, force=True)
                else:
                    raise error

        asr_model = model_dir
        logger.info(f"ASR Model: {model_dir}")

    model = whisperx.load_model(
        asr_model,
        os.environ.get("SONITR_DEVICE"),
        compute_type=compute_type,
        language=SOURCE_LANGUAGE,
        asr_options=asr_options,
    )

    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        chunk_size=segment_duration_limit,
        print_progress=True,
    )

    if result["language"] == "zh" and not prompt:
        result["language"] = "zh-TW"
        logger.info("Chinese - Traditional (zh-TW)")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return audio, result


def align_speech(audio, result):
    DAMHF.update(DAMT)  # lang align

    if result["language"] not in DAMHF and result["language"] not in EXTRA_ALIGN:
        logger.warning("Automatic detection: Source language not compatible with align")
        raise ValueError(
            f"Detected language {result['language']} incompatible, "
            "you can select the source language to avoid this error."
        )

    if result["language"] in EXTRA_ALIGN and EXTRA_ALIGN[result["language"]] == "":
        lang_name = INVERTED_LANGUAGES.get(result["language"], result["language"])
        logger.warning(f"No compatible wav2vec2 model found for '{lang_name}', skipping alignment.")
        return result

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=os.environ.get("SONITR_DEVICE"),
        model_name=None if result["language"] in DAMHF else EXTRA_ALIGN[result["language"]],
    )

    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        os.environ.get("SONITR_DEVICE"),
        return_char_alignments=True,
        print_progress=False,
    )

    del model_a
    gc.collect()
    torch.cuda.empty_cache()
    return result


diarization_models = {
    "pyannote_3.1": "pyannote/speaker-diarization-3.1",
    "pyannote_2.1": "pyannote/speaker-diarization@2.1",
    "disable": "",
}


def reencode_speakers(result):
    if result["segments"] and result["segments"][0]["speaker"] == "SPEAKER_00":
        return result

    speaker_mapping = {}
    counter = 0
    logger.debug("Reencode speakers")

    for segment in result["segments"]:
        old_speaker = segment["speaker"]
        if old_speaker not in speaker_mapping:
            speaker_mapping[old_speaker] = f"SPEAKER_{counter:02d}"
            counter += 1
        segment["speaker"] = speaker_mapping[old_speaker]

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Fonction NeMo (nouvelle, bien placée)
# ──────────────────────────────────────────────────────────────────────────────
def diarize_with_nemo(audio_path: str, num_speakers: int = None, device: str = 'cuda') -> dict:
    """
    Diarisation avec NeMo ClusteringDiarizer - format compatible WhisperX.
    """
    logger.info("Démarrage de la diarisation NeMo...")
    out_dir = "temp_nemo_diar"
    os.makedirs(out_dir, exist_ok=True)

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": num_speakers,
            "rttm_filepath": None,
            "uem_filepath": None
        }, f)

    cfg = {
        "diarizer": {
            "manifest_filepath": manifest_path,
            "out_dir": out_dir,
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {"model_name": "vad_multilingual_marblenet"},
            "speaker_embeddings": {"model_name": "titanet_large"},
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": num_speakers is not None,
                    "max_num_speakers": num_speakers or 8,
                }
            }
        }
    }

    cfg_path = os.path.join(out_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    model = ClusteringDiarizer(cfg=cfg).to(device)
    model.diarize()

    rttm_files = glob.glob(f"{out_dir}/pred_rttms/*.rttm")
    if not rttm_files:
        raise ValueError("Aucun fichier RTTM généré par NeMo.")

    rttm_file = rttm_files[0]
    segments = []
    speaker_counter = 0
    speaker_map = {}

    with open(rttm_file, "r") as f:
        for line in f:
            if line.startswith("SPEAKER"):
                parts = line.split()
                start = float(parts[3])
                dur = float(parts[4])
                spk_label = parts[7]

                if spk_label not in speaker_map:
                    speaker_map[spk_label] = f"SPEAKER_{speaker_counter:02d}"
                    speaker_counter += 1

                segments.append({
                    "start": start,
                    "end": start + dur,
                    "speaker": speaker_map[spk_label]
                })

    shutil.rmtree(out_dir, ignore_errors=True)
    logger.info("Diarisation NeMo terminée.")
    return {"segments": segments}


# ──────────────────────────────────────────────────────────────────────────────
# Fonction diarisation principale - MODIFIÉE pour choisir pyannote ou NeMo
# ──────────────────────────────────────────────────────────────────────────────
def diarize_speech(
    audio_wav,
    result,
    min_speakers,
    max_speakers,
    YOUR_HF_TOKEN,
    model_name="pyannote/speaker-diarization@2.1",
):
    if max(min_speakers, max_speakers) <= 1 or not model_name:
        logger.info("Un seul speaker détecté ou diarisation désactivée → assignation SPEAKER_00")
        result_diarize = result
        result_diarize["segments"] = [
            {**item, "speaker": "SPEAKER_00"} for item in result_diarize["segments"]
        ]
        return reencode_speakers(result_diarize)

    try:
        # ← Ici tu peux changer pour tester NeMo
        use_nemo = True  # Change à False pour revenir à pyannote

        if use_nemo:
            logger.info("Utilisation du modèle NeMo pour la diarisation")
            diarize_segments = diarize_with_nemo(
                audio_wav,
                num_speakers=max_speakers,
                device=os.environ.get("SONITR_DEVICE")
            )
        else:
            logger.info(f"Utilisation de pyannote ({model_name}) pour la diarisation")
            diarize_model = whisperx.DiarizationPipeline(
                model_name=model_name,
                use_auth_token=YOUR_HF_TOKEN,
                device=os.environ.get("SONITR_DEVICE"),
            )
            diarize_segments = diarize_model(
                audio_wav, min_speakers=min_speakers, max_speakers=max_speakers
            )
            del diarize_model

        result_diarize = whisperx.assign_word_speakers(diarize_segments, result)

        for segment in result_diarize["segments"]:
            if "speaker" not in segment:
                segment["speaker"] = "SPEAKER_00"
                logger.warning(
                    f"Aucun speaker détecté à {segment['start']}. "
                    f"Premier TTS utilisé pour : {segment['text']}"
                )

        gc.collect()
        torch.cuda.empty_cache()

    except Exception as error:
        error_str = str(error)
        gc.collect()
        torch.cuda.empty_cache()

        if "'NoneType' object has no attribute 'to'" in error_str:
            if model_name == diarization_models["pyannote_2.1"]:
                raise ValueError(
                    "Acceptez la licence pyannote 2.1 sur Hugging Face :\n"
                    "https://huggingface.co/pyannote/speaker-diarization\n"
                    "https://huggingface.co/pyannote/segmentation"
                )
            elif model_name == diarization_models["pyannote_3.1"]:
                raise ValueError(
                    "Nouvelle licence pyannote 3.1 :\n"
                    "https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                    "https://huggingface.co/pyannote/segmentation-3.0"
                )
        else:
            raise error

    return reencode_speakers(result_diarize)
