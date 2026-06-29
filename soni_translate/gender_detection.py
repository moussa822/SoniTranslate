import os
import torch
import torchaudio
from transformers import pipeline
from soni_translate.logging_setup import logger

class VoiceGenderDetector:
    # On passe sur un modèle multilingue d'une précision chirurgicale (F1 score: 0.9993)
    def __init__(self, model_id="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"):
        self.model_id = model_id
        self.classifier = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        if self.classifier is None:
            logger.info(f"Loading multilingual voice gender classification model: {self.model_id}")
            self.classifier = pipeline(
                "audio-classification",
                model=self.model_id,
                device=0 if self.device == "cuda" else -1
            )

    def detect_speaker_genders(self, audio_path, diarization_result):
        self.load_model()
        
        speaker_segments = {}
        for segment in diarization_result.get("segments", []):
            speaker = segment.get("speaker")
            if speaker:
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append((segment["start"], segment["end"]))
        
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            logger.error(f"Error loading audio for gender detection: {e}")
            return {}

        speaker_genders = {}

        for speaker, segments in speaker_segments.items():
            speaker_chunks = []
            accumulated_duration = 0.0

            for start, end in segments:
                duration = end - start
                if duration < 0.5:
                    continue
                
                start_frame = int(start * sample_rate)
                end_frame = int(end * sample_rate)
                
                chunk = waveform[:, start_frame:end_frame]
                speaker_chunks.append(chunk)
                accumulated_duration += duration
                
                if accumulated_duration >= 8.0:  # 8 secondes suffisent largement
                    break

            if not speaker_chunks:
                logger.warning(f"No valid audio segments found for speaker {speaker}")
                speaker_genders[speaker] = "unknown"
                continue

            combined_waveform = torch.cat(speaker_chunks, dim=1)
            
            # Étape essentielle : Ré-échantillonnage à 16000 Hz pour la compatibilité absolue avec Wav2Vec2
            if sample_rate != 16000:
                transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                combined_waveform = transform(combined_waveform)

            # Conversion en Mono
            mono_waveform = combined_waveform.mean(dim=0, keepdim=True)
            
            # Sauvegarde d'un fichier WAV temporaire propre pour l'analyse
            temp_wav_path = f"temp_gender_{speaker}.wav"
            torchaudio.save(temp_wav_path, mono_waveform, 16000)

            try:
                # Analyse directe du fichier propre
                predictions = self.classifier(temp_wav_path)
                gender_label = predictions[0]["label"].lower() # 'male' ou 'female'
                speaker_genders[speaker] = gender_label
                logger.info(f"Gender detected for {speaker}: {gender_label} (Confidence: {predictions[0]['score']:.2f})")
            except Exception as e:
                logger.error(f"Failed to detect gender for {speaker}: {e}")
                speaker_genders[speaker] = "unknown"
            finally:
                # Nettoyage immédiat du fichier temporaire
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)

        return speaker_genders


def auto_assign_voices(speaker_genders, target_language="french"):
    """
    Assigne automatiquement des voix EdgeTTS de référence
    en fonction du genre détecté et de la langue cible.
    """
    lang = target_language.lower()
    
    # Voix par défaut pour le Français (avec le suffixe SoniTranslate requis !)
    french_male = "fr-FR-HenriNeural-Male"
    french_female = "fr-FR-DeniseNeural-Female"
    
    # Voix par défaut pour l'Anglais (avec le suffixe SoniTranslate requis !)
    english_male = "en-US-AndrewMultilingualNeural-Male"
    english_female = "en-US-EmmaMultilingualNeural-Female"

    assigned_voices = {}
    for speaker, gender in speaker_genders.items():
        if "french" in lang or "fr" in lang:
            assigned_voices[speaker] = french_female if gender == "female" else french_male
        else:
            assigned_voices[speaker] = english_female if gender == "female" else english_male
            
    return assigned_voices
