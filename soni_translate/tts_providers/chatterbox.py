import os
import torch
import torchaudio
from .base import BaseTTSProvider

# Cache de session global pour conserver le modèle chargé en VRAM
CHATTERBOX_MODEL_CACHE = None

class ChatterBoxProvider(BaseTTSProvider):
    def __init__(self):
        super().__init__()
        
        # Détection automatique du meilleur processeur disponible (CUDA, MPS ou CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def get_language_id(self, target_lang):
        """Mappe la langue de SoniTranslate vers le code ISO à 2 lettres de Chatterbox"""
        lang = target_lang.lower()
        if "french" in lang or "fr" in lang: return "fr"
        elif "spanish" in lang or "es" in lang: return "es"
        elif "japanese" in lang or "ja" in lang: return "ja"
        elif "chinese" in lang or "zh" in lang: return "zh"
        elif "italian" in lang or "it" in lang: return "it"
        elif "portuguese" in lang or "pt" in lang: return "pt"
        elif "hindi" in lang or "hi" in lang: return "hi"
        elif "german" in lang or "de" in lang: return "de"
        elif "danish" in lang or "da" in lang: return "da"
        elif "dutch" in lang or "nl" in lang: return "nl"
        elif "finnish" in lang or "fi" in lang: return "fi"
        elif "greek" in lang or "el" in lang: return "el"
        elif "hebrew" in lang or "he" in lang: return "he"
        elif "korean" in lang or "ko" in lang: return "ko"
        elif "malay" in lang or "ms" in lang: return "ms"
        elif "norwegian" in lang or "no" in lang: return "no"
        elif "polish" in lang or "pl" in lang: return "pl"
        elif "russian" in lang or "ru" in lang: return "ru"
        elif "swedish" in lang or "sv" in lang: return "sv"
        elif "swahili" in lang or "sw" in lang: return "sw"
        elif "turkish" in lang or "tr" in lang: return "tr"
        else: return "en"

    def load_model(self):
        """Charge le modèle Chatterbox en mémoire vive une seule fois."""
        global CHATTERBOX_MODEL_CACHE
        if CHATTERBOX_MODEL_CACHE is None:
            self.logger.info(f"Initializing Chatterbox Multilingual V3 model on '{self.device}'...")
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            
            # Utilisation de la méthode native et robuste de téléchargement automatique
            CHATTERBOX_MODEL_CACHE = ChatterboxMultilingualTTS.from_pretrained(
                device=self.device,
                t3_model="v3" # Chargement forcé de la dernière V3
            )
        return CHATTERBOX_MODEL_CACHE

    def generate(self, text, voice, target_lang, output_file, **kwargs):
        model = self.load_model()
        
        # Le nom de la voix passée est sous la forme : "Custom/ref_homme_fr"
        voice_name = voice.split("/")[-1]
        
        # Chemin absolu d'exécution sur Colab pour éviter les décalages de dossier
        project_root = "/content/SoniTranslate"
        ref_audio = os.path.join(project_root, "voice_library", f"{voice_name}.wav")
        
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio not found in voice library: {ref_audio}")
            
        lang_id = self.get_language_id(target_lang)
        
        # Paramètres d'expressivité de base (recommandés par Resemble AI)
        exaggeration = kwargs.get("exaggeration", 0.5)
        cfg_weight = kwargs.get("cfg_weight", 0.5)
        
        # --- SÉCURITÉ DE TRANSFERT D'ACCENT (L'astuce de pro !) ---
        # Si on fait du cross-lingual (voix fr parlant anglais), on coupe le cfg_weight
        # pour annuler le transfert d'accent français et garder l'anglais natif !
        is_cross_lingual = "fr" in voice_name.lower() and "en" in lang_id
        if is_cross_lingual:
            self.logger.info("Cross-lingual detected: setting cfg_weight to 0.0 to prevent French accent transfer.")
            cfg_weight = 0.0
            
        # Génération directe via Chatterbox (renvoie un tenseur PyTorch)
        wav_tensor = model.generate(
            text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            language_id=lang_id,
            audio_prompt_path=ref_audio
        )
        
        # Enregistrement natif ultra-stable à 24000Hz avec soundfile (évite torchcodec)
        wav_tensor = wav_tensor.detach().cpu()
        if wav_tensor.ndim == 2:
            wav_np = wav_tensor.transpose(0, 1).numpy()
        else:
            wav_np = wav_tensor.numpy()
        sf.write(output_file, wav_np, model.sr)

