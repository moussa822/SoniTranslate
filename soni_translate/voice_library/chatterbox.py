import os
import torch
import soundfile as sf
from .base import BaseTTSProvider

# Cache global de session pour garder Chatterbox chargé en mémoire vive
CHATTERBOX_MODEL_CACHE = None

class ChatterBoxProvider(BaseTTSProvider):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Dossier de stockage des modèles Chatterbox sur SoniTranslate
        self.local_dir = "weights/ChatterBox"

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
        """Charge le modèle Chatterbox en VRAM une seule fois."""
        global CHATTERBOX_MODEL_CACHE
        if CHATTERBOX_MODEL_CACHE is None:
            self.logger.info("Initializing Chatterbox Multilingual V3 model...")
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            
            os.makedirs(self.local_dir, exist_ok=True)
            
            # Téléchargement automatique depuis Hugging Face si les fichiers essentiels manquent
            from huggingface_hub import snapshot_download
            allow_list = [
                "ve.pt", 
                "t3_mtl23ls_v3.safetensors", 
                "s3gen.pt", 
                "grapheme_mtl_merged_expanded_v1.json", 
                "conds.pt", 
                "Cangjie5_TC.json"
            ]
            
            if not any(f in os.listdir(self.local_dir) for f in ["ve.pt", "s3gen.pt"]):
                self.logger.info("Downloading Chatterbox model files from Hugging Face...")
                snapshot_download(
                    repo_id="resembleAI/chatterbox",
                    local_dir=self.local_dir,
                    allow_patterns=allow_list,
                    local_dir_use_symlinks=False
                )
            
            # Initialisation
            CHATTERBOX_MODEL_CACHE = ChatterboxMultilingualTTS.from_local(
                self.local_dir,
                device=self.device,
                t3_model="v3" # Utilisation forcée de la V3
            )
        return CHATTERBOX_MODEL_CACHE

    def generate(self, text, voice, target_lang, output_file, **kwargs):
        model = self.load_model()
        
        # Récupération de l'audio de référence
        voice_name = voice.split("/")[-1]
        ref_audio = f"voice_library/{voice_name}.wav"
        
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio not found in voice library: {ref_audio}")
            
        lang_id = self.get_language_id(target_lang)
        
        # Récupération des paramètres d'exagération et de fidélité (0.5 par défaut)
        exaggeration = kwargs.get("exaggeration", 0.5)
        cfg_weight = kwargs.get("cfg_weight", 0.5)
        
        # Génération
        wav_tensor = model.generate(
            text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            language_id=lang_id,
            audio_prompt_path=ref_audio
        )
        
        # Conversion du tenseur PyTorch en NumPy
        wav_tensor = wav_tensor.detach().cpu()
        if wav_tensor.ndim == 2:
            wav_np = wav_tensor.transpose(0, 1).numpy()
        else:
            wav_np = wav_tensor.numpy()
            
        # Enregistrement natif à 24 000 Hz
        sf.write(output_file, wav_np, model.sr)
