import os
import torch
import soundfile as sf
from .base import BaseTTSProvider

# Cache global de session pour garder le modèle chargé en mémoire vive
OMNIVOICE_MODEL_CACHE = None

class OmniVoiceProvider(BaseTTSProvider):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Dossier de stockage des poids du modèle dans SoniTranslate
        self.local_dir = "weights/OmniVoice"

    def load_model(self):
        """Charge le modèle OmniVoice en mémoire vive une seule fois."""
        global OMNIVOICE_MODEL_CACHE
        if OMNIVOICE_MODEL_CACHE is None:
            self.logger.info("Initializing OmniVoice model from 'k2-fsa/OmniVoice'...")
            from omnivoice import OmniVoice
            
            # Création du dossier de stockage si inexistant
            os.makedirs(self.local_dir, exist_ok=True)
            
            # Téléchargement automatique depuis Hugging Face si le dossier est vide
            from huggingface_hub import snapshot_download
            if not os.listdir(self.local_dir):
                self.logger.info("Downloading OmniVoice model weights from Hugging Face...")
                snapshot_download(
                    repo_id="k2-fsa/OmniVoice",
                    local_dir=self.local_dir,
                    local_dir_use_symlinks=False
                )
            
            # Initialisation du modèle
            OMNIVOICE_MODEL_CACHE = OmniVoice.from_pretrained(
                self.local_dir,
                device_map=self.device,
                dtype=torch.float32 if self.device == 'cpu' else torch.float16
            )
        return OMNIVOICE_MODEL_CACHE

    def generate(self, text, voice, target_lang, output_file, **kwargs):
        model = self.load_model()
        
        # Le nom de la voix passée est sous la forme : "Custom/Moussa_FR"
        voice_name = voice.split("/")[-1]
        
        # Récupération de l'audio de référence et du texte associé dans ta Voice Library
        ref_audio = f"voice_library/{voice_name}.wav"
        ref_txt_path = f"voice_library/{voice_name}.txt"
        
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio not found in voice library: {ref_audio}")
            
        if os.path.exists(ref_txt_path):
            with open(ref_txt_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
        else:
            ref_text = "Hello there." # Fallback par défaut si pas de fichier texte
            
        # Génération de l'audio
        # OmniVoice renvoie une liste de waveforms, on prend la première [0]
        wav = model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            speed=1.0
        )
        
        # Enregistrement natif à 24 000 Hz comme défini par pyVideoTrans
        sf.write(output_file, wav[0], 24000)
