from .omnivoice import OmniVoiceProvider

# Enregistrement de nos Providers actifs
PROVIDERS_REGISTRY = {
    "Custom": OmniVoiceProvider()
}

def get_provider(voice_name):
    """
    Analyse le préfixe de la voix (ex: 'Custom/Moussa_FR' -> 'Custom')
    et retourne le Provider correspondant.
    """
    if not isinstance(voice_name, str):
        return None
        
    prefix = voice_name.split("/")[0] if "/" in voice_name else "Edge"
    return PROVIDERS_REGISTRY.get(prefix, None)
