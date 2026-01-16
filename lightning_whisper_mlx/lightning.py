import os
from .transcribe import transcribe_audio
from huggingface_hub import hf_hub_download

models = {
    "tiny": {
        "base": "mlx-community/whisper-tiny",
        "4bit": "mlx-community/whisper-tiny-mlx-4bit",
        "8bit": "mlx-community/whisper-tiny-mlx-8bit"
    },
    "small": {
        "base": "mlx-community/whisper-small-mlx",
        "4bit": "mlx-community/whisper-small-mlx-4bit",
        "8bit": "mlx-community/whisper-small-mlx-8bit"
    },
    "distil-small.en": {
        "base": "mustafaaljadery/distil-whisper-mlx",
    },
    "base": {
        "base": "mlx-community/whisper-base-mlx",
        "4bit": "mlx-community/whisper-base-mlx-4bit",
        "8bit": "mlx-community/whisper-base-mlx-8bit"
    },
    "medium": {
        "base": "mlx-community/whisper-medium-mlx",
        "4bit": "mlx-community/whisper-medium-mlx-4bit",
        "8bit": "mlx-community/whisper-medium-mlx-8bit"
    },
    "distil-medium.en": {
        "base": "mustafaaljadery/distil-whisper-mlx",
    },
    "large": {
        "base": "mlx-community/whisper-large-mlx",
        "4bit": "mlx-community/whisper-large-mlx-4bit",
        "8bit": "mlx-community/whisper-large-mlx-8bit",
    },
    "large-v2": {
        "base": "mlx-community/whisper-large-v2-mlx",
        "4bit": "mlx-community/whisper-large-v2-mlx-4bit",
        "8bit": "mlx-community/whisper-large-v2-mlx-8bit",
    },
    "distil-large-v2": {
        "base": "mustafaaljadery/distil-whisper-mlx",
    },
    "large-v3": {
        "base": "mlx-community/whisper-large-v3-mlx",
        "4bit": "mlx-community/whisper-large-v3-mlx-4bit",
        "8bit": "mlx-community/whisper-large-v3-mlx-8bit",
    },
    "distil-large-v3": {
        "base": "mustafaaljadery/distil-whisper-mlx",
    },
    "whisper-large-v3-turbo": {
        "base": "mlx-community/whisper-large-v3-turbo",
    },
}


class LightningWhisperMLX():
    def __init__(self, model, batch_size=12, quant=None):
        if quant and (quant != "4bit" and quant != "8bit"):
            raise ValueError("Quantization must be `4bit` or `8bit`")

        self.name = model
        self.batch_size = batch_size

        # Check if model is a local path
        if os.path.isdir(model):
            self._use_local_model(model)
            return

        # Otherwise use predefined models
        self._download_predefined_model(model, quant)

    def _use_local_model(self, model_path):
        """Use a local model directory."""
        self.model_path = os.path.abspath(model_path)
        print(f"Using local model from: {self.model_path}")

    def _download_predefined_model(self, model, quant):
        """Download a predefined model from Hugging Face."""
        if model not in models:
            raise ValueError("Please select a valid model")

        repo_id = self._get_repo_id(model, quant)
        self._update_name_for_quantization(model, quant)
        self._download_model_files(repo_id, model)

    def _get_repo_id(self, model, quant):
        """Get the Hugging Face repository ID."""
        if quant and "distil" not in model:
            return models[model][quant]
        return models[model]['base']

    def _update_name_for_quantization(self, model, quant):
        """Update model name for quantized distil models."""
        if quant and "distil" in model:
            if quant == "4bit":
                self.name += "-4-bit"
            else:
                self.name += "-8-bit"

    def _download_model_files(self, repo_id, model):
        """Download model weights and config files."""
        if "distil" in model:
            filename1 = f"./mlx_models/{self.name}/weights.safetensors"
            filename2 = f"./mlx_models/{self.name}/config.json"
            local_dir = "./"
        else:
            filename1 = "weights.safetensors"
            filename2 = "config.json"
            local_dir = f"./mlx_models/{self.name}"

        # Try to download safetensors first, fallback to npz if not available
        try:
            hf_hub_download(repo_id=repo_id, filename=filename1,
                            local_dir=local_dir)
        except Exception:
            # Fallback to npz if safetensors not available
            if "distil" in model:
                filename1 = f"./mlx_models/{self.name}/weights.npz"
            else:
                filename1 = "weights.npz"
            hf_hub_download(repo_id=repo_id, filename=filename1,
                            local_dir=local_dir)

        hf_hub_download(repo_id=repo_id, filename=filename2,
                        local_dir=local_dir)

    def transcribe(self, audio_path, language=None, initial_prompt=None, word_timestamps: bool = True, **kwargs):
        # Use local model path if available, otherwise use downloaded model
        if hasattr(self, 'model_path'):
            model_path = self.model_path
        else:
            model_path = f'./mlx_models/{self.name}'

        result = transcribe_audio(
            audio_path, path_or_hf_repo=model_path, language=language, batch_size=self.batch_size, initial_prompt=initial_prompt, word_timestamps=word_timestamps, **kwargs)
        return result
