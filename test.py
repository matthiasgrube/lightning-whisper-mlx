from lightning_whisper_mlx import LightningWhisperMLX

whisper = LightningWhisperMLX(
    model="whisper-large-v3-turbo", batch_size=12, quant=None)

text = whisper.transcribe(audio_path="./time_short.mp3", language="de")

print(text)
