import torch
from torch import Tensor, device as TorchDevice, dtype as TorchDtype
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pyannote.audio import Pipeline, Audio
from pyannote.core import Annotation
from pathlib import Path


# 文字起こし対象の音声ファイル
audio_file: Path = Path("./short.wav" )
# 話者分離を行うモデル
pyannote_model: str = "pyannote/speaker-diarization-3.1"
# 文字起こしを行うモデル
transcription_model: str = "openai/whisper-large-v3"

# デバイス選択とデータ型設定
device: TorchDevice = torch.device("cpu")
dtype: TorchDtype = torch.float32

# PyannoteパイプラインとWhisperモデル/プロセッサ、Audioハンドラをロード
pipeline: Pipeline = Pipeline.from_pretrained(pyannote_model).to(device)
processor: WhisperProcessor = WhisperProcessor.from_pretrained(transcription_model)
model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(transcription_model, torch_dtype=dtype).to(device).eval()
audio_handler: Audio = Audio(sample_rate=16000, mono=True) 


# --- 2. 話者分離を実行 ---
diarization: Annotation = pipeline(audio_file, num_speakers = 2) 

# diarization.itertracks()で各発話区間(segment)と話者ラベル(speaker)を取得
for segment, _, speaker in diarization.itertracks(yield_label=True):
    # audio_handler.crop()で該当区間の音声波形を読み込み (16kHzモノラルに変換)
    waveform, sample_rate = audio_handler.crop(audio_file, segment)

    # transformers版Whisperで文字起こしを実行
    input_features: Tensor = processor(
        waveform.squeeze().numpy().astype("float32"), # 波形をnumpy float32配列に
        sampling_rate=sample_rate,                 # サンプルレート指定
        return_tensors="pt"                      # PyTorchテンソルで返す
    ).input_features.to(device, dtype=dtype)       # モデルと同じデバイス・データ型へ

    # 2. モデルでIDシーケンスを生成 (勾配計算なし)
    with torch.no_grad():
        # 日本語を指定して文字起こしタスクを実行
        predicted_ids: Tensor = model.generate(
            input_features,
            forced_decoder_ids=processor.get_decoder_prompt_ids(language="ja", task="transcribe")
        )

    # 3. IDシーケンスをテキストにデコード
    text: str = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    print(f"[{segment.start:03.1f}s - {segment.end:03.1f}s] {speaker}: {text}")
