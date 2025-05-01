import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pyannote.audio import Pipeline, Audio
from pyannote.core import Segment # Segmentオブジェクト作成のためインポート
from pathlib import Path
import warnings
import sys
import logging
from typing import Optional, Any, Dict, Tuple
import csv
import datetime
import argparse
import torchaudio # ファイル長取得のためインポート

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)

# Hugging Face トークンに関するFutureWarningを抑制
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')


def format_time(seconds: float) -> str:
    """秒数を HH:MM:SS 形式の文字列に変換します。"""
    delta = datetime.timedelta(seconds=seconds)
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # マイナスになる場合を考慮 (クリッピング等で稀に発生する可能性)
    if total_seconds < 0:
        return "00:00:00"
    return f"{hours:02}:{minutes:02}:{seconds:02}"

class AudioProcessor:
    """音声ファイルの話者分離と文字起こしを行い、結果をCSVに出力するクラス。"""

    SEGMENT_TOO_SHORT = "[Segment too short]"
    TRANSCRIPTION_FAILED = "[Transcription failed]"
    PROCESSING_ERROR_PREFIX = "[Error processing/transcribing segment: "
    # エラーメッセージ用定数を追加
    SEGMENT_START_BEYOND_DURATION = "[Segment start beyond audio duration]"
    SEGMENT_START_GE_END = "[Segment start >= end after clipping]"
    CROPPING_EMPTY_TENSOR = "[Cropping resulted in empty tensor]"
    CROPPING_INVALID_SAMPLERATE = "[Cropping returned invalid sample rate]"
    INVALID_WAVEFORM_FOR_DURATION = "[Invalid waveform/samplerate for duration calculation]"


    def __init__(self,
                 audio_file: Path,
                 output_csv_path: Path,
                 transcription_model_id: str,
                 pyannote_model_id: str,
                 target_sample_rate: int = 16000,
                 min_segment_duration: float = 0.02):
        """AudioProcessorを初期化します。"""
        logging.info(f"Initializing AudioProcessor for file: {audio_file}")
        if not audio_file.is_file():
             # ファイルが存在しない場合は早期にエラー
             logging.critical(f"Audio file not found at initialization: {audio_file}")
             raise FileNotFoundError(f"Audio file not found: {audio_file}")

        self.audio_file = audio_file
        self.output_csv_path = output_csv_path
        logging.info(f"Output CSV path set to: {self.output_csv_path}")

        self.transcription_model_id = transcription_model_id
        self.pyannote_model_id = pyannote_model_id
        self.target_sample_rate = target_sample_rate
        self.min_segment_duration = min_segment_duration # 閾値を属性として保持

        self.device, self.dtype = self._setup_device()
        self.pipeline, self.processor, self.model = self._load_models()
        self.audio_handler = self._setup_audio_handler()
        self.audio_duration = self._get_audio_duration() # ★ファイル長を取得
        logging.info("AudioProcessor initialized successfully.")

    def _setup_device(self) -> tuple[torch.device, torch.dtype]:
        """デバイスとデータ型を設定します。"""
        logging.debug("Setting up device and data type...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
            logging.info(f"Using CUDA device. Using dtype: {dtype}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            # MPSはfloat16を完全にはサポートしていない場合があるためfloat32が無難
            dtype = torch.float32
            logging.info(f"Using MPS device. Using dtype: {dtype}")
        else:
            device = torch.device("cpu")
            dtype = torch.float32
            logging.info(f"Using CPU device. Using dtype: {dtype}")
        logging.debug(f"Device set to: {device}, Dtype set to: {dtype}")
        return device, dtype

    def _load_models(self) -> tuple[Pipeline, WhisperProcessor, WhisperForConditionalGeneration]:
        """PyannoteとWhisper(文字起こし)のモデル/プロセッサをロードします。"""
        logging.info("Loading models...")
        logging.info(f"Loading Pyannote pipeline ({self.pyannote_model_id})...")
        try:
            # Hugging Face Hubからトークンを使ってアクセスする必要がある場合がある
            # pipeline = Pipeline.from_pretrained(self.pyannote_model_id, use_auth_token="YOUR_HF_TOKEN")
            pipeline: Pipeline = Pipeline.from_pretrained(self.pyannote_model_id)
            pipeline.to(self.device)
            logging.info("Pyannote pipeline loaded successfully.")
        except Exception as e:
            logging.critical(f"Error loading Pyannote pipeline ({self.pyannote_model_id}): {e}")
            logging.critical("Please ensure you have accepted the user conditions on Hugging Face Hub for the model.")
            logging.critical(f"Model page: https://huggingface.co/{self.pyannote_model_id}")
            logging.critical("You might also need to provide a Hugging Face authentication token if the model requires it.")
            sys.exit(1)

        logging.info(f"Loading transcription model ({self.transcription_model_id})...")
        try:
            processor: WhisperProcessor = WhisperProcessor.from_pretrained(self.transcription_model_id)
            model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(
                self.transcription_model_id,
                torch_dtype=self.dtype # Whisperはfloat16に対応していることが多い
            ).to(self.device)
            model.eval() # 推論モードに設定
            logging.info("Transcription model loaded successfully.")
        except Exception as e:
            logging.critical(f"Error loading transcription model ({self.transcription_model_id}): {e}")
            sys.exit(1)

        logging.info("All models loaded.")
        return pipeline, processor, model

    def _setup_audio_handler(self) -> Audio:
        """pyannote.audio.Audio インスタンスをセットアップします。"""
        logging.debug("Setting up pyannote.audio.Audio handler...")
        # target_sample_rate を Audio に渡す
        handler = Audio(sample_rate=self.target_sample_rate, mono=True)
        logging.debug(f"Audio handler set up with sample rate {self.target_sample_rate} and mono=True.")
        return handler

    # ★追加: ファイル長取得メソッド
    def _get_audio_duration(self) -> float:
        """音声ファイルの合計長さを秒単位で取得します。"""
        try:
            info = torchaudio.info(str(self.audio_file))
            # sample_rate が 0 の場合 ZeroDivisionError になるためチェック
            if info.sample_rate <= 0:
                logging.error(f"Audio file has invalid sample rate ({info.sample_rate}). Cannot determine duration.")
                return float('inf')
            duration = info.num_frames / info.sample_rate
            logging.info(f"Detected audio duration: {duration:.3f} seconds.")
            return duration
        except FileNotFoundError:
             # __init__でチェック済みだが念のため
             logging.error(f"Audio file not found when trying to get duration: {self.audio_file}")
             raise
        except Exception as e:
            logging.error(f"Could not determine audio duration using torchaudio: {e}. Cropping might fail for segments near the end.")
            # 処理を続けるために inf を返す（チェックが無効になる）
            return float('inf')

    def diarize(self, known_num_speakers: Optional[int] = None) -> Optional[Any]:
        """音声ファイルに対して話者分離を実行します。"""
        logging.info(f"Running speaker diarization on {self.audio_file}...")
        if known_num_speakers is not None:
            logging.info(f"Using pre-defined number of speakers: {known_num_speakers}")
            diarization_params = {"num_speakers": known_num_speakers}
        else:
            logging.info("Estimating the number of speakers automatically.")
            # num_speakers=None または min_speakers/max_speakers を指定可能
            # パフォーマンス向上のため、推定範囲を指定することも検討
            diarization_params = {
                # "min_speakers": 1, # 必要に応じて
                # "max_speakers": 5 # 必要に応じて
            }

        try:
            # pipelineに直接ファイルパスを渡す
            diarization = self.pipeline(str(self.audio_file), **diarization_params)
            logging.info("Speaker diarization complete.")
            # diarization結果がNoneでないことを確認
            if diarization is None:
                 logging.warning("Diarization pipeline returned None.")
                 return None
            return diarization
        except Exception as e:
            logging.error(f"Error during speaker diarization: {e}", exc_info=True)
            return None

    def transcribe_segment(self, waveform: torch.Tensor, sample_rate: int) -> Optional[str]:
        """単一の音声波形セグメントを文字起こしします。"""
        # 入力テンソルのチェック
        if waveform is None or waveform.numel() == 0:
            logging.warning("Received empty waveform for transcription.")
            return None
        if sample_rate <= 0:
            logging.warning(f"Received invalid sample rate ({sample_rate}) for transcription.")
            return None

        logging.debug(f"Transcribing segment with shape {waveform.shape} and sample rate {sample_rate}...")

        try:
            # WhisperProcessorにかける前にnumpy配列に変換 (float32を期待することが多い)
            waveform_np = waveform.numpy().astype('float32')

            logging.debug("Processing segment for transcription model...")
            input_features: torch.Tensor = self.processor(
                waveform_np,
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features

            # データ型とデバイスをモデルに合わせる
            input_features = input_features.to(self.device, dtype=self.dtype)
            logging.debug(f"Input features moved to device: {input_features.device}, dtype: {input_features.dtype}")

            logging.debug("Running transcription model generation...")
            # 推論中の勾配計算を無効化
            with torch.no_grad():
                # generateのパラメータ調整 (必要に応じて)
                # forced_decoder_ids を使って言語を指定する方が確実な場合がある
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="ja", task="transcribe")

                generated_ids = self.model.generate(
                    input_features,
                    # language="japanese", # language引数は非推奨になる可能性あり
                    forced_decoder_ids=forced_decoder_ids, # 強制デコードIDを使用
                    max_length=256 # 必要に応じて調整 (長くすると長い発話に対応できるが計算量増)
                )

            logging.debug("Decoding transcription...")
            # generated_ids が空でないか確認
            if generated_ids is None or generated_ids.numel() == 0:
                 logging.warning("Transcription model generated empty IDs.")
                 return None

            transcription: str = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            transcription_stripped = transcription.strip()
            logging.debug(f"Transcription successful: '{transcription_stripped}'")
            # 文字起こし結果が空文字列の場合も考慮
            if not transcription_stripped:
                 logging.debug("Transcription result is an empty string.")
                 # 空文字列を返すか、Noneを返すか、特別な文字列を返すか選択
                 return "" # または "[Empty transcription]" など

            return transcription_stripped

        except Exception as e:
            logging.warning(f"Error during transcription for a segment: {e}", exc_info=True)
            return None

    # ★★★ _process_audio_segment を修正 ★★★
    def _process_audio_segment(self, segment: Any, speaker: str) -> Dict[str, Any]:
        """
        一つの音声セグメントを処理し、文字起こし結果を含む辞書を返します。
        """
        segment_start: float = segment.start
        segment_end: float = segment.end
        original_segment_end = segment_end # 元の終了時間を保持
        start_time_str = format_time(segment_start)
        end_time_str = format_time(segment_end) # 初期値

        result = {
            "start": start_time_str,
            "end": end_time_str, # 初期値
            "speaker": speaker,
            "text": "",
            "is_valid": False # デフォルトは無効（書き込まない）
        }

        try:
            # --- セグメント境界チェックと調整 ---
            if segment_start >= self.audio_duration:
                 logging.warning(f"Segment start time {segment_start:.2f}s is at or beyond audio duration {self.audio_duration:.2f}s. Skipping.")
                 result["text"] = AudioProcessor.SEGMENT_START_BEYOND_DURATION
                 # is_valid=False のまま返す
                 # result["is_valid"] = True # ★記録したい場合
                 return result

            if segment_end > self.audio_duration:
                logging.warning(f"Segment end time {original_segment_end:.2f}s exceeds audio duration {self.audio_duration:.2f}s. Clipping end time to {self.audio_duration:.2f}s.")
                segment_end = self.audio_duration # ファイル長にクリップ
                end_time_str = format_time(segment_end) # 表示用タイムスタンプを更新
                result["end"] = end_time_str # 結果辞書も更新

            # 開始時間 >= 終了時間 になった場合 (クリップの結果、または元々非常に短い場合)
            if segment_start >= segment_end:
                 logging.warning(f"Segment start time {segment_start:.2f}s is greater than or equal to (clipped) end time {segment_end:.2f}s. Skipping.")
                 result["text"] = AudioProcessor.SEGMENT_START_GE_END
                 # is_valid=False のまま返す
                 # result["is_valid"] = True # ★記録したい場合
                 return result

            # crop に渡すための Segment オブジェクトを作成 (調整後の時間で)
            crop_segment = Segment(segment_start, segment_end)

            # --- crop 処理 ---
            logging.debug(f"Cropping audio segment {crop_segment} (Original end: {original_segment_end:.2f}s)...")
            waveform_tensor, sample_rate = self.audio_handler.crop(str(self.audio_file), crop_segment)

            # --- crop 結果のログとチェック ---
            logging.debug(f"Segment cropped. Waveform shape: {waveform_tensor.shape}, Sample Rate: {sample_rate}, Dtype: {waveform_tensor.dtype}")
            if waveform_tensor is None or waveform_tensor.numel() == 0:
                 logging.warning(f"Cropping returned an empty tensor for segment {crop_segment}. Skipping transcription.")
                 result["text"] = AudioProcessor.CROPPING_EMPTY_TENSOR
                 # is_valid=False のまま返す
                 # result["is_valid"] = True # ★記録したい場合
                 return result
            if sample_rate is None or sample_rate <= 0:
                 logging.warning(f"Cropping returned an invalid sample rate ({sample_rate}) for segment {crop_segment}. Skipping transcription.")
                 result["text"] = AudioProcessor.CROPPING_INVALID_SAMPLERATE
                 # is_valid=False のまま返す
                 # result["is_valid"] = True # ★記録したい場合
                 return result

            # --- 波形処理とセグメント長計算 ---
            # crop は (channel, time) の形のはずなので、mono=True なら channel=1
            # squeeze(0) で channel 次元を削除するか、waveform_tensor[0] を使う
            if waveform_tensor.ndim == 2 and waveform_tensor.shape[0] == 1:
                 waveform_processed: torch.Tensor = waveform_tensor.squeeze(0)
            elif waveform_tensor.ndim == 1: # 既に1次元の場合
                 waveform_processed: torch.Tensor = waveform_tensor
            else:
                 logging.warning(f"Unexpected waveform shape after cropping: {waveform_tensor.shape}. Trying to use first channel if possible.")
                 # 多チャンネルや予期せぬ形状の場合、最初のチャンネルを使う試み
                 if waveform_tensor.ndim > 1 and waveform_tensor.shape[0] > 0:
                      waveform_processed = waveform_tensor[0]
                 else: # どうしようもない場合
                      waveform_processed = waveform_tensor # そのまま渡してみる（エラーになる可能性大）


            logging.debug(f"Waveform processed to shape: {waveform_processed.shape}")

            segment_duration = 0.0
            # waveform_processed が1次元で要素を持つか確認
            if waveform_processed.ndim == 1 and waveform_processed.shape[0] > 0:
                 segment_duration = waveform_processed.shape[0] / sample_rate
                 logging.debug(f"Calculated segment duration: {segment_duration:.3f}s (shape[0]={waveform_processed.shape[0]}, sample_rate={sample_rate})")
            else:
                 # 長さが計算できない場合
                 logging.warning(f"Could not calculate valid segment duration for segment {crop_segment}. "
                                 f"Processed waveform ndim: {waveform_processed.ndim}, shape: {waveform_processed.shape}, sample_rate: {sample_rate}. Assuming zero duration.")
                 segment_duration = 0.0
                 result["text"] = AudioProcessor.INVALID_WAVEFORM_FOR_DURATION # エラー理由を記録

                 # この場合も is_valid=False のままか、Trueにして記録するか選択
                 # if not result["text"]: result["text"] = AudioProcessor.INVALID_WAVEFORM_FOR_DURATION
                 # result["is_valid"] = True
                 return result # is_valid=Falseのままなのでここで終了


            # --- 短すぎるセグメントの処理 ---
            if segment_duration < self.min_segment_duration:
                logging.warning(f"Segment {crop_segment} too short ({segment_duration:.3f}s < {self.min_segment_duration}s), skipping transcription.")
                result["text"] = AudioProcessor.SEGMENT_TOO_SHORT
                # is_valid は False のまま（CSVには書き込まない）
                # result["is_valid"] = True # ★もし短くてもCSVに記録したいならTrueにする
                return result # is_valid=Falseなのでここで終了
            else:
                # --- 文字起こし実行 ---
                transcription = self.transcribe_segment(waveform_processed, sample_rate)
                if transcription is not None:
                    # 正常終了
                    logging.info(f"  [{start_time_str} - {end_time_str}] {speaker}: {transcription}")
                    result["text"] = transcription
                    result["is_valid"] = True # 正常に文字起こしできたので有効
                else:
                    # 文字起こし自体が失敗 (Noneが返ってきた場合)
                    logging.warning(f"  [{start_time_str} - {end_time_str}] {speaker}: Transcription failed for this segment.")
                    result["text"] = AudioProcessor.TRANSCRIPTION_FAILED
                    result["is_valid"] = True # エラーでも記録は残すので有効

        except ValueError as ve: # crop で範囲外エラーが発生した場合など
             logging.warning(f"ValueError during processing segment [{segment_start:.2f}s - {original_segment_end:.2f}s]: {ve}", exc_info=False)
             result["text"] = f"{AudioProcessor.PROCESSING_ERROR_PREFIX}ValueError: {ve}]"
             result["is_valid"] = True # エラーでも記録
        except Exception as e:
             # その他の予期せぬエラー
            logging.warning(f"Unexpected error processing or transcribing audio segment [{segment_start:.2f}s - {original_segment_end:.2f}s]: {e}", exc_info=True)
            result["text"] = f"{AudioProcessor.PROCESSING_ERROR_PREFIX}{e}]"
            result["is_valid"] = True # エラーでも記録

        return result

    def _write_segment_to_csv(self, csv_writer: csv.writer, csv_file_handle: Any, segment_data: Dict[str, Any]):
        """
        処理されたセグメントデータをCSVファイルに書き込みます。
        is_valid が True の場合のみ書き込みます。
        """
        if not segment_data.get("is_valid", False): # .getで安全にアクセス
            logging.debug(f"Skipping writing segment [{segment_data.get('start', '?')} - {segment_data.get('end', '?')}] to CSV as it's marked invalid (is_valid=False). Reason: {segment_data.get('text', '[No reason provided]')}")
            return # is_validがFalseの場合は書き込まない

        try:
            csv_writer.writerow([
                segment_data["start"],
                segment_data["end"],
                segment_data["speaker"],
                segment_data["text"]
            ])
            csv_file_handle.flush() # 逐次書き込みのためにflush
        except Exception as write_e:
            logging.error(f"Error writing row to CSV for segment [{segment_data.get('start', '?')} - {segment_data.get('end', '?')}]: {write_e}")


    def process_and_save_to_csv(self, known_num_speakers: Optional[int] = None) -> bool:
        """
        音声ファイルの話者分離と文字起こしの全プロセスを実行し、結果をCSVに逐次書き込みします。
        """
        logging.info("Starting audio processing and CSV saving pipeline...")

        diarization = self.diarize(known_num_speakers=known_num_speakers)
        if diarization is None:
            logging.error("Failed to get diarization results. Aborting processing.")
            return False

        try:
            # 出力ディレクトリが存在しない場合は作成
            self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                header = ["start", "end", "speaker", "text"]
                csv_writer.writerow(header)
                csvfile.flush()
                logging.info(f"Writing results to {self.output_csv_path}...")

                # Pyannote 3.x の .itertracks(yield_label=True) を使う
                try:
                    # yield_label=True で (Segment, track_id, speaker_label) のタプルを取得
                    # イテレータをリストに変換してソートする
                    segments_with_labels = list(diarization.itertracks(yield_label=True))
                except Exception as e:
                    logging.error(f"Failed to iterate over diarization tracks: {e}", exc_info=True)
                    return False # セグメント取得失敗

                if not segments_with_labels:
                    logging.warning("No speaker segments detected in the audio by diarization pipeline.")
                    # 空のCSVファイルは作成されるが、処理自体は成功とする
                    return True

                # 開始時間でソート (Segmentオブジェクトは比較可能)
                sorted_segments: list[Tuple[Segment, str, str]] = sorted(segments_with_labels, key=lambda x: x[0].start)

                logging.info(f"Found {len(sorted_segments)} speaker segments. Starting processing and CSV writing loop.")

                segment: Segment
                _track_id: str # 不要だが受け取る必要あり
                speaker: str
                for i, (segment, _track_id, speaker) in enumerate(sorted_segments):
                    segment_index = i + 1
                    logging.info(f"--- Processing segment {segment_index}/{len(sorted_segments)} [{segment.start:.2f}s - {segment.end:.2f}s] Speaker: {speaker} ---")

                    # 1. セグメント処理 (文字起こし含む)
                    processed_data = self._process_audio_segment(segment, speaker)

                    # 2. CSV書き込み (is_valid が True の場合のみ)
                    self._write_segment_to_csv(csv_writer, csvfile, processed_data)

                    # エラーがあった場合のログレベル調整（オプション）
                    # if not processed_data["is_valid"] and AudioProcessor.PROCESSING_ERROR_PREFIX in processed_data["text"]:
                    #     logging.error(f"Segment {segment_index} encountered an error: {processed_data['text']}")
                    # elif not processed_data["is_valid"]:
                    #      logging.warning(f"Segment {segment_index} was skipped or invalid: {processed_data['text']}")


                    logging.info(f"--- Finished segment {segment_index}/{len(sorted_segments)} ---")


            logging.info(f"Successfully finished writing results to {self.output_csv_path}")
            return True

        except OSError as e:
            logging.error(f"Error opening or writing to CSV file {self.output_csv_path}: {e}")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred during the main CSV processing loop: {e}", exc_info=True)
            return False

def create_transcript_csv_path(audio_file_path: Path) -> Path:
    """
    指定された音声ファイルパスから、指定の命名規則に従った
    文字起こし結果CSVファイルのPathオブジェクトを生成します。
    出力先はカレントディレクトリになります。
    """
    base_name = audio_file_path.stem
    now = datetime.datetime.now()
    timestamp_str = now.strftime('%Y%m%d%H%M%S')
    output_filename = f"{base_name}-transcription-{timestamp_str}.csv"
    # カレントディレクトリに出力
    output_path = Path.cwd() / output_filename
    return output_path

# ---- スクリプト実行部分 ----
if __name__ == "__main__":

    # --- 追加: torchaudio のインポートチェック ---
    try:
        import torchaudio
    except ImportError:
        # ログ設定が有効になる前にエラーが発生する可能性があるため、stderr にも出力
        print("Critical Error: torchaudio is required but not installed. Please install it using: pip install torchaudio", file=sys.stderr)
        logging.critical("Error: torchaudio is required but not installed. Please install it using: pip install torchaudio")
        sys.exit(1)


    parser = argparse.ArgumentParser(description="Audio processing script with speaker diarization and transcription.")
    parser.add_argument("audio_file_path", type=Path, help="Path to the audio file to process.")
    # 出力パスを必須ではなくオプション（デフォルト生成）に変更
    parser.add_argument("--output_csv_path", type=Path, default=None,
                        help="Path to the output CSV file. If not specified, generates '<audio_filename>-transcription-<timestamp>.csv' in the current directory.")
    parser.add_argument("--transcription_model_id", type=str, default="openai/whisper-large-v3", help="Hugging Face ID of the transcription model (e.g., 'openai/whisper-large-v3', 'openai/whisper-medium').")
    parser.add_argument("--pyannote_model_id", type=str, default="pyannote/speaker-diarization-3.1", help="Hugging Face ID of the Pyannote diarization model (e.g., 'pyannote/speaker-diarization-3.1').")
    parser.add_argument("--num_speakers", type=int, default=None, help="Known number of speakers. If not specified, the model estimates automatically.")
    parser.add_argument("--min_segment_duration", type=float, default=0.02, help="Minimum duration (seconds) for a segment to be transcribed. Segments shorter than this will be skipped.")
    # 必要なら他の引数も追加 (例: デバイス指定、トークンなど)
    # parser.add_argument("--device", type=str, default=None, choices=['cuda', 'cpu', 'mps'], help="Force specific device ('cuda', 'cpu', 'mps'). Auto-detects if not set.")
    # parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face Hub token if required for private models.")

    args = parser.parse_args()

    audio_file_path = args.audio_file_path
    output_csv_path = args.output_csv_path
    transcription_model_id = args.transcription_model_id
    pyannote_model_id = args.pyannote_model_id
    known_num_speakers = args.num_speakers
    min_segment_duration = args.min_segment_duration

    # --- 初期チェック ---
    if not audio_file_path.is_file():
        logging.critical(f"Critical Error: Audio file not found at {audio_file_path}")
        sys.exit(1)
    else:
        logging.info(f"Audio file found at {audio_file_path}.")

    # 出力パスが指定されていない場合は生成
    if output_csv_path is None:
        try:
             output_csv_path = create_transcript_csv_path(audio_file_path)
             logging.info(f"Output CSV path not specified, defaulting to: {output_csv_path}")
        except Exception as e:
             logging.critical(f"Critical Error: Could not generate default output CSV path: {e}")
             sys.exit(1)
    else:
         logging.info(f"Output CSV path specified: {output_csv_path}")


    # 出力ディレクトリの確認/作成 (ファイルを開く直前ではなく、早めに行う)
    try:
        output_dir = output_csv_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory checked/created: {output_dir}")
    except OSError as e:
        logging.critical(f"Critical Error: Could not create output directory {output_dir}: {e}")
        sys.exit(1)
    except Exception as e: # その他の予期せぬエラー (Pathオブジェクト関連など)
        logging.critical(f"Critical Error: Failed to prepare output directory {output_csv_path.parent}: {e}")
        sys.exit(1)


    # --- メイン処理 ---
    logging.info("Script execution started.")
    logging.info(f"Audio file path: {audio_file_path}")
    logging.info(f"Transcription model ID: {transcription_model_id}")
    logging.info(f"Pyannote Diarization model ID: {pyannote_model_id}")
    logging.info(f"Minimum segment duration for transcription: {min_segment_duration}s")

    if known_num_speakers is not None:
        logging.info(f"Number of speakers specified: {known_num_speakers}")
    else:
        logging.info("Number of speakers not specified, will estimate automatically.")


    try:
        # AudioProcessor のインスタンス化でファイル存在チェックとファイル長取得が行われる
        processor = AudioProcessor(
            audio_file=audio_file_path,
            output_csv_path=output_csv_path,
            transcription_model_id=transcription_model_id,
            pyannote_model_id=pyannote_model_id,
            min_segment_duration=min_segment_duration
        )

        success = processor.process_and_save_to_csv(known_num_speakers=known_num_speakers)

        if success:
            logging.info(f"Processing complete. Results saved to {processor.output_csv_path}")
        else:
            # 失敗理由は process_and_save_to_csv 内のログで出力されているはず
            logging.error("Processing failed. Please check the logs above for details.")
            sys.exit(1) # 失敗時はエラーコードで終了

    except FileNotFoundError as e:
         # AudioProcessor初期化時のファイル未検出
         logging.critical(f"Critical Error during initialization: {e}")
         sys.exit(1)
    except Exception as e:
        # AudioProcessorの初期化失敗や、process_and_save_to_csvの外での予期せぬエラー
        logging.critical(f"A critical error occurred during the main execution: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Script execution finished successfully.")
    sys.exit(0) # 正常終了
