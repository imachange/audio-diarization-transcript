import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pyannote.audio import Pipeline, Audio
from pathlib import Path
import warnings
import sys
import logging
from typing import Optional, Any, Dict, Tuple
import csv
import datetime
import argparse

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
    return f"{hours:02}:{minutes:02}:{seconds:02}"

class AudioProcessor:
    """音声ファイルの話者分離と文字起こしを行い、結果をCSVに出力するクラス。"""

    SEGMENT_TOO_SHORT = "[Segment too short]"
    TRANSCRIPTION_FAILED = "[Transcription failed]"
    PROCESSING_ERROR_PREFIX = "[Error processing/transcribing segment: "

    def __init__(self,
                 audio_file: Path,
                 output_csv_path: Path,
                 transcription_model_id: str,
                 pyannote_model_id: str,
                 target_sample_rate: int = 16000,
                 min_segment_duration: float = 0.02): # 短すぎると判断する閾値を追加
        """AudioProcessorを初期化します。"""
        logging.info(f"Initializing AudioProcessor for file: {audio_file}")
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
            pipeline: Pipeline = Pipeline.from_pretrained(self.pyannote_model_id)
            pipeline.to(self.device)
            logging.info("Pyannote pipeline loaded successfully.")
        except Exception as e:
            logging.critical(f"Error loading Pyannote pipeline ({self.pyannote_model_id}): {e}")
            logging.critical("Please ensure you have accepted the user conditions on Hugging Face Hub for the model.")
            logging.critical(f"Model page: https://huggingface.co/{self.pyannote_model_id}")
            sys.exit(1)

        logging.info(f"Loading transcription model ({self.transcription_model_id})...")
        try:
            processor: WhisperProcessor = WhisperProcessor.from_pretrained(self.transcription_model_id)
            model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(
                self.transcription_model_id,
                torch_dtype=self.dtype
            ).to(self.device)
            model.eval()
            logging.info("Transcription model loaded successfully.")
        except Exception as e:
            logging.critical(f"Error loading transcription model ({self.transcription_model_id}): {e}")
            sys.exit(1)

        logging.info("All models loaded.")
        return pipeline, processor, model

    def _setup_audio_handler(self) -> Audio:
        """pyannote.audio.Audio インスタンスをセットアップします。"""
        logging.debug("Setting up pyannote.audio.Audio handler...")
        handler = Audio(sample_rate=self.target_sample_rate, mono=True)
        logging.debug(f"Audio handler set up with sample rate {self.target_sample_rate} and mono=True.")
        return handler

    def diarize(self, known_num_speakers: Optional[int] = None) -> Optional[Any]:
        """音声ファイルに対して話者分離を実行します。"""
        logging.info(f"Running speaker diarization on {self.audio_file}...")
        if known_num_speakers is not None:
            logging.info(f"Using pre-defined number of speakers: {known_num_speakers}")
        else:
            logging.info("Estimating the number of speakers automatically.")

        try:
            diarization = self.pipeline(str(self.audio_file), num_speakers=known_num_speakers)
            logging.info("Speaker diarization complete.")
            return diarization
        except Exception as e:
            logging.error(f"Error during speaker diarization: {e}", exc_info=True)
            return None

    def transcribe_segment(self, waveform: torch.Tensor, sample_rate: int) -> Optional[str]:
        """単一の音声波形セグメントを文字起こしします。"""
        logging.debug(f"Transcribing segment with shape {waveform.shape} and sample rate {sample_rate}...")
        try:
            logging.debug("Processing segment for transcription model...")
            input_features: torch.Tensor = self.processor(
                waveform.numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features
            input_features = input_features.to(self.device, dtype=self.dtype)
            logging.debug(f"Input features moved to device: {input_features.device}, dtype: {input_features.dtype}")

            logging.debug("Running transcription model generation...")
            with torch.no_grad():
                # generateのパラメータ調整 (必要に応じて)
                generated_ids = self.model.generate(
                    input_features,
                    language="japanese",
                    max_length=256 
                )

            logging.debug("Decoding transcription...")
            transcription: str = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            transcription_stripped = transcription.strip()
            logging.debug(f"Transcription successful: '{transcription_stripped}'")
            return transcription_stripped

        except Exception as e:
            logging.warning(f"Error during transcription for a segment: {e}", exc_info=True)
            return None

    def _process_audio_segment(self, segment: Any, speaker: str) -> Dict[str, Any]:
        """
        一つの音声セグメントを処理し、文字起こし結果を含む辞書を返します。

        Args:
            segment: pyannoteから得られたセグメントオブジェクト。
            speaker: セグメントに割り当てられた話者ラベル。

        Returns:
            処理結果を含む辞書。キー: 'start', 'end', 'speaker', 'text', 'is_valid'
            'is_valid' は、セグメントが処理可能でCSVに書き込むべきかを示すブール値。
        """
        segment_start: float = segment.start
        segment_end: float = segment.end
        start_time_str = format_time(segment_start)
        end_time_str = format_time(segment_end)
        result = {
            "start": start_time_str,
            "end": end_time_str,
            "speaker": speaker,
            "text": "",
            "is_valid": False # デフォルトは無効（書き込まない）
        }

        try:
            logging.debug(f"Cropping audio segment from {segment_start:.2f}s to {segment_end:.2f}s...")
            waveform_tensor, sample_rate = self.audio_handler.crop(str(self.audio_file), segment)
            logging.debug(f"Segment cropped successfully. Shape: {waveform_tensor.shape}, Sample Rate: {sample_rate}")

            # Pyannoteが出力するセグメントはモノラルを想定しているため、通常squeeze()は不要だが念のため
            waveform_processed: torch.Tensor = waveform_tensor.squeeze()
            logging.debug(f"Waveform potentially squeezed to shape: {waveform_processed.shape}")

            # セグメントの長さをチェック
            segment_duration = 0.0
            if waveform_processed.ndim > 0 and sample_rate > 0:
                 segment_duration = waveform_processed.shape[0] / sample_rate

            if segment_duration < self.min_segment_duration:
                logging.warning(f"Segment [{segment_start:.2f}s - {segment_end:.2f}s] too short ({segment_duration:.3f}s < {self.min_segment_duration}s), skipping transcription.")
                result["text"] = AudioProcessor.SEGMENT_TOO_SHORT
                # is_valid は False のまま（CSVには書き込まない）
            else:
                # 文字起こし実行
                transcription = self.transcribe_segment(waveform_processed, sample_rate)
                if transcription is not None:
                    logging.info(f"  [{segment_start:03.2f}s - {segment_end:03.2f}s] {speaker}: {transcription}")
                    result["text"] = transcription
                    result["is_valid"] = True # 正常に文字起こしできたので有効
                else:
                    logging.warning(f"  [{segment_start:03.2f}s - {segment_end:03.2f}s] {speaker}: Transcription failed for this segment.")
                    result["text"] = AudioProcessor.TRANSCRIPTION_FAILED
                    result["is_valid"] = True # エラーでも記録は残すので有効

        except Exception as e:
            logging.warning(f"Error processing or transcribing audio segment [{segment_start:.2f}s - {segment_end:.2f}s]: {e}", exc_info=True)
            result["text"] = f"{AudioProcessor.PROCESSING_ERROR_PREFIX}{e}]"
            result["is_valid"] = True # エラーでも記録は残すので有効

        return result

    def _write_segment_to_csv(self, csv_writer: csv.writer, csv_file_handle: Any, segment_data: Dict[str, Any]):
        """
        処理されたセグメントデータをCSVファイルに書き込みます。
        """
        if not segment_data["is_valid"]:
            logging.debug(f"Skipping writing segment [{segment_data['start']} - {segment_data['end']}] to CSV as it's marked invalid (e.g., too short).")
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
            logging.error(f"Error writing row to CSV for segment [{segment_data['start']} - {segment_data['end']}]: {write_e}")


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
            with open(self.output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                header = ["start", "end", "speaker", "text"]
                csv_writer.writerow(header)
                csvfile.flush()
                logging.info(f"Writing results to {self.output_csv_path}...")

                # Pyannote 3.x では .itertracks() は直接リストではない可能性があるため、リストに変換してからソート
                try:
                    # yield_label=True で (Segment, track_id, speaker_label) のタプルを取得
                    segments_list = list(diarization.itertracks(yield_label=True))
                except Exception as e:
                    logging.error(f"Failed to iterate over diarization tracks: {e}", exc_info=True)
                    return False # セグメント取得失敗

                if not segments_list:
                    logging.warning("No speaker segments detected in the audio by diarization pipeline.")
                    return True # 話者がいないのはエラーではない

                # 開始時間でソート
                sorted_segments: list[Tuple[Any, str, str]] = sorted(segments_list, key=lambda x: x[0].start)

                logging.info(f"Found {len(sorted_segments)} speaker segments. Starting processing and CSV writing loop.")

                segment: Any
                _track_id: str # 不要だが受け取る必要あり
                speaker: str
                for i, (segment, _track_id, speaker) in enumerate(sorted_segments):
                    segment_index = i + 1
                    logging.info(f"--- Processing segment {segment_index}/{len(sorted_segments)} ---")

                    # 1. セグメント処理 (文字起こし含む)
                    processed_data = self._process_audio_segment(segment, speaker)

                    # 2. CSV書き込み (書き込み可否は_process_audio_segmentの結果次第)
                    self._write_segment_to_csv(csv_writer, csvfile, processed_data)

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
    指定されたWAVファイルパスから、指定の命名規則に従った
    文字起こし結果CSVファイルのPathオブジェクトを生成します。
    出力先はカレントディレクトリになります。

    Args:
        input_wav_path_str: 入力WAVファイルのパス文字列 (例: "/path/to/audio.wav")

    Returns:
        生成されたCSVファイルのPathオブジェクト (例: PosixPath('audio-transcription-20250501151031.csv'))
    """

    # 拡張子なしのファイル名を取得 (例: "audio")
    base_name =  audio_file_path.stem


    now = datetime.datetime.now()
    timestamp_str = now.strftime('%Y%m%d%H%M%S')

    # 5. 新しいファイル名を組み立て
    output_filename = f"{base_name}-transcription-{timestamp_str}.csv"

    # 6. 出力ファイルパスのPathオブジェクトを作成（カレントディレクトリ基準）
    output_path = Path.cwd() / output_filename 

    return output_path



# ---- スクリプト実行部分 ----
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Audio processing script with speaker diarization and transcription.")
    parser.add_argument("audio_file_path", type=Path, help="Path to the audio file to process.")
    parser.add_argument("--output_csv_path", type=Path, default=None, help="Path to the output CSV file. Defaults to audio filename with .csv extension.")
    parser.add_argument("--transcription_model_id", type=str, default="openai/whisper-large-v3", help="Hugging Face ID of the transcription model.")
    parser.add_argument("--pyannote_model_id", type=str, default="pyannote/speaker-diarization-3.1", help="Hugging Face ID of the Pyannote diarization model.")
    parser.add_argument("--num_speakers", type=int, default=None, help="Known number of speakers. If None, estimates automatically.")
    parser.add_argument("--min_segment_duration", type=float, default=0.02, help="Minimum duration (seconds) for a segment to be transcribed.") # 引数追加

    args = parser.parse_args()

    audio_file_path = args.audio_file_path
    output_csv_path = args.output_csv_path
    transcription_model_id = args.transcription_model_id
    pyannote_model_id = args.pyannote_model_id
    known_num_speakers = args.num_speakers
    min_segment_duration = args.min_segment_duration # 引数取得

    if output_csv_path is None:
        output_csv_path = create_transcript_csv_path(audio_file_path)
        logging.info(f"Output CSV path not specified, defaulting to: {output_csv_path}")
    else:
         logging.info(f"Output CSV path specified: {output_csv_path}")

    logging.info("Script execution started.")
    logging.info(f"Audio file path: {audio_file_path}")
    logging.info(f"Transcription model ID: {transcription_model_id}")
    logging.info(f"Pyannote Diarization model ID: {pyannote_model_id}")
    logging.info(f"Minimum segment duration for transcription: {min_segment_duration}s")

    if known_num_speakers is not None:
        logging.info(f"Number of speakers specified: {known_num_speakers}")
    else:
        logging.info("Number of speakers not specified, will estimate automatically.")

    if not audio_file_path.is_file():
        logging.critical(f"Critical Error: Audio file not found at {audio_file_path}")
        sys.exit(1)
    else:
        logging.info(f"Audio file found at {audio_file_path}.")

    output_dir = output_csv_path.parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory checked/created: {output_dir}")
    except OSError as e:
        logging.critical(f"Critical Error: Could not create output directory {output_dir}: {e}")
        sys.exit(1)

    try:
        processor = AudioProcessor(
            audio_file=audio_file_path,
            output_csv_path=output_csv_path,
            transcription_model_id=transcription_model_id,
            pyannote_model_id=pyannote_model_id,
            min_segment_duration=min_segment_duration # 初期化時に渡す
        )

        success = processor.process_and_save_to_csv(known_num_speakers=known_num_speakers)

        if success:
            logging.info(f"Processing complete. Results saved to {processor.output_csv_path}")
        else:
            logging.error("Processing failed. Check the logs for details.")

    except Exception as e:
        logging.critical(f"An critical error occurred during the main execution: {e}", exc_info=True)
        sys.exit(1)

    logging.info("Script execution finished.")
