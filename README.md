# audio-diarization-transcript

ローカル環境で動作し、話者分離機能も備えた音声文字起こしプログラム

## 実行環境

* **Python:** 3.11
* **主要ライブラリ:** PyTorch, Transformers, pyannote.audio, torchaudio (詳細は `pyproject.toml` を参照し、`uv sync` でインストールされます)
* **実行ハードウェア:**
    *  M3 MacBook Air 2024 メモリ 24GB

## 環境構築

1.  **uv の導入:**
    * Pythonのパッケージ管理ツール `uv` をインストールします。導入方法は公式ドキュメントを参照してください: [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)

2.  **依存ライブラリのインストール:**
    * プロジェクトのルートディレクトリで以下のコマンドを実行し、必要なライブラリをインストールします。
    ```bash
    uv sync
    ```

3.  **Hugging Face の設定:**
    * **アクセストークン取得:** Hugging Face Hub にログインし、アクセストークン（READ権限）を取得します。
        * 参考: [Hugging Faceでアクセストークンを取得する方法](https://monomonotech.jp/kurage/memo/m250108_huggingface_get_token)
    * **モデル利用規約への同意:** Hugging Face Hub上で、以下のモデルの利用規約に同意する必要があります。（スクリプトが内部で依存している可能性があるため）
        * [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
        * [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
    * **Hugging Face CLI ログイン:** コマンドラインから Hugging Face にログインし、取得したトークンを設定します。`uv` 環境内で `huggingface-cli` を実行します。
    ```bash
    uv run huggingface-cli login
    ```
    * 以下のようなプロンプトが表示されるので、取得したアクセストークンをペーストしてEnterキーを押します（入力は表示されません）。
    ```text
      _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
      _|    _|  _|    _|  _|        _|              _|    _|_|    _|  _|              _|        _|    _|  _|        _|
      _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
      _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
      _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

      A token is already saved on your machine. Run `huggingface-cli whoami` to get more information or `huggingface-cli logout` if you want to log out.
      Setting a new token will erase the existing one.
      To log in, `huggingface_hub` requires a token generated from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) .
    Enter your token (input will not be visible): [ここに取得したアクセストークンをペースト]
    ```

## コマンドの使い方

`main.py` スクリプトはコマンドライン引数を受け取ります。`uv` を使って実行します。

**基本コマンド:**
```bash
uv run python main.py <audio_file_path> [OPTIONS]
```

**引数:**

* **`audio_file_path` (必須):**
    * 処理対象の音声ファイル（例: `.wav`, `.mp3`）へのパスを指定します。
    * 例: `data/my_audio.wav`

* **`--output_csv_path PATH` (オプション):**
    * 出力するCSVファイルのパスを指定します。
    * 指定しない場合、スクリプトを実行したカレントディレクトリに `<音声ファイル名>-transcription-<YYYYMMDDHHMMSS>.csv` という形式で自動生成されます。
    * 例: `--output_csv_path results/transcript.csv`

* **`--transcription_model_id MODEL_ID` (オプション):**
    * 文字起こしに使用する Hugging Face のモデルIDを指定します。
    * デフォルト: `openai/whisper-large-v3`
    * 他のモデル例: `openai/whisper-medium`, `openai/whisper-small` , `kotoba-tech/kotoba-whisper-v2.2`等。使用するモデルは事前に Hugging Face Hub で確認してください。
    * 例: `--transcription_model_id openai/whisper-medium`

* **`--pyannote_model_id MODEL_ID` (オプション):**
    * 話者分離に使用する Hugging Face の Pyannote モデルIDを指定します。
    * デフォルト: `pyannote/speaker-diarization-3.1`
    * 他のモデルを使用する場合は、そのモデルIDを指定します。
    * 例: `--pyannote_model_id pyannote/speaker-diarization-2.1` (古いモデルなど)

* **`--num_speakers N` (オプション):**
    * 音声ファイルに含まれる話者の数を整数で指定します。
    * 指定しない場合、モデルが自動的に話者数を推定します。
    * 例: `--num_speakers 3`

* **`--min_segment_duration SECONDS` (オプション):**
    * 文字起こしを行う最小の音声セグメント長（秒）を指定します。
    * この値より短いセグメントは処理がスキップされ、CSVには `[Segment too short]` と記録されることがあります（ただし、デフォルトではスキップされた行はCSVに出力されません）。
    * デフォルト: `0.02`
    * 例: `--min_segment_duration 0.5` (0.5秒未満の発話は無視)

**実行例:**

1.  **必須引数のみで実行 (他はデフォルト値を使用):**
    ```bash
    uv run python main.py path/to/your/audio.mp3
    ```
    * 出力ファイルはカレントディレクトリに `audio-transcription-YYYYMMDDHHMMSS.csv` のような名前で生成されます。
    * Whisper-large-v3 と pyannote/speaker-diarization-3.1 を使用し、話者数は自動推定されます。

2.  **オプションを指定して実行:**
    ```bash
    uv run python main.py input/meeting.wav --output_csv_path output/meeting_transcript.csv --num_speakers 4 --transcription_model_id openai/whisper-medium
    ```
    * `input/meeting.wav` を処理します。
    * 結果は `output/meeting_transcript.csv` に保存されます。
    * 話者数を4人と指定します。
    * 文字起こしには Whisper-medium モデルを使用します。

## 出力ファイルの説明

スクリプトは、話者分離と文字起こしの結果をCSVファイルに出力します。

* **ファイル名:**
    * `--output_csv_path` で指定されたパスに出力されます。
    * 指定がない場合は、カレントディレクトリに `<入力ファイル名>-transcription-<YYYYMMDDHHMMSS>.csv` という名前で生成されます。 (`<YYYYMMDDHHMMSS>` は実行時のタイムスタンプです。)

* **フォーマット:**
    * CSVファイルはUTF-8エンコーディング (BOM付き) で保存されます。
    * 以下の列を含みます:
        * `start`: 発話セグメントの開始時間 (フォーマット: `HH:MM:SS`)
        * `end`: 発話セグメントの終了時間 (フォーマット: `HH:MM:SS`)
        * `speaker`: 識別された話者のラベル (例: `SPEAKER_00`, `SPEAKER_01`, ...)
        * `text`: 文字起こしされたテキスト内容。処理中にエラーが発生した場合、`[エラー内容]` (例: `[Transcription failed]`) が記録されることがあります。

* **データ例:**
    ```csv
    "start","end","speaker","text"
    "00:00:00","00:00:04","SPEAKER_01","さて今回はテクニカルライティング入門講座の抜粋資料ですねはい"
    "00:00:05","00:00:11","SPEAKER_01","技術者にとって、この書くスキルって、結構永遠の課題みたいなところがあるかもしれません。"
    "00:00:11","00:00:15","SPEAKER_00","そうですね苦手意識持ってる方は少なくないですよね"
    "00:00:15","00:00:16","SPEAKER_01","ですよね"
    "00:00:16","00:00:27","SPEAKER_01","今日はそのあたりを克服して、情報を明確にかつ簡潔に伝えるための実践的なヒント、その革新部分を一緒に掘り下げていきましょうか。"
    ```
