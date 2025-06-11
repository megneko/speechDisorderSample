from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import whisper
import tempfile
import shutil
import os

app = FastAPI()

# Whisperモデルをロード（"base" で軽量。必要に応じて tiny/base/small/medium/large）
model = whisper.load_model("base")


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # Whisperで文字起こし
        result = model.transcribe(temp_file_path)
        return JSONResponse(content={"text": result["text"]})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # 一時ファイル削除
        os.remove(temp_file_path)
