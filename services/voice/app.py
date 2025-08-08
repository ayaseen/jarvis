from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import whisper
import torch
from TTS.api import TTS  # coqui-tts keeps this import path
import io
import numpy as np
import soundfile as sf
import os
import logging
import base64
import tempfile
import traceback
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS Voice Service", version="1.0.0")

# Caches (mount these as volumes if you want persistence)
WHISPER_CACHE = os.getenv("WHISPER_CACHE", "/models/whisper")
TTS_CACHE = os.getenv("TTS_CACHE", "/models/tts")

# Defaults can be overridden via env
DEFAULT_TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")


class VoiceEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Coqui cache dir (coqui-tts honors TTS_HOME)
        os.environ["TTS_HOME"] = TTS_CACHE

        self.whisper_model = None
        self.tts = None

    async def load_models(self):
        # Load Whisper (download cached under WHISPER_CACHE)
        try:
            self.whisper_model = whisper.load_model(
                DEFAULT_WHISPER_MODEL,
                device=self.device,
                download_root=WHISPER_CACHE,
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None

        # Load Coqui TTS (set device via .to(...))
        try:
            self.tts = TTS(model_name=DEFAULT_TTS_MODEL, progress_bar=False).to(self.device)
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            self.tts = None

    def _detect_tts_sample_rate(self) -> int:
        """
        Try to discover the output sample rate from the TTS model/config.
        Fall back to 22050 if not found.
        """
        for chain in [
            ("output_sample_rate",),
            ("synthesizer", "output_sample_rate"),
            ("synthesizer", "tts_config", "audio", "output_sample_rate"),
            ("synthesizer", "tts_model", "config", "audio", "output_sample_rate"),
            ("synthesizer", "tts_model", "config", "audio", "sample_rate"),
        ];
        :
            obj = self.tts
            try:
                for attr in chain:
                    obj = getattr(obj, attr)
                if isinstance(obj, (int, float)) and int(obj) > 0:
                    return int(obj)
            except Exception:
                continue
        return 22050

    async def transcribe(self, audio_file: UploadFile) -> str:
        if not self.whisper_model:
            raise ValueError("Whisper model not loaded")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            try:
                audio_bytes = await audio_file.read()
                tmp_file.write(audio_bytes)
                tmp_file.flush()

                result = self.whisper_model.transcribe(
                    tmp_file.name,
                    fp16=(self.device == "cuda"),
                    language="en",
                )
                return result.get("text", "").strip()
            except Exception:
                logger.error(f"Transcription error: {traceback.format_exc()}")
                raise
            finally:
                try:
                    os.unlink(tmp_file.name)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")

    async def synthesize(self, text: str) -> bytes:
        if not self.tts:
            raise ValueError("TTS model not loaded")

        try:
            wav = self.tts.tts(text=text)
            if not isinstance(wav, np.ndarray):
                wav = np.array(wav, dtype=np.float32)

            sr = self._detect_tts_sample_rate()
            buffer = io.BytesIO()
            sf.write(buffer, wav, sr, format="WAV")
            buffer.seek(0)
            return buffer.read()
        except Exception:
            logger.error(f"TTS error: {traceback.format_exc()}")
            raise


engine: Optional[VoiceEngine] = None


@app.on_event("startup")
async def startup_event():
    global engine
    engine = VoiceEngine()
    await engine.load_models()
    logger.info("Voice Engine initialization complete")


@app.get("/health")
async def health_check():
    ok_whisper = bool(engine and engine.whisper_model)
    ok_tts = bool(engine and engine.tts)
    status = {
        "status": "healthy" if ok_whisper and ok_tts else "degraded",
        "device": engine.device if engine else "unknown",
        "whisper_model_loaded": ok_whisper,
        "tts_model_loaded": ok_tts,
    }
    if status["status"] == "degraded":
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    return status


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    if not engine or not engine.whisper_model:
        raise HTTPException(status_code=503, detail="Whisper model not available")
    try:
        text = await engine.transcribe(audio)
        return {"text": text, "status": "success"}
    except Exception as e:
        logger.error(f"Transcribe endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SynthesizeRequest(BaseModel):
    text: str


@app.post("/synthesize")
async def synthesize_speech(req: SynthesizeRequest):
    if not engine or not engine.tts:
        raise HTTPException(status_code=503, detail="TTS model not available")
    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        audio_bytes = await engine.synthesize(req.text)
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )
    except Exception as e:
        logger.error(f"Synthesize endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/base64")
async def synthesize_speech_base64(req: SynthesizeRequest):
    if not engine or not engine.tts:
        raise HTTPException(status_code=503, detail="TTS model not available")
    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        audio_bytes = await engine.synthesize(req.text)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {"audio": audio_base64, "format": "wav"}
    except Exception as e:
        logger.error(f"Synthesize base64 endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

