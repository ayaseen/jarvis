# services/voice/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import whisper
import torch
from TTS.api import TTS
import io
import numpy as np
import soundfile as sf
import os
import logging
import base64
import tempfile
import traceback
from typing import Optional
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS Voice Service", version="1.0.0")

# Caches (mount these as volumes if you want persistence)
WHISPER_CACHE = os.getenv("WHISPER_CACHE", "/models/whisper")
TTS_CACHE = os.getenv("TTS_CACHE", "/models/tts")

# Defaults can be overridden via env
DEFAULT_TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
DEFAULT_WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")  # Changed from base to tiny for lower memory

# Force CPU mode if GPU memory is low
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"


class VoiceEngine:
    def __init__(self):
        # Check GPU availability and memory
        self.device = self._select_device()
        logger.info(f"Using device: {self.device}")

        # Coqui cache dir (coqui-tts honors TTS_HOME)
        os.environ["TTS_HOME"] = TTS_CACHE

        self.whisper_model = None
        self.tts = None

    def _select_device(self):
        """Select device based on GPU availability and memory"""
        if FORCE_CPU:
            logger.info("Forced CPU mode via environment variable")
            return "cpu"
        
        if not torch.cuda.is_available():
            logger.info("CUDA not available, using CPU")
            return "cpu"
        
        try:
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            free = gpu_memory - allocated
            
            logger.info(f"GPU Memory - Total: {gpu_memory:.2f}GB, Allocated: {allocated:.2f}GB, Free: {free:.2f}GB")
            
            # If less than 2GB free, use CPU to avoid OOM
            if free < 2.0:
                logger.warning("Less than 2GB GPU memory free, using CPU to avoid OOM")
                return "cpu"
            
            return "cuda"
        except Exception as e:
            logger.error(f"Error checking GPU memory: {e}")
            return "cpu"

    async def load_models(self):
        # Load Whisper with error handling and memory optimization
        try:
            # Clear any existing GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load smaller Whisper model
            self.whisper_model = whisper.load_model(
                DEFAULT_WHISPER_MODEL,
                device=self.device,
                download_root=WHISPER_CACHE,
            )
            logger.info(f"Whisper model ({DEFAULT_WHISPER_MODEL}) loaded successfully on {self.device}")
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU OOM loading Whisper, falling back to CPU")
            self.device = "cpu"
            try:
                self.whisper_model = whisper.load_model(
                    DEFAULT_WHISPER_MODEL,
                    device="cpu",
                    download_root=WHISPER_CACHE,
                )
                logger.info("Whisper model loaded on CPU")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self.whisper_model = None
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None

        # Load Coqui TTS with error handling
        try:
            # Use a lighter TTS model for lower memory consumption
            light_tts_model = "tts_models/en/ljspeech/tacotron2-DDC"
            
            if torch.cuda.is_available() and self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            self.tts = TTS(model_name=light_tts_model, progress_bar=False)
            
            # Try GPU first, fallback to CPU if OOM
            if self.device == "cuda":
                try:
                    self.tts = self.tts.to(self.device)
                    logger.info(f"TTS model loaded successfully on {self.device}")
                except torch.cuda.OutOfMemoryError:
                    logger.warning("GPU OOM for TTS, using CPU")
                    self.device = "cpu"
                    self.tts = TTS(model_name=light_tts_model, progress_bar=False).to("cpu")
                    logger.info("TTS model loaded on CPU")
            else:
                self.tts = self.tts.to("cpu")
                logger.info("TTS model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            logger.error(traceback.format_exc())
            self.tts = None

    def _detect_tts_sample_rate(self) -> int:
        """
        Try to discover the output sample rate from the TTS model/config.
        Fall back to 22050 if not found.
        """
        if not self.tts:
            return 22050
            
        chains = [
            ("output_sample_rate",),
            ("synthesizer", "output_sample_rate"),
            ("synthesizer", "tts_config", "audio", "output_sample_rate"),
            ("synthesizer", "tts_model", "config", "audio", "output_sample_rate"),
            ("synthesizer", "tts_model", "config", "audio", "sample_rate"),
        ]

        for chain in chains:
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

                # Transcribe with lower memory settings
                result = self.whisper_model.transcribe(
                    tmp_file.name,
                    fp16=False,  # Disable FP16 to avoid issues on CPU
                    language="en",
                    beam_size=1,  # Reduce beam size for lower memory
                    best_of=1,  # Reduce candidates for lower memory
                )
                return result.get("text", "").strip()
            except Exception as e:
                logger.error(f"Transcription error: {traceback.format_exc()}")
                raise
            finally:
                try:
                    os.unlink(tmp_file.name)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")
                    
                # Clear memory after transcription
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    async def synthesize(self, text: str) -> bytes:
        if not self.tts:
            raise ValueError("TTS model not loaded")

        try:
            # Generate speech
            wav = self.tts.tts(text=text)
            
            if not isinstance(wav, np.ndarray):
                wav = np.array(wav, dtype=np.float32)

            sr = self._detect_tts_sample_rate()
            buffer = io.BytesIO()
            sf.write(buffer, wav, sr, format="WAV")
            buffer.seek(0)
            
            # Clear memory after synthesis
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return buffer.read()
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU OOM during synthesis, please restart with FORCE_CPU=true")
            raise HTTPException(status_code=503, detail="GPU memory exhausted, please use CPU mode")
        except Exception as e:
            logger.error(f"TTS error: {traceback.format_exc()}")
            raise


engine: Optional[VoiceEngine] = None


@app.on_event("startup")
async def startup_event():
    global engine
    engine = VoiceEngine()
    await engine.load_models()
    logger.info("Voice Engine initialization complete")


@app.on_event("shutdown")
async def shutdown_event():
    global engine
    if engine:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        engine = None
    logger.info("Voice Engine shutdown complete")


@app.get("/health")
async def health_check():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
        
    ok_whisper = bool(engine and engine.whisper_model)
    ok_tts = bool(engine and engine.tts)
    
    # More lenient health check - service is "degraded" but still functional if one model loads
    if ok_whisper or ok_tts:
        status = "degraded" if not (ok_whisper and ok_tts) else "healthy"
        return {
            "status": status,
            "device": engine.device if engine else "unknown",
            "whisper_model_loaded": ok_whisper,
            "tts_model_loaded": ok_tts,
            "whisper_model": DEFAULT_WHISPER_MODEL if ok_whisper else None,
            "tts_model": DEFAULT_TTS_MODEL if ok_tts else None,
            "memory_mode": "CPU" if engine.device == "cpu" else "GPU"
        }
    else:
        raise HTTPException(status_code=503, detail="No models loaded")


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


@app.get("/")
async def root():
    return {
        "service": "JARVIS Voice Service",
        "version": "1.0.0",
        "status": "online",
        "device": engine.device if engine else "unknown",
        "models": {
            "whisper": DEFAULT_WHISPER_MODEL,
            "tts": DEFAULT_TTS_MODEL
        }
    }
