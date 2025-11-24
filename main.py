import os

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("DISABLE_TRANSFORMERS_AV_IMPORTS", "1")

import re
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import logfire
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from src.integrations.llm_client import TypedLLMClient
from src.pipeline.pipeline import LoggingObserver, PipelineBuilder, PipelineContext
from src.pipeline.video_profiles import apply_profile_env, resolve_video_profile
from src.pipeline.steps import (
    BuildSegmentsStep,
    ExecuteRecipeStep,
    GenerateNarrativeStep,
    PersistMetadataStep,
    RequestRecipeStep,
    SynthesizeNarrationStep,
    ComposeBackgroundMusicStep,
    BurnInCaptionsStep,
)

logfire.configure()
logfire.instrument_openai()
logfire.instrument_anthropic()

load_dotenv()

ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
VERBOSE = os.getenv("VERBOSE", "1") == "1"

warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

if not ELEVEN_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY not configured.")

LLM = TypedLLMClient()
ELEVEN_CLIENT = ElevenLabs(api_key=ELEVEN_KEY)

_build_log_path: Optional[str] = None


def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def flog(msg: str) -> None:
    if VERBOSE:
        print(msg)
    if _build_log_path:
        try:
            with open(_build_log_path, "a", encoding="utf-8") as handle:
                handle.write(msg + "\n")
        except Exception:
            pass


def load_voice_settings() -> Dict[str, Union[float, bool]]:
    presets: Dict[str, Dict[str, Union[float, bool]]] = {
        "baseline": {"stability": 1.0, "similarity_boost": 0.85, "style": 0.1, "use_speaker_boost": True, "speed": 1.0},
        "consistent": {"stability": 1.0, "similarity_boost": 0.9, "style": 0.05, "use_speaker_boost": True,
                       "speed": 1.0},
        "performance": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.35, "use_speaker_boost": True,
                        "speed": 1.0},
    }
    preset_key = os.getenv("ELEVENLABS_VOICE_PRESET", "baseline").strip().lower()
    if preset_key not in presets:
        warnings.warn(
            f"Unknown ELEVENLABS_VOICE_PRESET '{preset_key}'. Falling back to 'baseline'.",
            stacklevel=2,
        )
    settings = presets.get(preset_key, presets["baseline"]).copy()

    overrides = {
        "stability": os.getenv("ELEVENLABS_STABILITY"),
        "similarity_boost": os.getenv("ELEVENLABS_SIMILARITY_BOOST"),
        "style": os.getenv("ELEVENLABS_STYLE"),
        "speed": os.getenv("ELEVENLABS_SPEED"),
    }
    for key, value in overrides.items():
        if value is None:
            continue
        try:
            settings[key] = float(value)
        except ValueError:
            warnings.warn(
                f"Invalid value '{value}' for ELEVENLABS_{key.upper()}; using preset default.",
                stacklevel=2,
            )
    use_speaker_raw = os.getenv("ELEVENLABS_USE_SPEAKER_BOOST")
    if use_speaker_raw is not None:
        normalized = use_speaker_raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            settings["use_speaker_boost"] = True
        elif normalized in {"0", "false", "no", "off"}:
            settings["use_speaker_boost"] = False
        else:
            warnings.warn(
                f"Invalid value '{use_speaker_raw}' for ELEVENLABS_USE_SPEAKER_BOOST; using preset default.",
                stacklevel=2,
            )
    return settings


def sanitize_voice_settings(settings: Dict[str, Union[float, bool]]) -> Dict[str, Union[float, bool]]:
    sanitized: Dict[str, Union[float, bool]] = settings.copy()

    def clamp_unit(name: str) -> None:
        value = sanitized.get(name)
        if value is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            warnings.warn(
                f"Invalid value '{value}' for ElevenLabs {name}; using default.",
                stacklevel=2,
            )
            sanitized.pop(name, None)
            return
        if numeric < 0.0 or numeric > 1.0:
            clamped = min(max(numeric, 0.0), 1.0)
            warnings.warn(
                f"ElevenLabs {name} {numeric} out of range [0, 1]; clamping to {clamped}.",
                stacklevel=2,
            )
            numeric = clamped
        sanitized[name] = numeric

    for field in ("stability", "similarity_boost", "style"):
        clamp_unit(field)

    speed = sanitized.get("speed")
    if speed is not None:
        try:
            speed_value = float(speed)
        except (TypeError, ValueError):
            warnings.warn(
                f"Invalid value '{speed}' for ElevenLabs speed; using 1.0.",
                stacklevel=2,
            )
            speed_value = 1.0
        if speed_value <= 0.0:
            warnings.warn(
                f"ElevenLabs speed {speed_value} must be > 0; defaulting to 1.0.",
                stacklevel=2,
            )
            speed_value = 1.0
        sanitized["speed"] = speed_value
    else:
        sanitized["speed"] = 1.0

    if "use_speaker_boost" in sanitized:
        sanitized["use_speaker_boost"] = bool(sanitized["use_speaker_boost"])
    else:
        sanitized["use_speaker_boost"] = True

    return sanitized


VOICE_SETTINGS = sanitize_voice_settings(load_voice_settings())


def init_build_log(base_dir: str) -> None:
    global _build_log_path
    _build_log_path = os.path.join(base_dir, "build.log")
    Path(_build_log_path).write_text("=== Build Log ===\n", encoding="utf-8")


def safe_name(value: str) -> str:
    return re.sub(r"[^\w\-\.]+", "_", value.strip())


def find_latest_run_dir(topic: str) -> Optional[str]:
    prefix = f"out_{safe_name(topic)}_"
    candidates = []
    for entry in os.listdir("."):
        if entry.startswith(prefix):
            full = os.path.join(".", entry)
            if os.path.isdir(full):
                candidates.append((os.path.getmtime(full), full))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def build_video(topic: str, reuse_existing_run: bool = True) -> str:
    base = find_latest_run_dir(topic) if reuse_existing_run else None
    if base and reuse_existing_run:
        log(f"♻️ Reusing artifacts from {base}")
    else:
        base = f"./out_{safe_name(topic)}_{int(time.time())}"
        Path(base).mkdir(parents=True, exist_ok=True)

    init_build_log(base)
    flog(f"Run directory: {base}")

    profile = resolve_video_profile(os.getenv("VIDEO_FORMAT"))
    apply_profile_env(profile)

    context = PipelineContext(
        topic=topic,
        base_dir=base,
        reuse_existing_run=reuse_existing_run,
        observer=LoggingObserver(flog),
    )
    context.set("logger", flog)
    context.set("video_profile", profile)

    builder = PipelineBuilder()
    pipeline = (
        builder
        .add_step(GenerateNarrativeStep(LLM))
        .add_step(SynthesizeNarrationStep(ELEVEN_CLIENT, VOICE_ID, voice_settings=VOICE_SETTINGS))
        .add_step(BuildSegmentsStep())
        .add_step(RequestRecipeStep(LLM))
        .add_step(ComposeBackgroundMusicStep(ELEVEN_CLIENT, LLM))
        .add_step(ExecuteRecipeStep())
        .add_step(BurnInCaptionsStep())
        .add_step(PersistMetadataStep())
        .build()
    )

    pipeline.execute(context)

    result = context.get("executor_result")
    if not result:
        raise RuntimeError("Pipeline finished without executor result.")

    log(f"✅ Final video ready: {result.final_video_path}")
    return result.final_video_path


def main() -> None:
    topic = os.getenv("TOPIC",
                      "Por que sentimos, cientificamente, aquele 'arrepio' ou intuição de que algo ruim está prestes a acontecer?")
    build_video(topic)


if __name__ == "__main__":
    main()
