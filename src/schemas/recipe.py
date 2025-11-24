from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator


# ==================== Segment Grouping Plan (Phase 1: fast grouping decision) ====================

class SegmentGroup(BaseModel):
    """A group of segment indices that should be merged into a single scene."""
    segment_indices: List[int] = Field(
        ...,
        min_length=1,
        description="List of segment indices (from segments.json) to merge into one scene. Indices are 1-based."
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of why these segments were grouped together."
    )

    model_config = ConfigDict(extra="forbid")


class SegmentGroupingPlan(BaseModel):
    """Lightweight plan indicating how to group segments into scenes."""
    groups: List[SegmentGroup] = Field(
        ...,
        min_length=1,
        description="List of segment groups. Each group becomes one scene in the recipe."
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_grouping(self):
        if not self.groups:
            raise ValueError("Groups list cannot be empty")

        # Collect all segment indices
        all_indices = []
        for group in self.groups:
            all_indices.extend(group.segment_indices)

        # Check no duplicates (each segment appears exactly once)
        if len(all_indices) != len(set(all_indices)):
            duplicates = [idx for idx in all_indices if all_indices.count(idx) > 1]
            raise ValueError(f"Segment indices appear in multiple groups: {set(duplicates)}")

        # Check sequential coverage (all indices from 1 to max should be present)
        sorted_indices = sorted(all_indices)
        expected_indices = list(range(1, max(sorted_indices) + 1))
        if sorted_indices != expected_indices:
            missing = set(expected_indices) - set(sorted_indices)
            extra = set(sorted_indices) - set(expected_indices)
            errors = []
            if missing:
                errors.append(f"missing indices: {sorted(missing)}")
            if extra:
                errors.append(f"extra indices: {sorted(extra)}")
            raise ValueError(f"Segment indices are not sequential: {', '.join(errors)}")

        return self


class AudioTrack(BaseModel):
    path: str = Field(..., description="Filesystem path to the narration audio file.")
    duration_sec: float = Field(..., ge=0.0, description="Total audio duration in seconds.")

    model_config = ConfigDict(extra="forbid")


class RenderPolicy(BaseModel):
    adapter: Literal["editly", "ffmpeg"] = Field(..., description="Render adapter to use for final assembly.")
    fps: int = Field(..., ge=1, description="Output frames per second for render.")
    pixel_format: str = Field(..., description="Video pixel format (e.g., yuv420p).")
    preset: str = Field(..., description="Encoder preset for FFmpeg/Editly.")
    audio_codec: str = Field(..., description="Audio codec identifier (e.g., aac).")
    audio_bitrate: str = Field(..., description="Audio bitrate (e.g., 192k).")
    fade_in_sec: float = Field(..., ge=0.0)
    fade_out_sec: float = Field(..., ge=0.0)
    zoom_in_speed: float = Field(..., description="Global zoom-in speed for Editly effects.")
    zoom_out_speed: float = Field(..., description="Global zoom-out speed for Editly effects.")
    mode: Literal["single_pass", "per_scene"]

    model_config = ConfigDict(extra="forbid")


class SelectorIntentWeights(BaseModel):
    clip_sim: float = Field(..., description="Weight for CLIP similarity score.")
    resolution_score: float = Field(..., description="Weight for asset resolution contribution.")
    sharpness: float = Field(..., description="Weight for sharpness heuristics.")
    ocr_penalty: float = Field(..., description="Penalty weight for OCR text presence.")
    face_bonus: float = Field(..., description="Bonus weight when faces are detected.")
    abstract_penalty: float = Field(..., description="Penalty weight for abstract/synthetic patterns.")


class SelectorWeights(BaseModel):
    factual: SelectorIntentWeights = Field(..., description="Weights applied when scene intent is factual.")
    conceptual: SelectorIntentWeights = Field(..., description="Weights applied when scene intent is conceptual.")


class SelectorThresholds(BaseModel):
    min_clip_sim_factual: float = Field(..., description="Minimum CLIP similarity for factual intents.")
    min_clip_sim_conceptual: float = Field(..., description="Minimum CLIP similarity for conceptual intents.")
    min_resolution: List[int] = Field(
        ..., min_length=2, max_length=2, description="Minimum [width,height] allowed for assets."
    )
    overlay_limit: float = Field(..., ge=0.0, le=1.0,
                                 description="Maximum allowed overlay delta for OCR-like detections.")


class SelectorLimits(BaseModel):
    max_ocr_chars: int = Field(..., ge=0, description="Maximum OCR characters allowed before rejection.")
    min_sharpness: float = Field(..., description="Minimum sharpness score threshold.")
    relax_clip_margin: float = Field(
        default=0.05,
        ge=0.0,
        description="Maximum CLIP similarity relaxation applied when no stock asset passes the strict gate.",
    )


class SelectorPolicy(BaseModel):
    weights: SelectorWeights
    thresholds: SelectorThresholds
    limits: SelectorLimits

    model_config = ConfigDict(extra="forbid")


class SearchPolicy(BaseModel):
    deny_terms: List[str] = Field(default_factory=list, description="Terms to exclude from stock searches.")
    whitelist_domains: List[str] = Field(default_factory=list, description="Preferred domains for factual imagery.")

    model_config = ConfigDict(extra="forbid")


class AssetRules(BaseModel):
    clip_threshold: float = Field(..., description="Minimum CLIP similarity required for the selected asset.")
    overlay_limit: float = Field(..., ge=0.0, le=1.0, description="Maximum overlay/OCR ratio permitted for the asset.")
    min_resolution: List[int] = Field(
        ..., min_length=2, max_length=2, description="Minimum [width,height] accepted for the asset."
    )
    search_strategy: List[str] = Field(
        ..., min_length=1, description="Ordered stock providers to consult (e.g., ['google','pexels','pixabay'])."
    )

    model_config = ConfigDict(extra="forbid")


class Asset(BaseModel):
    type: Literal["image", "video"] = Field(..., description="Asset media type expected by the executor.")
    search_queries: List[str] = Field(
        ...,
        min_length=3,
        description=(
            "Ordered English stock queries (≥3) ranked by preference, each concise (≤3 words) and mutually distinct "
            "to maximize coverage across providers."
        ),
    )

    semantic_text: str = Field(
        ..., description="Primary English semantic description (8–12 words: subject + action + setting)."
    )
    semantic_text_variants: List[str] = Field(
        ..., description="2–4 short English variants (8–12 words) covering angle/action/context."
    )
    negative_semantic_texts: List[str] = Field(
        ...,
        description="3–6 negative descriptors to avoid (e.g., movie poster text, vector illustration, CGI render, watermark)."
    )
    grounded_entities: Optional[List[str]] = Field(
        default=None,
        description="Optional list of proper nouns/real entities for factual assets (e.g., Mariana Trench, NOAA).",
    )

    generate_prompt: str = Field(
        ...,
        min_length=100,
        description=(
            "Narrative visual brief (≥100 chars) detailing subject, composition, lighting and mood for image/keyframe generation. "
            "Avoid technical directives (resolution, fps, bitrate, codec, duration, aspect ratio). "
            "If on-image text is required, include it between quotes."
        ),
    )
    video_generate_prompt: Optional[str] = Field(
        default=None,
        min_length=100,
        description=(
            "For assets type='video', a dedicated (≥100 chars) motion brief describing pacing, camera feel and actions for video generation. "
            "Keep it distinct from the image prompt and avoid technical parameters."
        ),
    )
    source_preference: Literal["stock", "generated"] = Field(..., description="Preferred acquisition mode.")
    duration_hint_sec: float = Field(..., ge=0.0, description="Target duration in seconds for this asset.")
    transition: str = Field(..., description="Transition identifier between scene clips.")
    zoomDirection: Optional[str] = Field(
        default=None,
        description="Ken Burns direction for image assets ('in', 'out', 'left', 'right').",
    )
    zoomAmount: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Ken Burns zoom amount (0.05=slow, 0.1=default, 0.2=rápido).",
    )
    rules: AssetRules = Field(..., description="Per-asset policy overrides, including search strategy and thresholds.")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _post_validate(self) -> "Asset":
        self._ensure_video_prompts()
        self._dedupe_negatives()
        return self

    def _normalize_search_queries(self) -> None:
        normalized: List[str] = []
        seen = set()
        for raw in self.search_queries:
            text = " ".join((raw or "").strip().split())
            if not text:
                continue
            words = text.split()
            if len(words) > 5:
                raise ValueError("Each search query must use at most 5 words.")
            lower = text.lower()
            if lower in seen:
                continue
            normalized.append(text)
            seen.add(lower)
        if len(normalized) < 3:
            raise ValueError("Provide at least three distinct search queries per asset.")
        self.search_queries = normalized

    def _ensure_video_prompts(self) -> None:
        if self.type == "video" and not self.video_generate_prompt:
            raise ValueError("video_generate_prompt is required for assets of type 'video'.")

    def _validate_semantic_texts(self) -> None:

        if not self.semantic_text or not isinstance(self.semantic_text, str):
            raise ValueError("semantic_text is required and must be a non-empty string (EN, 8–12 words).")
        if not self.semantic_text_variants or not isinstance(self.semantic_text_variants, list):
            raise ValueError("semantic_text_variants is required (provide 2–4 short variants in EN).")
        if not self.negative_semantic_texts or not isinstance(self.negative_semantic_texts, list):
            raise ValueError("negative_semantic_texts is required (provide 3–6 negative descriptors in EN).")

        def _word_count_ok(s: str) -> bool:
            ws = [w for w in (s or "").strip().split() if w]
            return 8 <= len(ws) <= 12

        self.semantic_text = " ".join(self.semantic_text.strip().split())
        if not _word_count_ok(self.semantic_text):
            raise ValueError("semantic_text must contain 8–12 words.")

        cleaned_variants: List[str] = []
        for v in self.semantic_text_variants:
            vv = " ".join((v or "").strip().split())
            if not vv:
                continue
            if not _word_count_ok(vv):
                raise ValueError("each semantic_text_variants item must contain 8–12 words.")
            cleaned_variants.append(vv)
        if len(cleaned_variants) < 2:
            raise ValueError("provide at least 2 semantic_text_variants items (8–12 words each).")
        if len(cleaned_variants) > 4:
            cleaned_variants = cleaned_variants[:4]
        self.semantic_text_variants = cleaned_variants

    def _dedupe_negatives(self) -> None:

        seen = set()
        uniq: List[str] = []
        for raw in self.negative_semantic_texts:
            s = " ".join((raw or "").strip().split())
            if not s:
                continue
            low = s.lower()
            if low in seen:
                continue
            uniq.append(s)
            seen.add(low)
        if len(uniq) < 3:
            raise ValueError("provide at least 3 negative_semantic_texts (distinct, case-insensitive).")
        self.negative_semantic_texts = uniq[:6]

    @property
    def primary_search_query(self) -> str:
        return self.search_queries[0]

    @property
    def search_query(self) -> str:
        return self.primary_search_query


class SceneBlueprint(BaseModel):
    scene_index: int
    segment_index: int
    scene_role: Literal["hook", "build", "climax", "payoff", "outro"]
    visual_mode: Literal["dynamic", "static", "narrative"]
    intent: Literal["factual", "conceptual"]
    emotion: str
    motion_style: str
    color_mood: str
    focus_object: Optional[str] = None
    environment: Optional[str] = None
    asset_count: int
    impact_level: int
    overlay_text: Optional[str] = None


class Scene(BaseModel):
    index: int = Field(..., ge=0, description="Scene index (0-based).")
    start_time: float = Field(..., ge=0.0, description="Scene start timestamp in seconds.")
    end_time: float = Field(..., ge=0.0, description="Scene end timestamp in seconds.")
    text: str = Field(..., description="Narrative text for the scene (pt-BR).")
    overlay_text: Optional[str] = Field(default=None, description="Optional overlay text to render on screen.")
    visual_mode: Literal["dynamic", "static", "narrative"] = Field(..., description="High-level visual pacing.")
    intent: Literal["factual", "conceptual"] = Field(..., description="Guides stock vs generation decisions.")
    transition: str = Field(..., description="Transition identifier between scene clips.")
    assets: List[Asset] = Field(default_factory=list, description="Ordered list of assets composing the scene.")


class Policy(BaseModel):
    render: RenderPolicy = Field(..., description="Global render configuration (adapter, fps, etc.).")
    selector_policy: SelectorPolicy = Field(..., description="Weights/thresholds used by the smart asset picker.")
    search_policy: SearchPolicy = Field(..., description="Search-related policies such as deny lists and whitelists.")

    model_config = ConfigDict(extra="forbid")


class VideoRecipe(BaseModel):
    title: str = Field(..., description="Final video title in pt-BR.")
    description: str = Field(..., description="Video description in pt-BR.")
    tags: List[str] = Field(default_factory=list, description="List of SEO tags (pt-BR).")
    language: str = Field(..., description="Language code for the narrative (e.g., pt-BR).")
    audio: AudioTrack
    policy: Policy
    scenes: List[Scene] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class VideoRecipeMetadata(BaseModel):
    title: str
    description: str
    tags: List[str] = Field(default_factory=list)
    language: str
    audio: AudioTrack
    policy: Policy
    background_music_prompt: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class VideoRecipeSceneChunk(BaseModel):
    scenes: List[Scene] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class NarrativePlan(BaseModel):
    narration_text: str = Field(..., description="Single string with the full narration text (PT-BR).")

    model_config = ConfigDict(extra="forbid")


class MusicPlan(BaseModel):
    """Simple container for ElevenLabs Music prompt generation."""
    prompt: str

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "AudioTrack",
    "RenderPolicy",
    "SelectorIntentWeights",
    "SelectorWeights",
    "SelectorThresholds",
    "SelectorLimits",
    "SelectorPolicy",
    "SearchPolicy",
    "AssetRules",
    "Asset",
    "SceneBlueprint",
    "Scene",
    "Policy",
    "VideoRecipe",
    "VideoRecipeMetadata",
    "VideoRecipeSceneChunk",
    "NarrativePlan",
    "MusicPlan",
]
