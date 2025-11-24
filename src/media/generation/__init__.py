from .seedance import generate_video_with_seedance
from .runpod_flux import RunPodFluxClient, flux_client

# Backwards compatibility: qwen generator is deprecated in favor of RunPod Flux.
try:
    from .qwen import generate_image_with_qwen  # type: ignore
except Exception:  # pragma: no cover
    generate_image_with_qwen = None  # type: ignore

__all__ = [
    "RunPodFluxClient",
    "flux_client",
    "generate_image_with_qwen",
    "generate_video_with_seedance",
]
