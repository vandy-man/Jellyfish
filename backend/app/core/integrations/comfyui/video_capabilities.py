"""ComfyUI 视频能力声明与覆盖注册。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.integrations.video_capabilities import ALLOWED_RATIOS, VideoModelCapability

if TYPE_CHECKING:
    from app.core.contracts.video_generation import VideoGenerationInput

_COMFYUI_DEFAULT = VideoModelCapability(
    supports_seed=True,
    supports_watermark=False,
    allowed_ratios=set(ALLOWED_RATIOS),
    default_ratio="9:16",
    min_seconds=1,
    max_seconds=20,
)

# key: 模型前缀（小写）
_COMFYUI_MODEL_OVERRIDES: dict[str, VideoModelCapability] = {}


def register_comfyui_video_capability(*, model_prefix: str, capability: VideoModelCapability) -> None:
    prefix = model_prefix.strip().lower()
    if not prefix:
        raise ValueError("model_prefix must not be empty")
    _COMFYUI_MODEL_OVERRIDES[prefix] = capability


def clear_comfyui_video_capability_overrides() -> None:
    _COMFYUI_MODEL_OVERRIDES.clear()


def _pick_override(model: str | None) -> VideoModelCapability | None:
    if not model:
        return None
    value = model.strip().lower()
    if not value:
        return None
    # 最长前缀优先，避免通用前缀覆盖具体前缀。
    for prefix, cap in sorted(_COMFYUI_MODEL_OVERRIDES.items(), key=lambda item: len(item[0]), reverse=True):
        if value.startswith(prefix):
            return cap
    return None


def resolve_comfyui_video_capability(model: str | None) -> VideoModelCapability:
    return _pick_override(model) or _COMFYUI_DEFAULT


def validate_comfyui_video_options(input_: VideoGenerationInput) -> None:
    """ComfyUI 能力校验入口（避免调用侧传 provider 字面量）。"""
    from app.core.integrations.video_capabilities import validate_video_options

    validate_video_options(provider="comfyui", model=input_.model, input_=input_)
