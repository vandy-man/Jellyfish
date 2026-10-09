"""ComfyUI /prompt 工作流构建：MiniMax H3 单镜 T2V/I2V（4-6 步 turbo 配方，对应 h3_chain.py 节点图）。"""

from __future__ import annotations

from typing import Any

from app.core.contracts.video_generation import VideoGenerationInput, _strip_optional_b64

# H3 固定权重（与本机 ComfyUI models 对应；升级模型时同步改这里）
MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
LORA = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
TE = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
V_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
A_VAE = "minimax_h3_audio_vae_fp32.safetensors"

STEPS = 6  # turbo LoRA 4-6 步配方；sol-attention 提速
SAMPLER = "euler"
FPS = 24

# ratio -> (width, height)，取 8 的倍数、面积贴近 0.88MP（768x1152 基准）
RATIO_SIZES: dict[str, tuple[int, int]] = {
    "9:16": (768, 1152),
    "3:4": (864, 1152),
    "1:1": (960, 960),
    "4:3": (1152, 864),
    "16:9": (1152, 768),
    "21:9": (1344, 576),
}


def ratio_to_size(ratio: str) -> tuple[int, int]:
    return RATIO_SIZES.get(ratio, RATIO_SIZES["9:16"])


def pick_first_frame(input_: VideoGenerationInput) -> str | None:
    """I2V 首帧；优先级：first > key > last。"""
    for raw in (
        _strip_optional_b64(input_.first_frame_base64),
        _strip_optional_b64(input_.key_frame_base64),
        _strip_optional_b64(input_.last_frame_base64),
    ):
        if raw:
            return raw
    return None


def build_comfyui_graph(
    input_: VideoGenerationInput,
    *,
    first_frame_name: str | None,
    seed: int,
) -> dict[str, Any]:
    """构建 /prompt 的 API format graph（与 h3_chain.py build_prompt 同构）。"""
    width, height = ratio_to_size(input_.ratio)
    length = max(FPS, int(input_.seconds or 5) * FPS)
    prompt = input_.prompt or ""

    graph: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": LORA, "strength_model": 1.0, "model": ["1", 0]}},
        "3": {"class_type": "MiniMaxH3ScheduledSolAttentionPatch", "inputs": {
            "model": ["2", 0], "enabled": True, "tau_start": 1.3, "tau_end": 0.8, "curve": "linear",
            "min_tokens": 4096, "strict": False, "dense_percent": 0.0, "thresh_type": "diag",
            "int8_qk": False, "int8_pv": False, "sink_conditioning": "exact_kv", "dense_blocks": ""}},
        "4": {"class_type": "MiniMaxH3FusedModulation", "inputs": {"model": ["3", 0], "enabled": True}},
        "5": {"class_type": "MiniMaxH3ChunkFeedForward", "inputs": {"model": ["4", 0], "enabled": True, "chunks": 2, "min_tokens": 8192}},
        "6": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE, "type": "minimax", "device": "default"}},
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": V_VAE}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": A_VAE}},
        "10": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["6", 0], "vae": ["7", 0], "prompt": prompt,
            "width": width, "height": height, "length": length,
            "first_frame": ["20", 0] if first_frame_name else None,
            "last_frame": None}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed, "control": "fixed"}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}},
        "13": {"class_type": "BasicScheduler", "inputs": {"scheduler": "beta", "steps": STEPS, "denoise": 1.0, "model": ["5", 0]}},
        "14": {"class_type": "BasicGuider", "inputs": {"model": ["5", 0], "conditioning": ["10", 0]}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["11", 0], "guider": ["14", 0], "sampler": ["12", 0], "sigmas": ["13", 0], "latent_image": ["10", 1]}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["7", 0]}},
        "17": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["15", 0], "vae": ["8", 0]}},
        "18": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "audio": ["17", 0], "fps": FPS, "loop": 8, "colorspace": "sRGB"}},
        "19": {"class_type": "SaveVideo", "inputs": {"video": ["18", 0], "filename_prefix": "jellyfish/shot", "format": "auto", "codec": "auto", "bitrate": "auto"}},
    }
    if first_frame_name:
        graph["20"] = {"class_type": "LoadImage", "inputs": {"image": first_frame_name}}
    else:
        graph["10"]["inputs"]["first_frame"] = None
    return graph
