"""ComfyUI HTTP 集成：/upload/image、/prompt、/history、/view。"""

from __future__ import annotations

import base64
import uuid
from typing import Any

from app.core.contracts.provider import ProviderConfig
from app.core.contracts.video_generation import VideoGenerationInput
from app.core.integrations.comfyui.video_payload import build_comfyui_graph, pick_first_frame


def _data_url_parts(value: str) -> tuple[str, bytes]:
    """data:image/png;base64,xxx 或纯 base64 -> (mime, bytes)。"""
    v = value.strip()
    mime = "image/png"
    if v.startswith("data:"):
        head, _, b64 = v.partition(",")
        mime = head.removeprefix("data:").partition(";")[0] or mime
    else:
        b64 = v
    return mime, base64.b64decode(b64)


class ComfyUIVideoApiAdapter:
    """ComfyUI：提交 graph 并按 prompt_id 轮询 /history。"""

    async def create_video(
        self,
        *,
        cfg: ProviderConfig,
        input_: VideoGenerationInput,
        timeout_s: float,
    ) -> str:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("httpx is required for video generation tasks") from e

        base_url = (cfg.base_url or "http://127.0.0.1:16080").rstrip("/")
        client_id = uuid.uuid4().hex
        first_frame_name: str | None = None

        raw_first = pick_first_frame(input_)
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            if raw_first:
                mime, blob = _data_url_parts(raw_first)
                ext = {"image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime, ".png")
                files = {"image": (f"jellyfish_first{ext}", blob, mime)}
                r = await client.post(f"{base_url}/upload/image", files=files, data={"overwrite": "true"})
                r.raise_for_status()
                first_frame_name = str(r.json().get("name") or "")
                if not first_frame_name:
                    raise RuntimeError(f"ComfyUI /upload/image missing name: {r.text!r}")

            seed = int(input_.seed) if input_.seed is not None and input_.seed >= 0 else uuid.uuid4().int % (2**32)
            graph = build_comfyui_graph(input_, first_frame_name=first_frame_name, seed=seed)
            r = await client.post(
                f"{base_url}/prompt",
                json={"prompt": graph, "client_id": client_id},
            )
            r.raise_for_status()
            data: dict[str, Any] = r.json()
            prompt_id = str(data.get("prompt_id") or "")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI /prompt missing prompt_id: {data!r}")
            return prompt_id

    async def get_video(
        self,
        *,
        cfg: ProviderConfig,
        video_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        """返回归一化状态：{status: running/completed/failed, video_url?, error?}。"""
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("httpx is required for video generation tasks") from e

        base_url = (cfg.base_url or "http://127.0.0.1:16080").rstrip("/")
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(f"{base_url}/history/{video_id}")
            r.raise_for_status()
            history: dict[str, Any] = r.json()

        entry = history.get(video_id)
        if not entry:
            return {"status": "running"}

        status_info = entry.get("status") or {}
        status_str = str(status_info.get("status_str") or "").lower()
        completed = bool(status_info.get("completed"))
        outputs = entry.get("outputs") or {}

        video_url: str | None = None
        for node_out in outputs.values():
            for key in ("videos", "images", "gifs"):
                for item in node_out.get(key) or []:
                    if not isinstance(item, dict) or not item.get("filename"):
                        continue
                    if str(item.get("type") or "output") not in ("output", "temp"):
                        continue
                    query = httpx.QueryParams(
                        {
                            "filename": item["filename"],
                            "subfolder": item.get("subfolder") or "",
                            "type": item.get("type") or "output",
                        }
                    )
                    video_url = f"{base_url}/view?{query}"
                    break
                if video_url:
                    break
            if video_url:
                break

        if status_str == "error" or (completed and not video_url):
            messages = (status_info.get("messages") or [])
            err = next((m for m in messages if str(m[0]).endswith("execution_error")), None)
            return {"status": "failed", "error": repr(err) or "no video output"}

        if video_url and completed:
            return {"status": "completed", "video_url": video_url}
        return {"status": "running"}
