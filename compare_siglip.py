from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import torch
import imageio
from PIL import Image, ImageOps
from diffsynth import VideoData, save_video, load_state_dict
from diffsynth.pipelines.wan_video_semantic import ModelConfig, WanVideoPipeline
from examples.wanvideo.model_training.custom_utils import find_closest_resolution, get_all_resolution
from transformers import AutoProcessor, AutoModel

DEFAULT_MODEL_ROOT = ""

DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")

# ---------- video helpers ----------

def detect_video_resolution(video_path: str, fallback: tuple[int, int]) -> tuple[int, int]:
    width = height = None
    try:
        reader = imageio.get_reader(video_path)
        try:
            meta = reader.get_meta_data()
            size = meta.get("source_size") or meta.get("size")
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                width, height = int(size[0]), int(size[1])
        except Exception:
            pass
        finally:
            try:
                reader.close()
            except Exception:
                pass
    except Exception:
        pass
    if width and height:
        return width, height
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    except Exception:
        pass
    if width and height:
        return width, height
    return fallback


def detect_video_fps(video_path: str) -> Optional[float]:
    fps = None
    try:
        reader = imageio.get_reader(video_path)
        try:
            meta = reader.get_meta_data()
            fps = meta.get("fps") or meta.get("framerate")
        except Exception:
            pass
        finally:
            try:
                reader.close()
            except Exception:
                pass
    except Exception:
        pass
    if fps and fps > 0:
        return float(fps)
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
        cap.release()
    except Exception:
        pass
    if fps and fps > 0:
        return float(fps)
    return None


def compute_letterbox_params(
    orig_w: int, orig_h: int, target_w: int, target_h: int
) -> tuple[int, int, int, int, int, int]:
    if orig_w <= 0 or orig_h <= 0:
        return target_w, target_h, 0, 0, 0, 0
    orig_ratio = orig_w / orig_h
    target_ratio = target_w / target_h
    if abs(orig_ratio - target_ratio) < 1e-3:
        return target_w, target_h, 0, 0, 0, 0
    if orig_ratio < target_ratio:
        resize_h = target_h
        resize_w = max(1, int(round(resize_h * orig_ratio)))
        pad_total = max(0, target_w - resize_w)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_bottom = 0
    else:
        resize_w = target_w
        resize_h = max(1, int(round(resize_w / orig_ratio)))
        pad_total = max(0, target_h - resize_h)
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        pad_left = pad_right = 0
    return resize_w, resize_h, pad_left, pad_right, pad_top, pad_bottom


def _pad_numpy_array(arr: np.ndarray, pad_left: int, pad_right: int, pad_top: int, pad_bottom: int) -> np.ndarray:
    if arr.ndim == 2:
        pad_widths = ((pad_top, pad_bottom), (pad_left, pad_right))
    elif arr.ndim == 3:
        if arr.shape[0] <= 4:
            pad_widths = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        else:
            pad_widths = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        raise ValueError(f"Unsupported array shape for padding: {arr.shape}")
    return np.pad(arr, pad_widths, mode="constant", constant_values=0)


def pad_frame(frame, pad_left: int, pad_right: int, pad_top: int, pad_bottom: int):
    if not (pad_left or pad_right or pad_top or pad_bottom):
        return frame
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
        padded = _pad_numpy_array(arr, pad_left, pad_right, pad_top, pad_bottom)
        return torch.from_numpy(padded).to(frame.dtype).to(frame.device)
    if isinstance(frame, Image.Image):
        return ImageOps.expand(frame, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)
    if isinstance(frame, np.ndarray):
        return _pad_numpy_array(frame, pad_left, pad_right, pad_top, pad_bottom)
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def load_clip_frames(video_path: str, height: int, width: int, max_frames: Optional[int], resolutions=None) -> List[torch.Tensor]:
    if resolutions is None:
        orig_w, orig_h = detect_video_resolution(video_path, fallback=(width, height))
        resize_w, resize_h, pad_l, pad_r, pad_t, pad_b = compute_letterbox_params(orig_w, orig_h, width, height)
        reader = VideoData(video_file=video_path, height=resize_h, width=resize_w)
        total = len(reader)
        limit = min(total, max_frames) if max_frames is not None else total
        frames: List[torch.Tensor] = []
        for i in range(limit):
            frame = reader[i]
            frame = pad_frame(frame, pad_l, pad_r, pad_t, pad_b)
            frames.append(frame)
        return frames
    else:
        reader = VideoData(video_file=video_path)
        ratios = [res[0] / res[1] for res in resolutions]
        old_height, old_width = reader.shape()
        new_width, new_height = find_closest_resolution(old_width, old_height, ratios, resolutions)
        reader.set_shape(new_height, new_width)
        total = len(reader)
        limit = min(total, max_frames) if max_frames is not None else total
        frames: List[torch.Tensor] = []
        for i in range(limit):
            frames.append(reader[i])
        w, h = reader[0].size
        print(f"[Info] Resized: {old_height}x{old_width} -> {new_height}x{new_width}, actual: {h}x{w}", flush=True)
        return frames


def ensure_4k_plus_1_frames(frames: List[torch.Tensor]) -> tuple[List[torch.Tensor], bool]:
    """Pad frames until len % 4 == 1 (Wan requirement)."""
    if not frames:
        return frames, False
    remainder = len(frames) % 4
    if remainder == 1:
        return frames, False
    pad_count = (1 - remainder) % 4
    if pad_count <= 0:
        return frames, False
    last = frames[-1]
    for _ in range(pad_count):
        try:
            frames.append(last.clone())
        except AttributeError:
            frames.append(last)
    return frames, True

def pad_to_square_and_resize(frame: torch.Tensor, target_size: int) -> torch.Tensor:
    """Pad frame to square and resize to target_size."""
    c, h, w = frame.shape
    if h == w:
        return torch.nn.functional.interpolate(frame.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False).squeeze(0)
    max_side = max(h, w)
    pad_left = (max_side - w) // 2
    pad_right = max_side - w - pad_left
    pad_top = (max_side - h) // 2
    pad_bottom = max_side - h - pad_top
    padded = torch.nn.functional.pad(frame, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
    resized = torch.nn.functional.interpolate(padded.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False).squeeze(0)
    return resized

def resize_to_square(frame: torch.Tensor, target_size: int) -> torch.Tensor:
    """Resize frame to square by stretching."""
    return torch.nn.functional.interpolate(frame.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False).squeeze(0)

class SiglipWrapper:
    def __init__(self, model_root: str = DEFAULT_MODEL_ROOT, patch_size: int = 16, target_size: int = 384):
        self.processor = AutoProcessor.from_pretrained(model_root)
        self.model = AutoModel.from_pretrained(model_root)
        self.root = model_root
        self.patch_size = patch_size
        self.target_size = target_size

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        # frame is [C, H, W], convert to [H, W, C] and uint8
        img = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0), outputs.pooler_output  # [H', W', D]
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-file", type=str, required=True, help="Path to the semantic latents file (e.g., .pt)")
    parser.add_argument("--video-file", type=str, required=True, help="Path to the video file (e.g., .mp4)")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)

    args = parser.parse_args()
    semantic_latents = torch.load(args.semantic_file)
    print(f"Loaded semantic latents from {args.semantic_file} with shape {semantic_latents.shape}")
    frames = load_clip_frames(args.video_file, args.height, args.width, max_frames=81)
    
    frame = frames[0]

    siglip_variants = [
        SiglipWrapper("google/siglip-so400m-patch14-384", patch_size=14, target_size=384),
        SiglipWrapper("google/siglip-base-patch16-224", patch_size=16, target_size=224),
    ]

    preprocess_fns = {
        "pad_to_square_and_resize": lambda x, target_size: pad_to_square_and_resize(x, target_size),
        "resize_to_square": lambda x, target_size: resize_to_square(x, target_size),
        "none": lambda x, target_size: x,
    }

    for siglip in siglip_variants:
        for preprocess_name, preprocess_fn in preprocess_fns.items():
            preprocessed = preprocess_fn(frame, siglip.target_size)
            siglip_patches, siglip_pooled = siglip.forward(preprocessed)
            print(f"{siglip.root}/{preprocess_name}: patches shape={siglip_patches.shape}, pooled shape={siglip_pooled.shape}")
            

if __name__ == "__main__":
    main()
