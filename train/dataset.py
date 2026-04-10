from torch.utils.data import Dataset
import torch
import json
import decord
import numpy as np
from PIL import Image

class SemanticEditDataset(Dataset):
    """
    Dataset for semantic video/image editing. Manifest files contain paths to videos or images.
    Returns source and target videos/images as [C, T, H, W] tensors (T=1 for images) along with
    the corresponding text prompt. Frames are normalized to [0, 1], and videos are truncated to 
    the first 4k+1 frames.
    """

    def __init__(self, manifests, mode="video"):
        self.data = []
        self.mode = mode
        for manifest in manifests:
            with open(manifest, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.mode == "video":
            source_path = item["original_video"]
            target_path = item["video"]
        
            # Load video frames using decord
            vr = decord.VideoReader(source_path)
            source_frames = vr.get_batch(range(len(vr))).asnumpy()  # Get all frames as a numpy array

            vr = decord.VideoReader(target_path)
            target_frames = vr.get_batch(range(len(vr))).asnumpy()  # Get all frames as a numpy array

            # convert to torch (shape is [num_frames, height, width, channels])
            source_frames = torch.from_numpy(source_frames).float() / 255.0  # Normalize to [0, 1]
            target_frames = torch.from_numpy(target_frames).float() / 255.0  # Normalize to [0, 1]

            # ensure frame count has the form 4k+1
            frame_count = source_frames.shape[0]
            if (frame_count - 1) % 4 != 0:
                frame_count = (frame_count - 1) // 4 * 4 + 1

            # cap at 81 frames as wan's limit.
            if frame_count > 81:
                frame_count = 81 

            source_frames = source_frames[:frame_count]
            target_frames = target_frames[:frame_count]

            return {
                "source": source_frames.permute(3, 0, 1, 2), # [C, T, H, W]
                "target": target_frames.permute(3, 0, 1, 2), # [C, T, H, W]
                "prompt": item["prompt"].strip(),
            }
        elif self.mode == "image":
            source_path = item["original_image"]
            target_path = item["image"]
            source_image = Image.open(source_path).convert("RGB")
            target_image = Image.open(target_path).convert("RGB")

            # convert to torch
            source_image = torch.from_numpy(np.array(source_image)).float() / 255.0  # Normalize to [0, 1]
            target_image = torch.from_numpy(np.array(target_image)).float() / 255.0  # Normalize to [0, 1]
            return {
                "source": source_image.permute(2, 0, 1).unsqueeze(1),  # Convert to [C, 1, H, W]
                "target": target_image.permute(2, 0, 1).unsqueeze(1),  # Convert to [C, 1, H, W]
                "prompt": item["prompt"].strip(),
            }
