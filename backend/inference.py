"""
Inference module for First-Order-Model face animation.
Based on: https://colab.research.google.com/github/eyaler/avatars4all/blob/master/fomm_bibi.ipynb
"""

import os
import sys
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings

import imageio
import numpy as np
import torch
import yaml
from scipy.spatial import ConvexHull
from skimage.transform import resize
from skimage import img_as_ubyte
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Add first-order-model to path
FIRST_ORDER_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "first-order-model")
)
if FIRST_ORDER_MODEL_DIR not in sys.path:
    sys.path.insert(0, FIRST_ORDER_MODEL_DIR)

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from sync_batchnorm import DataParallelWithCallback


@dataclass
class AnimationConfig:
    """Configuration for animation generation."""
    find_best_frame: bool = True
    relative: bool = True
    adapt_scale: bool = True
    cpu: bool = False


def load_models(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    cpu: bool = False
) -> Tuple:
    """
    Load the generator and keypoint detector models.
    
    Returns:
        Tuple of (generator, kp_detector)
    """
    if config_path is None:
        config_path = os.path.join(FIRST_ORDER_MODEL_DIR, "config", "vox-adv-256.yaml")
    if checkpoint_path is None:
        checkpoint_path = os.path.join(FIRST_ORDER_MODEL_DIR, "vox-adv-cpk.pth.tar")
    
    # Check if files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)

    generator = OcclusionAwareGenerator(
        **config['model_params']['generator_params'],
        **config['model_params']['common_params']
    )
    
    kp_detector = KPDetector(
        **config['model_params']['kp_detector_params'],
        **config['model_params']['common_params']
    )
    
    if not cpu and torch.cuda.is_available():
        generator.cuda()
        kp_detector.cuda()
        checkpoint = torch.load(checkpoint_path)
    else:
        cpu = True
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
    
    generator.eval()
    kp_detector.eval()
    
    print(f"✓ Models loaded successfully (CPU mode: {cpu})")
    return generator, kp_detector


def normalize_kp(kp_source, kp_driving, kp_driving_initial, 
                 adapt_movement_scale=False, use_relative_movement=False, 
                 use_relative_jacobian=False):
    """Normalize keypoints for animation."""
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(
                kp_driving['jacobian'], 
                torch.inverse(kp_driving_initial['jacobian'])
            )
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def find_best_frame(source, driving, cpu=False):
    """Find the frame in driving video that best aligns with source image."""
    import face_alignment
    
    def normalize_kp_local(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp
    
    device = 'cpu' if cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, 
        flip_input=True,
        device=device
    )
    
    kp_source = fa.get_landmarks(255 * source)
    if kp_source is None:
        return 0
    kp_source = normalize_kp_local(kp_source[0])
    
    norm = float('inf')
    frame_num = 0
    
    for i, image in enumerate(driving):
        try:
            kp_driving = fa.get_landmarks(255 * image)
            if kp_driving is None:
                continue
            kp_driving = normalize_kp_local(kp_driving[0])
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        except Exception:
            continue
    
    return frame_num


def _make_animation_internal(source_image, driving_video, generator, kp_detector,
                              relative=True, adapt_movement_scale=True, cpu=False):
    """Generate animation frames."""
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        
        if not cpu and torch.cuda.is_available():
            source = source.cuda()
        
        driving = torch.tensor(
            np.array(driving_video)[np.newaxis].astype(np.float32)
        ).permute(0, 4, 1, 2, 3)
        
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2]), desc="Generating animation"):
            driving_frame = driving[:, :, frame_idx]
            if not cpu and torch.cuda.is_available():
                driving_frame = driving_frame.cuda()
            
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(
                kp_source=kp_source, 
                kp_driving=kp_driving,
                kp_driving_initial=kp_driving_initial, 
                use_relative_movement=relative,
                use_relative_jacobian=relative, 
                adapt_movement_scale=adapt_movement_scale
            )
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            predictions.append(
                np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            )
    
    return predictions


def make_animation(
    source_path: str,
    driving_path: str,
    output_path: str,
    generator,
    kp_detector,
    config: AnimationConfig
):
    """
    Generate an animated video from source image and driving video.
    
    Args:
        source_path: Path to source image
        driving_path: Path to driving video
        output_path: Path to save output video
        generator: Loaded generator model
        kp_detector: Loaded keypoint detector model
        config: Animation configuration
    """
    cpu = config.cpu or not torch.cuda.is_available()
    
    # Load and preprocess source image
    source_image = imageio.imread(source_path)
    source_image = resize(source_image, (256, 256))[..., :3]
    
    # Load and preprocess driving video
    reader = imageio.get_reader(driving_path)
    fps = reader.get_meta_data().get('fps', 25)
    driving_video = []
    try:
        for im in reader:
            driving_video.append(resize(im, (256, 256))[..., :3])
    except RuntimeError:
        pass
    reader.close()
    
    if len(driving_video) == 0:
        raise ValueError("Could not read any frames from driving video")
    
    # Find best frame if requested
    if config.find_best_frame:
        best = find_best_frame(source_image, driving_video, cpu=cpu)
        print(f"Best aligned frame: {best}")
        
        # Generate forward and backward animations
        driving_forward = driving_video[best:]
        driving_backward = driving_video[:(best+1)][::-1]
        
        predictions_forward = _make_animation_internal(
            source_image, driving_forward, generator, kp_detector,
            relative=config.relative, adapt_movement_scale=config.adapt_scale, cpu=cpu
        )
        predictions_backward = _make_animation_internal(
            source_image, driving_backward, generator, kp_detector,
            relative=config.relative, adapt_movement_scale=config.adapt_scale, cpu=cpu
        )
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = _make_animation_internal(
            source_image, driving_video, generator, kp_detector,
            relative=config.relative, adapt_movement_scale=config.adapt_scale, cpu=cpu
        )
    
    # Save output video
    imageio.mimsave(output_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    print(f"✓ Animation saved to: {output_path}")


if __name__ == "__main__":
    # Test loading models
    print("Testing model loading...")
    generator, kp_detector = load_models()
    print("Models loaded successfully!")
