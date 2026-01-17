# inference.py
import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import PoseTransformer
from dataset import PoseSequenceDataset  # just to know input dim / normalization rules
import helpers

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image


def getClimbingImage():
    climbingImageFile = "climbing_img.jpg"
    wall_img = Image.open(climbingImageFile).convert("RGB")
    wall_img = wall_img.resize((512, 512))  # match your skeleton map size
    return wall_img

def overlay_skeleton_on_wall(skeleton_map, wall_image):
    """
    skeleton_map: uint8 array (512x512) or RGB
    wall_image: PIL Image RGB
    """
    # convert skeleton to RGB
    skeleton_rgb = np.stack([skeleton_map]*3, axis=-1)
    skeleton_pil = Image.fromarray(skeleton_rgb)

    # optional: make skeleton semi-transparent
    overlay = Image.blend(wall_image, skeleton_pil, alpha=0.6)  # alpha 0.6 for skeleton
    return overlay

def generate_realistic_climber(pred_seq, wall_image, prompt="A climber on a climbing wall", device="cuda"):
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    import torch

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)

    frames = []
    for t in range(pred_seq.shape[0]):
        kps = pred_seq[t].reshape(33, 3)[:,:2]  # xy
        skeleton_map = helpers.keypoints_to_pose_map(kps)
        skeleton_overlay = overlay_skeleton_on_wall(skeleton_map, wall_image)
        
        output = pipe(
            prompt=prompt,
            image=skeleton_overlay,
            guidance_scale=7.5,
            num_inference_steps=20
        )
        frames.append(output.images[0])
    return frames

def load_model(checkpoint_path, input_dim, device="cuda"):
    model = PoseTransformer(input_dim=input_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model

def predict_future(model, context_seq, predict_len=60, device="cuda"):
    """
    context_seq: np.array (K, D) normalized the same way as training
    returns: (predict_len, D)
    """
    model.eval()
    K = context_seq.shape[0]
    seq_len = K + predict_len
    input_seq = np.zeros((1, seq_len, context_seq.shape[1]), dtype=np.float32)
    input_seq[0,:K,:] = context_seq
    with torch.no_grad():
        t = torch.from_numpy(input_seq).to(device)
        out = model(t)  # (1, seq_len, D)
        pred = out[0, K:, :].cpu().numpy()
    return pred

if __name__ == "__main__":
    # Demo usage (requires you to have a saved model and a sample file)
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", required=True)
    # parser.add_argument("--sample", required=True, help="path to .npy sequence (T,33,4) or a single sequence saved by dataset")
    # args = parser.parse_args()
    # arr = np.load(args.sample)  # (T,33,4)
    files = []
    root_dir = "data" if os.path.exists("data") else f"No data folder found in {os.getcwd()}"
    for sub in os.listdir(root_dir):
        p = os.path.join(root_dir, sub, "poses.npy")
        if os.path.exists(p):
            files.append(p)

    sample = files[0]
    print("Using sample file:", sample)
    arr = np.load(sample)  # shape (T, 33, 4)
    print("Original shape:", arr.shape)

    # Parameters
    start_frame = 10  # first frame of the segment
    end_frame = 30    # last frame of the segment (exclusive)

    segment = arr[start_frame:end_frame]  # shape will be (end_frame-start_frame, 33, 4)
    print("Segment shape:", segment.shape)

    # Preprocess like dataset: take xy+vis flatten and center on hips
    seq_len = arr.shape[0]
    K = 8
    use_vis = True
    data = arr[:,:,:3]  # (T,33,3)
    T, L, D = data.shape
    mid_hip = (arr[:,23,:2] + arr[:,24,:2]) / 2.0
    for t in range(T):
        data[t,:,:2] = data[t,:,:2] - mid_hip[t:t+1]
    flat = data.reshape(T, L*D)
    context = flat[:K]
    input_dim = flat.shape[1]
    # model = load_model(args.model, input_dim, device="cpu")
    modelFile = "checkpoints/model_ep0.pth"
    gpuDevice = "cuda"
    model = load_model(modelFile, input_dim, device=gpuDevice)

    pred = predict_future(model, context, predict_len=seq_len-K, device=gpuDevice)
    out_seq = np.concatenate([context, pred], axis=0)

    wall_img = getClimbingImage()
    frames = []

    for t in range(out_seq.shape[0]):
        kps = out_seq[t].reshape(L, D)[:, :2]  # xy only
        skeleton_map = helpers.keypoints_to_pose_map(kps)  # returns 512x512 uint8
        frame = overlay_skeleton_on_wall(skeleton_map, wall_img)
        frames.append(frame)


    # Generate realistic climber frames
    predClimbOutFile = "predicted_climb.mp4"
    # frames = generate_realistic_climber(out_seq)
    helpers.save_frames_as_video(frames, output_path=predClimbOutFile)

