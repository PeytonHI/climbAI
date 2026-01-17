""" Inference script for predicting future climbing poses and generating realistic climber images using Stable Diffusion ControlNet."""
import os
import time
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
    wallImg = Image.open(climbingImageFile).convert("RGB")
    wallImg = wallImg.resize((512, 512))  # match your skeleton map size
    return wallImg

def overlaySkeletonOnWall(skeletonMap, wallImage):
    """
    skeleton_map: uint8 array (512x512) or RGB
    wall_image: PIL Image RGB
    """
    # convert skeleton to RGB
    skeletonRgb = np.stack([skeletonMap]*3, axis=-1)
    skeletonPil = Image.fromarray(skeletonRgb)

    # optional: make skeleton semi-transparent
    overlayImg = Image.blend(wallImage, skeletonPil, alpha=0.6)  # alpha 0.6 for skeleton
    return overlayImg

def generateRealisticClimber(predSeq, wallImage, prompt="A climber on a climbing wall", device="cuda"):
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    import torch
    controlNet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlNet,
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)

    framesList = []
    for t in range(predSeq.shape[0]):
        kps = predSeq[t].reshape(33, 3)[:,:2]  # xy
        skeletonMap = helpers.keypointsToPoseMap(kps)
        skeletonOverlay = overlaySkeletonOnWall(skeletonMap, wallImage)

        output = pipe(
            prompt=prompt,
            image=skeletonOverlay,
            guidance_scale=7.5,
            num_inference_steps=20
        )
        framesList.append(output.images[0])
    return framesList

def loadModel(checkpointPath, inputDim, device="cuda"):
    model = PoseTransformer(input_dim=inputDim)
    model.load_state_dict(torch.load(checkpointPath, map_location=device))
    model.to(device).eval()
    return model

def predictFuture(model, contextSeq, predictLen=60, device="cuda"):
    """
    contextSeq: np.array (K, D) normalized the same way as training
    returns: (predictLen, D)
    """
    model.eval()
    K = contextSeq.shape[0]
    seqLen = K + predictLen
    inputSeq = np.zeros((1, seqLen, contextSeq.shape[1]), dtype=np.float32)
    inputSeq[0,:K,:] = contextSeq
    with torch.no_grad():
        t = torch.from_numpy(inputSeq).to(device)
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
    rootDir = "data" if os.path.exists("data") else f"No data folder found in {os.getcwd()}"
    for sub in os.listdir(rootDir):
        p = os.path.join(rootDir, sub, "poses.npy")
        if os.path.exists(p):
            files.append(p)

    sample = files[0]
    print("Using sample file:", sample)
    arr = np.load(sample)  # shape (T, 33, 4)
    print("Original shape:", arr.shape)

    # Parameters
    startFrame = 10  # first frame of the segment
    endFrame = 30    # last frame of the segment (exclusive)

    segment = arr[startFrame:endFrame]  # shape will be (end_frame-start_frame, 33, 4)
    print("Segment shape:", segment.shape)

    # Preprocess like dataset: take xy+vis flatten and center on hips
    seqLen = arr.shape[0]
    K = 8
    useVis = True
    data = arr[:,:,:3]  # (T,33,3)
    T, L, D = data.shape
    midHip = (arr[:,23,:2] + arr[:,24,:2]) / 2.0
    for t in range(T):
        data[t,:,:2] = data[t,:,:2] - midHip[t:t+1]
    flat = data.reshape(T, L*D)
    contextSeq = flat[:K]
    inputDim = flat.shape[1]
    # model = load_model(args.model, input_dim, device="cpu")
    modelFile = "checkpoints/model_ep0.pth"
    gpuDevice = "cuda"
    model = loadModel(modelFile, inputDim, device=gpuDevice)

    pred = predictFuture(model, contextSeq, predictLen=seqLen-K, device=gpuDevice)
    outSeq = np.concatenate([contextSeq, pred], axis=0)

    wallImg = getClimbingImage()
    framesList = []

    for t in range(outSeq.shape[0]):
        kps = outSeq[t].reshape(L, D)[:, :2]  # xy only
        skeletonMap = helpers.keypointsToPoseMap(kps)  # returns 512x512 uint8
        frame = overlaySkeletonOnWall(skeletonMap, wallImg)
        framesList.append(frame)


    # Generate realistic climber frames
    predClimbOutFile = "predicted_climb.mp4"
    # framesList = generateRealisticClimber(outSeq)
    helpers.saveFramesAsVideo(framesList, outputPath=predClimbOutFile)

