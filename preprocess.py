# preprocess.py
import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

mpPose = mp.solutions.pose

def extractFrames(videoPath, outFramesDir, fpsTarget=30):
    cap = cv2.VideoCapture(videoPath)
    origFps = cap.get(cv2.CAP_PROP_FPS) or fpsTarget
    frameStep = max(1, int(round(origFps / fpsTarget)))  # slowdown fps to target fps
    idx = 0
    saved = 0

    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pbar = tqdm(total=totalFrames, desc=f"Extracting frames from {os.path.basename(videoPath)}")  # progress bar

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frameStep == 0:  # valid frame to save
            fname = os.path.join(outFramesDir, f"frame_{saved:06d}.jpg") # 999,999 limit
            cv2.imwrite(fname, frame)
            saved += 1
        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"Saved {saved} frames to {outFramesDir}")
    # return saved

def extractPoseFromFrames(framesDir, outPoseNpy, visibilityThresh=0.3):
    print("Extracting poses from frames in:", framesDir)
    poseFiles = []
    for f in os.listdir(framesDir):
        if f.endswith(".jpg"):
            poseFiles.append(os.path.join(framesDir, f))
    poseList = []
    # Use MediaPipe Pose to extract landmarks

    print("Files found in framesDir:", os.listdir(framesDir))

    with mpPose.Pose(static_image_mode=True, model_complexity=0) as pose:
        for f in tqdm(poseFiles, desc="Pose"):
            img = cv2.imread(f)
            imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = pose.process(imgRgb)
            if res.pose_landmarks:
                # 33 landmarks: each with x, y, z, visibility
                lm = res.pose_landmarks.landmark
                arr = np.array([[p.x, p.y, p.z, p.visibility if p.visibility >= visibilityThresh else 0.0] for p in lm], dtype=np.float32)
            else:
                arr = np.zeros((33, 4), dtype=np.float32)  # no detection -> zeros
            poseList.append(arr)
    poses = np.stack(poseList)  # (numFrames, 33, 4)
    np.save(outPoseNpy, poses)
    print("Saved poses:", outPoseNpy, poses.shape)
    return poses

if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--video", required=True)
    # parser.add_argument("--outdir", default="data")
    # parser.add_argument("--fps", type=int, default=30)
    # args = parser.parse_args()

    # base = os.path.splitext(os.path.basename(args.video))[0]
    # frames_dir = os.path.join(args.outdir, base, "frames")
    # os.makedirs(frames_dir, exist_ok=True)

    # print("Extracting frames...")
    # extract_frames(args.video, frames_dir, fps_target=args.fps)
    # pose_npy = os.path.join(args.outdir, base, "poses.npy")
    # extract_pose_from_frames(frames_dir, pose_npy)

    # jsonDir = os.path.join("climbVideoTrainingDownloads","json")

    print("Processing all videos in downloaded directory...")

    # videoDir = os.path.join("climbVideoTrainingDownloads", "videos")
    videoDir = r"E:\videos"

    outRoot = "data"
    os.makedirs(outRoot, exist_ok=True)

    for origVideoFile in os.listdir(videoDir):
        if not origVideoFile.endswith(".mp4"):
            continue
        
        videoFile = origVideoFile.removesuffix("mp4")
        videoFile = os.path.splitext(videoFile)[0]
        if videoFile not in os.listdir(outRoot):
            print(f"Processing video: {videoFile}")
            videoPath = os.path.join(videoDir, origVideoFile)
            base = os.path.splitext(videoFile)[0]
            # base = videoFile.removesuffix("mp4")

            framesDir = os.path.join("data", base, "frames")
            os.makedirs(framesDir, exist_ok=True)
            extractFrames(videoPath, framesDir, fpsTarget=30)

            poseNpy = os.path.join("data", base, "poses.npy")
            os.makedirs(os.path.dirname(poseNpy), exist_ok=True)
            extractPoseFromFrames(framesDir, poseNpy)

            print(f"Finished processing {videoFile}\n")