""" Preprocessing script to extract frames from climbing videos and obtain pose landmarks using MediaPipe. """
import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# Globals
mpPose = mp.solutions.pose # load MediaPipe solutions.pose module, create pose object

def extractFramesFromVideo(videoPath, outFramesDir, fpsTarget=30):
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

# Extract pose landmarks from all frames from a file
# In: framesFileDir (directory containing frames for a single video)
# Out: poses (numpy array of shape (numFrames, 33, 4))
# Args:
#   framesFileDir: directory containing frames for a single video
#   visibilityThresh: minimum visibility threshold for landmarks (joints detected from person, 0-1, changeable)
def extractPoseFromFrames(framesFileDir, visibilityThresh=0.3):
    print("Extracting poses from frames in:", framesFileDir)

    # Get list of all frame files paths in framesFileDir
    poseFiles = []
    for file in os.listdir(framesFileDir):
        if file.endswith(".jpg"):
            poseFiles.append(os.path.join(framesFileDir, file))

    poseList = []  # list to store pose landmarks for each frame

    print("Files found in framesFileDir:", os.listdir(framesFileDir))

    # Use MediaPipe Pose to extract landmarks from each frame
    # Params: static_image_mode=False (true = treat input images as static, false = video [default]), 
    #   min_detection_confidence (minimum model confidence for detection, higher = more accurate but potentially less detections),
    #   min_tracking_confidence (minimum model confidence for tracking, higher = more accurate but potentially less tracks),
    #   model_complexity=0 (lightweight model -> 0,1,2 options for heavier/better)
    # Notes:
    #   mpPose object is using python context manager -> with .. as.. calls object. __enter__, then object.__exit__  at end of block 
    #   from mpPose.Pose and assigns whatever __enter__ returns to the variable after "as" (in this case, pose)
    with mpPose.Pose(static_image_mode=False, min_detection_confidence = 0.8, min_tracking_confidence=0.8, model_complexity=0) as pose:
        for frameFile in tqdm(poseFiles, desc="Pose"): # progress bar
            img = cv2.imread(frameFile)
            imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 uses BGR by default, convert to RGB
            result = pose.process(imgRgb) # extract pose landmarks from most prominent person detected (pose doc strings)
            
            # landmarks detected? extract list of landmarks from NamedTuple objects (pose module documentation)
            # extracts a list-like container of our landmarks (can access like a list).. I don't like this, check pose module documentation
            # 33 landmarks (1 per joint of person detected): each with x, y, z, visibility attributes
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark 
                array = [] 
                for lm in landmarks:  # iterate over 33 landmarks, only keep those with visibility above threshold
                    if lm.visibility >= visibilityThresh:  
                        visibility = lm.visibility 
                    else:
                        visibility = 0.0
                    array.append([lm.x, lm.y, lm.z, visibility]) 
                array = np.array(array, dtype=np.float32) # make sure float values
            else:
                array = np.zeros((33, 4), dtype=np.float32) # no detection -> default to zeros, (33, 4) = 33 landmarks, 4 attributes (x, y, z, visibility)
            poseList.append(array) # list of landmark lists of each frame -> 
    if not poseList:
        print(f"No poses found for {framesFileDir}")
        return None
    else:
        poses = np.stack(poseList)  # dims = (numFrames, 33, 4)
    np.save("poses.npy", poses)  # save poses to .npy file
    print("Saved poses:", "poses.npy", poses.shape)

    return poses


if __name__ == "__main__":
    # For command-line usage
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


    # Temporary python only usage
    print("Processing all videos in downloaded directory...")
    
    baseDir = os.path.dirname(os.path.abspath(__file__))
    videoDir = os.path.join(baseDir, "climbVideoTrainingDownloads", "videos")
    os.makedirs(videoDir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Process each video file
    for origVideoFile in os.listdir(videoDir):
        if not origVideoFile.endswith(".mp4"):
            continue
        
        videoFileName = origVideoFile.removesuffix(".mp4")

        # Check if video has already been processed
        if videoFileName not in os.listdir("data"):
            print(f"Processing video: {videoFileName}")
            videoFilePath = os.path.join(videoDir, origVideoFile)
            framesFileDir = os.path.join("data", videoFileName, "frames")
            os.makedirs(framesFileDir, exist_ok=True)

            # extractFramesFromVideo(videoFilePath, framesFileDir, fpsTarget=30) # extract frames from single video
            poseNpyDir= os.path.join("data", videoFileName, "poses")
            os.makedirs(poseNpyDir, exist_ok=True)

            extractPoseFromFrames(framesFileDir) # extract poses from all frames of single video
            print(f"Finished processing {videoFileName}\n")