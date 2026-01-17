""" Helper functions for pose map generation and video saving."""
import numpy as np
import cv2

POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (11, 12), (23, 24), (11, 23), (12, 24)
]

def keypointsToPoseMap(frameKps, imgSize=(512, 512)):
    img = np.zeros(imgSize, dtype=np.uint8)
    kps = (frameKps * np.array(imgSize)[None, :]).astype(int)
    for i, j in POSE_CONNECTIONS:
        cv2.line(img, tuple(kps[i]), tuple(kps[j]), 255, 2)
    for x, y in kps:
        cv2.circle(img, (x, y), 3, 255, -1)
    return img

def saveFramesAsVideo(framesList, outputPath="predicted_climb.mp4", fps=30):
    h, w = framesList[0].size[1], framesList[0].size[0]
    outVid = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for f in framesList:
        frame = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
        outVid.write(frame)
    outVid.release()
    print(f"Saved video to {outputPath}")

