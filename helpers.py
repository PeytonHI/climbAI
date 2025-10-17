# helpers.py
import numpy as np
import cv2

POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (11, 12), (23, 24), (11, 23), (12, 24)
]

def keypoints_to_pose_map(frame_kps, img_size=(512, 512)):
    img = np.zeros(img_size, dtype=np.uint8)
    kps = (frame_kps * np.array(img_size)[None, :]).astype(int)
    for i, j in POSE_CONNECTIONS:
        cv2.line(img, tuple(kps[i]), tuple(kps[j]), 255, 2)
    for x, y in kps:
        cv2.circle(img, (x, y), 3, 255, -1)
    return img

def save_frames_as_video(frames, output_path="predicted_climb.mp4", fps=30):
    h, w = frames[0].size[1], frames[0].size[0]
    out_vid = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for f in frames:
        frame = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
        out_vid.write(frame)
    out_vid.release()
    print(f"Saved video to {output_path}")

