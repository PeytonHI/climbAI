""" Download videos and json info from a YouTube playlist. """

import yt_dlp
import os

video_dir = 'climbVideoTrainingDownloads/videos'
# json_dir  = 'climbVideoTrainingDownloads/json'

os.makedirs(video_dir, exist_ok=True)
# os.makedirs(json_dir, exist_ok=True)

def progress_hook(d):
    last_percent = {}
    if d['status'] == 'downloading':
        # old_name = d['filename']
        # new_name = os.path.splitext(old_name)[0]  # removes extension
        # os.rename(old_name, new_name)
        # print(f"Renamed {old_name} → {new_name}")
        video_id = d['filename']
        total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
        downloaded = d.get('downloaded_bytes', 0)
        if total_bytes:
            percent = int(downloaded / total_bytes * 100)
            # Only print if 10% increment
            if video_id not in last_percent or percent // 10 > last_percent[video_id] // 10:
                print(f"{video_id}: {percent}%")
                last_percent[video_id] = percent
    elif d['status'] == 'finished':
        print(f"Finished: {d['filename']}")

yt_dlp_options = {
    'format': 'bestvideo[ext=mp4]',
    # 'write_info_json': True,
    'skip_download': False,
    'ignorealreadydownloaded': True,
    'outtmpl': {
        'default': f'{video_dir}/%(title)s.%(ext)s'
        # 'infojson': f'{json_dir}/%(playlist)s/%(title)s.%(ext)s'
    },
    'progress_hooks': [progress_hook]
}

playlist_url = "https://www.youtube.com/shorts/UBpd5yCCYOs"

print("Starting download...")
with yt_dlp.YoutubeDL(yt_dlp_options) as ydl:
    ydl.download([playlist_url])
print("Download complete.")

