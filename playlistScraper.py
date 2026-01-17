""" Download videos and json info from a YouTube playlist. """

import yt_dlp
import os

# Progress hook to report download progress
# In: dictionaries of data from yt_dlp with keys like status, filename, percent, etc.
#
# docs: https://github.com/yt-dlp/yt-dlp/blob/master/yt_dlp/YoutubeDL.py#L181
def progressHook(data):
    global lastPercentDownloaded

    if data['status'] == 'downloading':
        videoId = data['filename']
        totalBytes = data.get('total_bytes', 0) # sometimes videos only report estimates of bytes

        # No exact total bytes available, use estimate
        if totalBytes is None or totalBytes == 0: 
            totalBytes = data.get('total_bytes_estimate')

        downloadedBytes = data.get('downloaded_bytes', 0)

        # Avoid division by zero
        if totalBytes is not None and totalBytes > 0: 
            percentDownloaded = int(downloadedBytes / totalBytes * 100)

            # Print 10% increments
            incrementValue = 10
            currentIncrement = percentDownloaded // incrementValue
            lastIncrement = lastPercentDownloaded.get(videoId, 0) // incrementValue

            # Track videos already seen, only print when increment changes
            if currentIncrement > lastIncrement:
                print(f"{videoId}: {percentDownloaded}%")
                lastPercentDownloaded[videoId] = percentDownloaded

    # All pau             
    elif data['status'] == 'finished':
        print(f"Finished: {data['filename']}")

### Main

# Set up download directories
videoDir = os.path.join("climbVideoTrainingDownloads", "videos",)
# json_dir  = os.path.join("climbVideoTrainingDownloads", "json")
# os.makedirs(json_dir, exist_ok=True)
os.makedirs(videoDir, exist_ok=True)

lastPercentDownloaded = {}

# yt_dlp options for downloading videos and json info
ytDlpOptions = {
    'format': 'bestvideo[ext=mp4]', # best available mp4 format, skip other formats for now
    # 'write_info_json': True,
    # 'skip_download': True, # for testing json only
    'download_archive': f'{videoDir}/archive.txt', # to avoid redownloading, yt_dlp will check for existing entries 

    # Output template (how and where to save files)
    # specify default download path, yt_dlp expects this format, % are placeholders and s converts to string from yt_dlp api
    'outtmpl': {
        'default': f'{videoDir}/%(title)s.%(ext)s'
        # 'infojson': f'{json_dir}/%(playlist)s/%(title)s.%(ext)s'
    },
    'progress_hooks': [progressHook],  # yt_dlp calls a list of functions periodically for differing progress updates

}
playlistUrl = "https://www.youtube.com/shorts/UBpd5yCCYOs" # lovely hard code for now until CL added

print("Starting download...")

# Use yt_dlp to download the playlist
with yt_dlp.YoutubeDL(ytDlpOptions) as ydl:
    ydl.download([playlistUrl])

print("Download complete.")

