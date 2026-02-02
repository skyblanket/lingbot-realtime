#!/usr/bin/env python3
"""
Download training video dataset for DMD distillation
Options:
1. MixKit (free CC0 videos) - 4K nature, city, etc.
2. Pexels (free videos)
3. Use existing videos from /home/sky/captures/
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def download_mixkit_dataset(output_dir="./training_data", num_videos=50):
    """Download free videos from MixKit"""
    # MixKit categories that work well for world generation training
    categories = [
        "nature", "city", "travel", "aerial", "abstract",
        "technology", "business", "people", "lifestyle"
    ]
    
    print(f"[Dataset] Downloading {num_videos} videos from MixKit categories: {categories}")
    print("Note: This is a placeholder. MixKit requires scraping or API access.")
    print("Alternative: Use youtube-dl to download specific playlists")
    
    return []


def download_youtube_playlist(playlist_url, output_dir="./training_data"):
    """Download videos from YouTube playlist"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Dataset] Downloading from: {playlist_url}")
    print(f"[Dataset] Output: {output_dir}")
    
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", f"{output_dir}/%(playlist_index)03d_%(title)s.%(ext)s",
        "--no-playlist",  # Download as separate videos
        playlist_url
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[Dataset] Download complete!")
        return list(output_dir.glob("*.mp4"))
    except subprocess.CalledProcessError as e:
        print(f"[Dataset] Download failed: {e}")
        return []


def use_existing_videos(video_dir="/home/sky/captures", output_dir="./training_data"):
    """Copy existing videos to training directory"""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm"))
    
    if len(video_files) == 0:
        print(f"[Dataset] No videos found in {video_dir}")
        return []
    
    print(f"[Dataset] Found {len(video_files)} videos in {video_dir}")
    print(f"[Dataset] Copying to {output_dir}...")
    
    import shutil
    for i, video in enumerate(video_files):
        dest = output_dir / f"{i:03d}_{video.name}"
        shutil.copy2(video, dest)
        print(f"  [{i+1}/{len(video_files)}] {video.name}")
    
    return list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.webm"))


def download_sample_videos(output_dir="./training_data"):
    """Download a few sample CC0 videos for testing"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Direct links to sample CC0 videos
    sample_urls = [
        # Nature scenes
        "https://videos.pexels.com/video-files/857251/857251-hd_1920_1080_25fps.mp4",
        "https://videos.pexels.com/video-files/857195/857195-hd_1920_1080_25fps.mp4",
        # City scenes
        "https://videos.pexels.com/video-files/3129671/3129671-hd_1920_1080_30fps.mp4",
        # Aerial/Drone
        "https://videos.pexels.com/video-files/18069166/18069166-hd_1920_1080_25fps.mp4",
    ]
    
    print(f"[Dataset] Downloading {len(sample_urls)} sample videos...")
    
    downloaded = []
    for i, url in enumerate(sample_urls):
        output_file = output_dir / f"sample_{i:02d}.mp4"
        
        cmd = ["wget", "-q", "--show-progress", "-O", str(output_file), url]
        
        try:
            subprocess.run(cmd, check=True)
            downloaded.append(output_file)
            print(f"  ✓ Downloaded: {output_file.name}")
        except subprocess.CalledProcessError:
            print(f"  ✗ Failed: {url}")
    
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download training videos for DMD distillation")
    parser.add_argument("--source", choices=["mixkit", "youtube", "existing", "sample"], 
                       default="sample", help="Video source")
    parser.add_argument("--output", default="./training_data", help="Output directory")
    parser.add_argument("--youtube-url", help="YouTube playlist or video URL")
    parser.add_argument("--existing-dir", default="/home/sky/captures", 
                       help="Directory with existing videos")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LingBot-World Training Dataset Downloader")
    print("=" * 70)
    
    if args.source == "mixkit":
        videos = download_mixkit_dataset(args.output)
    elif args.source == "youtube":
        if not args.youtube_url:
            print("Error: --youtube-url required")
            sys.exit(1)
        videos = download_youtube_playlist(args.youtube_url, args.output)
    elif args.source == "existing":
        videos = use_existing_videos(args.existing_dir, args.output)
    else:  # sample
        videos = download_sample_videos(args.output)
    
    print("\n" + "=" * 70)
    print(f"Downloaded {len(videos)} videos to {args.output}")
    print("=" * 70)
    
    # List videos
    if videos:
        print("\nVideo files:")
        for v in sorted(videos)[:10]:
            size_mb = v.stat().st_size / (1024 * 1024)
            print(f"  - {v.name} ({size_mb:.1f} MB)")
        if len(videos) > 10:
            print(f"  ... and {len(videos) - 10} more")
    
    return len(videos)


if __name__ == "__main__":
    count = main()
    sys.exit(0 if count > 0 else 1)
