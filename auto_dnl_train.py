import os
import sys
from pathlib import Path
import yaml
import yt_dlp
import gc
import torch
from train_vqvae import main as train_vqvae_main
from train_generator import main as train_generator_main
import globals
import test


def download_youtube_audio(query, download_dir="DNL", max_results=10):
    """
    Searches YouTube for a given query and downloads the best audio as WAV,
    limited to the first 30 seconds of each track.
    """
    max_results = int(max_results)
    print(f"--- Fetching {max_results} tracks for: '{query}' (max 30s each) ---")

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    else:
        folder_path = Path(download_dir)
        file_count = sum(1 for item in folder_path.iterdir() if item.is_file())
        if file_count >= max_results:
            print(f"--- Already have {file_count} files in {download_dir}. Skipping download. ---")
            return

    def _first_30s(info_dict, ydl):
        """Download only the first 30 seconds of each track."""
        duration = info_dict.get('duration') or 30
        end = min(30, float(duration))
        return [{'start_time': 0, 'end_time': end}]

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'noplaylist': True,
        'default_search': f'ytsearch{max_results}:',
        'quiet': False,
        # Limit each song to ≤ 30 seconds
        'download_ranges': _first_30s,
        'force_keyframes_at_cuts': True,
        # Skip very short clips (< 5s)
        'match_filter': yt_dlp.utils.match_filter_func('duration >= 5'),
        # Anti rate-limit
        'sleep_interval': 1,
        'max_sleep_interval': 6,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
        print(f"--- Download Complete. Files saved to {download_dir} ---")
    except Exception as e:
        print(f"Error during YouTube download: {e}")


#############################################################################################
def main():
    for genere, query in globals.GENERES.items():
        search_term, number_of_songs_to_download, enabled = query.split("|")
        

        if enabled == '0':
            print(f"Skipping disabled genre: {genere}")
            continue

        print(f"Starting: {genere} (Query: {search_term} | songs={number_of_songs_to_download})")
        download_folder = globals.gen_path(genere)

        if int(number_of_songs_to_download) > 0:
            download_youtube_audio(search_term, download_folder, max_results=number_of_songs_to_download)
        else:
            print(f"--- songs=0 → skip download for {genere} ---")

    # 4. Train the VQ-VAE
    print("      STARTING VQ-VAE TRAINING STAGE       ")
    train_vqvae_main()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"CUDA Memory cleared. Currently allocated: {allocated:.2f}MB")

    print("     STARTING GENERATOR TRAINING STAGE     ")
    train_generator_main()

    print("     TESTING VAE     ")
    for genere, query in globals.GENERES.items():
        search_term, number_of_songs_to_download, enabled = query.split("|")
                
        if enabled == '0':
            continue
        test.test_vqvae_reconstruction(genere)


if __name__ == "__main__":
    main()

