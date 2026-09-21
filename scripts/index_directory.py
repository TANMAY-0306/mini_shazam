import sys
import time
from pathlib import Path
from scripts.index_song import index_audio_file

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def index_directory(folder_path: str = "songs"):
    dir_path = Path(folder_path).resolve()
    
    if not dir_path.is_dir():
        print(f"Error: Directory '{dir_path}' does not exist.")
        return

    # Recursively find all audio files
    audio_files = [
        f for f in dir_path.rglob("*") 
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    total_files = len(audio_files)
    if total_files == 0:
        print(f"No supported audio files ({', '.join(SUPPORTED_EXTENSIONS)}) found in '{dir_path}'.")
        return

    print(f"\n=======================================================")
    print(f" Found {total_files} audio file(s) in: {dir_path}")
    print(f"=======================================================\n")

    indexed_count = 0
    skipped_count = 0
    failed_count = 0

    start_total_time = time.time()

    for idx, file_path in enumerate(audio_files, start=1):
        print(f"[{idx}/{total_files}] Processing: {file_path.name}")
        t0 = time.time()
        try:
            song_id = index_audio_file(str(file_path))
            elapsed = time.time() - t0
            
            if song_id is not None:
                indexed_count += 1
                print(f"-> Done in {elapsed:.2f}s (Song ID: {song_id})\n")
            else:
                skipped_count += 1
                print(f"-> Skipped (Already indexed or unreadable)\n")
        except Exception as e:
            failed_count += 1
            print(f"-> Failed: {e}\n")

    total_duration = time.time() - start_total_time
    print("=======================================================")
    print("Batch Indexing Complete Summary:")
    print(f"  • Total Discovered: {total_files}")
    print(f"  • Successfully Added: {indexed_count}")
    print(f"  • Skipped (Duplicates): {skipped_count}")
    print(f"  • Errors/Failed: {failed_count}")
    print(f"  • Total Time Elapsed: {total_duration:.2f}s")
    print("=======================================================")


if __name__ == "__main__":
    target_folder = sys.argv[1] if len(sys.argv) > 1 else "songs"
    index_directory(target_folder)