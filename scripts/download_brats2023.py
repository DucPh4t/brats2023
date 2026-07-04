import os
import sys
from huggingface_hub import snapshot_download

def main():
    repo_id = "Angelou0516/brats2023-gli-dataset"
    local_dir = "/Users/nguyenducphat/Projects/ĐATN MRI/MRI dataset/BraTS2023_GLI"
    
    print(f"Starting download of dataset '{repo_id}'...")
    print(f"Target directory: {local_dir}")
    
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # Download using snapshot_download
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            max_workers=4
        )
        print("\n" + "="*50)
        print("🎉 Download completed successfully!")
        print(f"Data saved to: {downloaded_path}")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
