from huggingface_hub import HfApi, create_repo, upload_folder
import os

# --- CONFIGURATION ---
HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set HUGGING_FACE_TOKEN environment variable")

repo_id = "anudeep12k/glacuoma_tiny_bert"
local_model_path = "/Users/anudeep/Documents/glaucoma_detection/models/transformer_tiny-biobert"
# --------------------

api = HfApi()

# 1. Create repo (skip if already exists)
'''create_repo(
    name=repo_id.split("/")[1],
    token=HF_TOKEN,
    repo_type="model",
    exist_ok=True
)
'''

# 2. Upload entire folder
upload_folder(
    folder_path=local_model_path,
    path_in_repo="",        # upload into root of repo
    repo_id=repo_id,
    repo_type="model",
    token=HF_TOKEN,
)

print("Upload complete!")
print(f"Your model is at: https://huggingface.co/{repo_id}")