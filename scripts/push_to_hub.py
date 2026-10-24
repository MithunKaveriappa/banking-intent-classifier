import sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]          # e.g. YOUR_HF_USERNAME/banking77-distilbert
api = HfApi()
api.create_repo(repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="models/banking77_distilbert",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload fine-tuned DistilBERT for Banking77",
)
print(f"Done: https://huggingface.co/{repo_id}")