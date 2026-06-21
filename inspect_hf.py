from huggingface_hub import HfApi, hf_hub_download
import json

api = HfApi()
files = api.list_repo_files(repo_id="FaisaI/tadabur", repo_type="dataset")
for f in files:
    print(f)
