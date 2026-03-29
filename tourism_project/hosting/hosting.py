from huggingface_hub import HfApi
import os

# Initialize Hugging Face API with authentication token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the deployment folder to Hugging Face Space
# This includes: Dockerfile, app.py, and requirements.txt
api.upload_folder(
    folder_path="tourism_project/deployment",     # Local folder containing deployment files
    repo_id="Manish9119/tourism-package-prediction",  # Target Hugging Face Space
    repo_type="space",                            # Repository type is 'space'
    path_in_repo="",                              # Upload to root of the space
)

print("Deployment files uploaded successfully to Hugging Face Space!")
