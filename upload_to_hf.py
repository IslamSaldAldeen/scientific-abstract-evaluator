from huggingface_hub import HfApi

api = HfApi()

api.create_repo(
    repo_id = "IslamSaadAldeen/scientific-abstract-evaluator-lora",
    repo_type="model",
    exist_ok=True,
)

api.upload_folder(
    folder_path="hf_upload/scientific-abstract-evaluator-lora",
    repo_id = "IslamSaadAldeen/scientific-abstract-evaluator-lora",
    repo_type="model",
)

print("Upload completed successfully!")
