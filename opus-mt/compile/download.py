from huggingface_hub import snapshot_download
snapshot_download(repo_id="Helsinki-NLP/opus-mt-zh-en", ignore_regex=["*.h5", "*.ot"])
