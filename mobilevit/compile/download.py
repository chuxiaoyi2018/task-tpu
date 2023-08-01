from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id="apple/mobilevit-small", ignore_patterns=["*.h5", "*.ot", "*.mlpackage"], local_dir=os.getcwd() + "/tmp")
