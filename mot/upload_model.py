"""Upload a local model directory to the Hugging Face Hub."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Upload a local model directory to the Hugging Face Hub",
        epilog="MODEL_STUB is the target Hugging Face repo id such as "
               "RedHatAI/DeepSeek-V4-Flash-0731-NVFP4; MODEL_PATH is the "
               "local directory of model files to upload",
    )
    parser.add_argument("model_stub", metavar="MODEL_STUB",
                        help="target Hugging Face repo id")
    parser.add_argument("model_path", metavar="MODEL_PATH",
                        help="local model directory to upload")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        folder_path=args.model_path,
        repo_id=args.model_stub,
        repo_type="model",
    )


if __name__ == "__main__":
    main()
