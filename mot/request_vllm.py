"""Send a test chat-completion request to a local vLLM OpenAI server."""

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).with_name("request_vllm.sh")


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: mot.request_vllm <port>")
    sys.exit(subprocess.run(["bash", str(_SCRIPT), sys.argv[1]]).returncode)


if __name__ == "__main__":
    main()
