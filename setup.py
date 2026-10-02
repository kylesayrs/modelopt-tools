from setuptools import setup

setup(
    name="modelopt-tools",
    version="0.1.0",
    description="Tools for inspecting and working with model checkpoints",
    packages=["mot"],
    entry_points={
        "console_scripts": [
            "mot.inspect_tensors = mot.inspect_tensors:main",
        ],
    },
    install_requires=["huggingface_hub"],
    python_requires=">=3.9",
)
