from setuptools import find_packages, setup

setup(
    name="cantonese-asr-benchmark",
    version="0.1.0",
    description="Fine-tuning Whisper for Hong Kong Cantonese ASR with LoRA",
    author="Zhou Bojian",
    url="https://github.com/zhoubojian-stevenchow/cantonese-asr-benchmark",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers>=4.36.0",
        "datasets==2.21.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pycantonese>=3.4.0",
        "jiwer>=3.0.0",
        "evaluate>=0.4.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "opencc-python-reimplemented>=0.1.7",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
    },
)
