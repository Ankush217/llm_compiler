
from setuptools import setup, find_packages

setup(
    name="llmc-decoder-only-0m-relu",
    version="1.0.0",
    author="LLM Compiler",
    description="Generated LLM model",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
)
