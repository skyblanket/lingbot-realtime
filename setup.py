"""
Setup script for LingBot-World Real-Time
Installs the causal model package
"""

from setuptools import setup, find_packages

setup(
    name="lingbot-causal",
    version="0.1.0",
    description="Real-time streaming LingBot-World with block-wise causal attention",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.0",
        "diffusers>=0.32.0",
        "flash-attn>=2.5.0",
        "transformers",
        "accelerate",
        "einops",
        "numpy",
        "Pillow",
    ],
    python_requires=">=3.10",
)
