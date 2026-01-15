"""
Setup configuration for autgrad package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="autgrad",
    version="1.0.0",
    author="Neural Network Enthusiast",
    description="An automatic differentiation engine built from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "graphviz>=0.13.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "matplotlib>=3.3.0",
            "jupyter>=1.0.0",
        ],
    },
)
