from setuptools import setup, find_packages

setup(
    name="coco_grape",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "pytorch-lightning>=1.5.0",
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="CoCoGraPE: Conditional Compositional Graph Processing and Evaluation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CoCoGraPE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)