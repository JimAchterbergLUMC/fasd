from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fasd",
    version="0.1.2",
    description="Pytorch model for generating fidelity agnostic synthetic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jim Achterberg",
    author_email="j.l.achterberg@lumc.nl",
    packages=find_packages(),
    install_requires=["torch", "pandas", "scikit-learn", "tqdm"],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
