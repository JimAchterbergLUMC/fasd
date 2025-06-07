from setuptools import setup, find_packages

setup(
    name="fasd",
    version="0.1.0",
    description="Pytorch model for generating fidelity agnostic synthetic data",
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
