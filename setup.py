from setuptools import setup, find_packages

setup(
    name="ensemble_framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.6.0"
    ],
    author="Your Name",
    author_email="markgewhite@gmail.com",
    description="A framework for ensemble learning with patient-level predictions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)