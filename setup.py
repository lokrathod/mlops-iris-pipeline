# setup.py
from setuptools import setup, find_packages

setup(
    name="iris-mlops-pipeline",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    python_requires='>=3.8',
)