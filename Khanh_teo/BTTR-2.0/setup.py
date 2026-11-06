#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="bttr",
    version="0.0.1",
    description="Handwritten Mathematical Expression Recognition with Bidirectionally Trained Transformer",
    author="Wenqi Zhao",
    author_email="1027572886a@gmail.com",
    url="https://github.com/Green-Wood/BTTR",
    install_requires=[
        "pytorch-lightning",
        "einops",
        "editdistance",
    ],
    packages=find_packages(),
)
