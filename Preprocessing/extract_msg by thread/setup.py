#!/usr/bin/env python3
"""
Setup script for Thread Data Extraction
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="thread-data-extraction",
    version="1.0.0",
    author="Boost RAG Project",
    author_email="",
    description="Extract thread data from mbox and mail JSON files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/thread-data-extraction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Communications :: Email",
        "Topic :: Text Processing :: Markup",
        "Topic :: Utilities",
    ],
    python_requires=">=3.6",
    install_requires=[
        # No external dependencies - uses only Python standard library
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "extract-threads=extract_thread_data:main",
            "simple-extract=simple_thread_extractor:main",
            "test-extraction=test_thread_extraction:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
