from setuptools import setup, find_packages

setup(
    name="gemini-srt-translator",
    version="1.4.0",
    packages=find_packages(),
    install_requires=[
        "google-generativeai==0.8.3",
        "srt==3.5.3",
        "PyQt6>=6.0.0",
    ],
    entry_points={
        'console_scripts': [
            'gemini-srt-translator=gemini_srt_translator:translate',
            'gemini-srt-translator-gui=subtitle-translator-gui:main'
        ],
    },
    author="Matheus Castro",
    description="A tool to translate subtitles using Google Generative AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/maktail/gemini-srt-translator",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Environment :: X11 Applications :: Qt",
    ],
    python_requires='>=3.9',
)
