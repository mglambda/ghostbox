from setuptools import setup, find_packages
import os


GHOSTBOX_VERSION='0.18.3'


with open("README.md", "r", encoding="utf-8") as readme_file:
    README = readme_file.read()

setup(
    name='ghostbox',
    version=GHOSTBOX_VERSION,
    url="https://github.com/mglambda/ghostbox",
    author="Marius Gerdes",
    author_email="integr@gmail.com",
    description="Ghostbox is a command line interface to local LLM (large language model) server applications, such as koboldcpp or llama.cpp. It's primary purpose is to give a simple, stripped-down way to engage with AI chat bots.",
    long_description=README,
    long_description_content_type="text/markdown",
    license_files=["LICENSE"],
    package_data = {"ghostbox" : ["data/*.wav"]},
    scripts=["scripts/ghostbox", "scripts/ghostbox-tts-polly", "scripts/ghostbox-aws-client", "scripts/ghostbox-install", "scripts/ghostbox-tts-spd-say", "scripts/ghostbox-tortoise-loop", "scripts/ghostbox-tts-tortoise", "scripts/ghostbox-tts"],
    packages=find_packages(include=['ghostbox']),
    install_requires=["requests", "boto3", "appdirs", "lazy-object-proxy", "openai-whisper", "pyaudio", "pydub", "colorama", "automodinit", "deepspeed", "docstring_parser", "jsonpickle", "shutils", "moviepy2", "nltk", "wget", "kokoro_onnx[gpu]", "websockets"] 
)

