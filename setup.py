from setuptools import setup, find_packages
import os


GHOSTBOX_VERSION='0.7.0'


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
    scripts=["scripts/ghostbox", "scripts/ghostbox-tts-polly", "scripts/ghostbox-install"],
    packages=find_packages(include=['ghostbox']),
    install_requires=["requests","requests_html","boto3", "appdirs", "lazy-object-proxy", "pygame", "openai-whisper", "pyaudio", "pydub"] 
)

