from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme_file:
    README = readme_file.read()

setup(
    name='ghostbox',
    version='0.1.0',
    url="https://github.com/mglambda/ghostbox",
    author="Marius Gerdes",
    author_email="integr@gmail.com",
    description="Ghostbox is a command line interface to local LLM (large language model) server applications, such as koboldcpp or llama.cpp. It's primary purpose is to give a simple, stripped-down way to engage with AI chat bots.",
    long_description=README,
    long_description_content_type="text/markdown",
    scripts=["scripts/ghostbox", "scripts/ghostbox-tts-polly"],
    packages=find_packages(include=['ghostbox'])
)
