from setuptools import setup, find_packages

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="href",
    version="0.1.0",
    author="Xinxi Lyu",
    author_email="xinxil@allenai.org",
    description="Human Reference Guided Evaluation for Instruction Following",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='==3.10',
    entry_points={
        "console_scripts": [
            "evaluate=href.evaluate:evaluate.main",
        ],
    }
)