from setuptools import setup, find_packages

setup(
    name="beam_search",
    version="0.1",
    packages=find_packages(),
    description="A custom beam search decoding algorithm for Pytorch",
    author="Balthazar Bujard",
    author_email="balthazar.bujard@ircam.fr",
    install_requires=[
        "numpy",
        "torch"
    ],
)