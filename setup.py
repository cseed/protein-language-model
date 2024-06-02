from setuptools import setup, find_packages

setup(
    name='protein-language-model',
    version='0.1.0',
    description='A package for running AI experiments',
    packages=find_packages(),
    install_requires=[
        'torch',
        'google-cloud-storage',
        'numpy',
        'matplotlib',
        'flash-attn',
        'xmljson',
    ],
    python_requires='>=3.7',
)
