import io
from setuptools import setup, find_packages

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='treecse',
    packages=['treecse, dmrst_parser'],
    version='0.4',
    license='MIT',
    description='A sentence embedding tool based on TreeCSE',
    install_requires=[
        "tqdm",
        "scikit-learn",
        "scipy>=1.5.4,<1.6",
        "transformers",
        "torch",
        "numpy>=1.19.5,<1.20",
        "setuptools"
    ]
)