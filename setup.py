

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

with open('requirements.txt') as f:
    requires = [l.strip() for l in f.readlines()]

setup(
    name="genprompt",  
    version="0.0.1",  
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",  
    url="",  
    author="Isabel Cachola & Neha Verma", 
    author_email="",  
    keywords="",  
    packages=find_packages(), 
    python_requires=">=3.7, <4",
    install_requires=requires
)