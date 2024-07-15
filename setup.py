import setuptools

# Developer self-reminder for uploading in pypi:
# - install: wheel, twine
# - build  : python setup.py bdist_wheel
# - deploy : twine upload dist/*
# https://kynan.github.io/blog/2020/05/23/how-to-upload-your-package-to-the-python-package-index-pypi-test-server


with open("README.md", "r") as file:
    long_description = file.read()

with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()

setuptools.setup(
    name="MAMMOth-commons",
    version="0.0.19",
    author="Emmanouil (Manios) Krasanakis",
    author_email="maniospas@hotmail.com",
    description="Component interfaces of the MAMMOth fairness toolkit.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mammoth-eu/mammoth-commons",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
)
