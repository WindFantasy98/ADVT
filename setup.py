import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="advt",
    version="1.0.2",
    author="WindF98",
    author_email="wwj98713@163.com",
    description="A library for Attack & Defence on Video Task",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WindFantasy98/ADVT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
    ],
    python_requires='>=3.6',
)