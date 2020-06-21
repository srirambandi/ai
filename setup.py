import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="import-ai",
    version="1.3.9",
    author="Sri Ram Bandi",
    author_email="srirambandi.654@gmail.com",
    description="AI library in python using numpy, with end-to-end reverse auto-differentiable dynamic Computational Graph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/srirambandi/ai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
       "numpy>=1.18.1",
       "graphviz>=0.14"
    ],
    python_requires='>=3.6',
)
