import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()


setuptools.setup(
    name="flamecalc",
    version="0.1.0",
    author="SeBeom Lee",
    author_email="slee5@oberlin.edu",
    description="Pytorch based approach to calculus of variation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/k2sebeom/flamecalc",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)