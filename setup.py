import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='torch_arima',
    version='0.0.2',
    long_description=long_description,
    long_description_content_type="text/markdown",
    description='ARIMA time series implementation in PyTorch and Pyro',
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    author='Ben Zickel',
    license='BSD')
