import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='torch_arima',
    version='0.0.3',
    long_description=long_description,
    long_description_content_type="text/markdown",
    description='ARIMA time series implementation in PyTorch and Pyro',
    install_requires=['pyro-ppl>=1.9.1'],
    python_requires='>=3.8',
    packages=setuptools.find_packages(),
    url='https://github.com/BenZickel/torch_arima',
    author='Ben Zickel',
    license='BSD')
