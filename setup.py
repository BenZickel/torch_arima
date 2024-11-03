
import setuptools, subprocess

# Baseline version identifier
version='0.0.6'

# Try to add commit hash to version
try:
    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    version += f"+{commit_sha}"
# catch all exception to be safe
except Exception:
    pass  # probably not a git repo

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='torch_arima',
    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    description='ARIMA time series implementation in PyTorch and Pyro',
    install_requires=['pyro-ppl>=1.9.1'],
    python_requires='>=3.8',
    packages=setuptools.find_packages(),
    url='https://github.com/BenZickel/torch_arima',
    author='Ben Zickel',
    license='BSD')
