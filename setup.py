from setuptools import setup

exec(open("pydens/version.py").read())

version = {}
with open("pydens/version.py") as fp:
    exec(fp.read(), version)

try:
    import shmistogram
except:
    raise Exception("You first need to install shmistogram; try "
        + "`pip install git+https://github.com/zkurtz/shmistogram.git#egg=shmistogram`")

try:
    import cython
except:
    raise Exception("You must first install Cython; `pip install Cython`")
setup(
    name='pydens',
    version=version['__version__'],
    packages=['pydens',
        'pydens.classifiers',
        'pydens.models',
        'pydens.simulators',
        'pydens.wrappers'
    ],
    install_requires=[
        'scikit-learn',
        'fastkde',
        'lightgbm',
        'pandas',
        'psutil',
        'shmistogram @ git+https://github.com/zkurtz/shmistogram.git#egg=shmistogram'
    ],
    license='See LICENSE.txt'
)
