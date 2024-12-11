from setuptools import setup

exec(open("empdens/version.py").read())

version = {}
with open("empdens/version.py") as fp:
    exec(fp.read(), version)
#
# try:
#     import shmistogram
# except:
#     raise Exception("You first need to install shmistogram; try "
#         + "`pip install git+https://github.com/zkurtz/shmistogram.git#egg=shmistogram`")

# try:
#     import cython
# except:
#     raise Exception("You must first install Cython; `pip install Cython`")
setup(
    name='empdens',
    version=version['__version__'],
    packages=['empdens',
        'empdens.classifiers',
        'empdens.evaluation',
        'empdens.models',
        'empdens.simulators',
        'empdens.wrappers'
    ],
    install_requires=[
        'scikit-learn',
        'lightgbm',
        'pandas',
        'psutil',
        'shmistogram', # or 'shmistogram @ git+https://github.com/zkurtz/shmistogram.git#egg=shmistogram'
    ],
    package_dir={'empdens': 'empdens'},
    package_data={'empdens': ['resources/data/japanese_vowels.csv']},
    license='See LICENSE.txt'
)
