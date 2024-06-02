from setuptools import setup,find_packages

requirements=[r.strip() for r in open("requirements.txt").readlines()]

setup(
   name='asr2k',
   version='0.0.1',
   description='a speech model',
   author='Xinjian Li',
   author_email='xinjianl@cs.cmu.edu',
   url="https://github.com/xinjli/asr2k",
   packages=find_packages(),
   package_data={'': ['*.csv', '*.tsv', '*.yml']},
   install_requires=requirements,
)