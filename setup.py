from setuptools import setup, find_packages

setup(
  name = 'mlmxl-pytorch',
  packages = find_packages(),
  version = '0.0.5',
  license='MIT',
  description = 'MLM (Masked Language Modeling) - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/napoler/mlmxl-pytorch',
  keywords = [
    'transformers',
    'artificial intelligence',
    'pretraining',
    'unsupervised learning'
  ],
  install_requires=[
    'torch>=1.1.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
