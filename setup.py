from setuptools import setup, find_packages
setup(
  name = 'torchpq',
  packages = find_packages(),
<<<<<<< HEAD
  version = '0.3.0.0',
=======
  version = '0.2.0.2',
>>>>>>> e99c31ab31ef024f2f9cfee30338afd437baa2f5
  license='MIT',
  description = 'Efficient implementations of Product Quantization and its variants',
  author = 'demoriarty', 
  author_email = 'sahbanjan@gmail.com',
  url = 'https://github.com/DeMoriarty/TorchPQ',
<<<<<<< HEAD
  download_url = 'https://github.com/DeMoriarty/TorchPQ/archive/v_0300.tar.gz',
=======
  download_url = 'https://github.com/DeMoriarty/TorchPQ/archive/v_0202.tar.gz',
>>>>>>> e99c31ab31ef024f2f9cfee30338afd437baa2f5
  keywords = ['KMeans', 'K-means', 'ANN', 'pytorch','machine learning', 'pq', 'product quantization', 'IVFPQ', 'approximate nearest neighbors'],
  install_requires=[ 
    'numpy',
    'torch',
  ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
  include_package_data = True,
)