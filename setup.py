try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Convolutional restricted Boltzmann machine for learning DNA sequence features',
    'author': ['Roman Schulte-Sasse', 'Wolfgang Kopp'],
    'url': 'https://github.molgen.mpg.de/wkopp/crbm',
    'download_url': 'https://github.molgen.mpg.de/wkopp/crbm',
    'author_email': ['sasse@molgen.mpg.de','kopp@molgen.mpg.de'],
    'version': '0.1',
    'install_requires': ['numpy','Biopython','pandas', 'sklearn','Theano',
        'joblib','matplotlib'],
    'packages': ['crbm'],
    'package_data': {'crbm':['data/*.fa']},
    'name': 'crbm'
}

setup(**config)

