try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from sphinx.setup_command import BuildDoc

cmdclass = {'build_sphinx': BuildDoc}

name = "cRBM"
version = "1.0"
release = "1.0.0"

config = {
    'description': 'Convolutional restricted Boltzmann machine for learning DNA sequence features',
    'author': ['Roman Schulte-Sasse', 'Wolfgang Kopp'],
    'url': 'https://github.molgen.mpg.de/wkopp/crbm',
    'download_url': 'https://github.molgen.mpg.de/wkopp/crbm',
    'author_email': ['sasse@molgen.mpg.de','kopp@molgen.mpg.de'],
    'version': version,
    'install_requires': ['numpy','Biopython','pandas', 'sklearn','Theano',
        'joblib','matplotlib', 'weblogo', 'seaborn'],
    'packages': ['crbm'],
    'tests_require': ['pytest'],
    'setup_requires': ['pytest-runner'],
    'package_data': {'crbm':['data/oct4.fa']},
    'zip_safe': False,
    'command_options': {
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release)}},
    'name': name
}

setup(**config)

