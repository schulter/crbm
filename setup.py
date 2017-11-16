try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

#from sphinx.setup_command import BuildDoc

#cmdclass = {'build_sphinx': BuildDoc}

name = "secomo"
exec(open('secomo/version.py').read())

config = {
    'description': 'SECOMO: Using convolutional restricted Boltzmann machines \
    to model DNA sequence features and contexts',
    'author': ['Roman Schulte-Sasse', 'Wolfgang Kopp'],
    'url': 'https://github.com/schulter/crbm',
    'download_url': 'https://github.com/schulter/crbm',
    'author_email': ['sasse@molgen.mpg.de','kopp@molgen.mpg.de'],
    'version': __version__,
    'install_requires': ['numpy', 'Biopython', 'pandas', 'sklearn','Theano',
        'joblib','matplotlib', 'weblogo', 'seaborn'],
    'packages': ['secomo'],
    'tests_require': ['pytest'],
    'setup_requires': ['pytest-runner'],
    'package_data': {'secomo':['data/oct4.fa']},
    'zip_safe': False,
    #'command_options': {
        #'build_sphinx': {
            #'project': ('setup.py', name),
            #'version': ('setup.py', version),
            #'release': ('setup.py', release)}},
    'name': name
}

setup(**config)
