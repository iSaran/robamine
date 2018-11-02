from setuptools import setup

setup(name='robamine',
      description='Python codebase with Gym Environments and RL training for AUTh-ARL robots',
      version='0.0.1',
      author='Iason Sarantopoulos',
      author_email='iasons@auth.gr',
      install_requires=['gym', 'tensorflow>=1.11.0', 'sphinx', 'sphinxcontrib-bibtex', 'pandas', 'sphinx_rtd_theme', 'numpydoc', 'matplotlib']
)
