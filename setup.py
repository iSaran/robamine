from setuptools import setup

setup(name='robamine',
      description='Python codebase with Gym Environments and RL training for AUTh-ARL robots',
      version='0.0.1',
      author='Iason Sarantopoulos',
      author_email='iasons@auth.gr',
      install_requires=['tensorflow==1.12.0', \
                        'gym==0.12.1', \
                        'mujoco-py==2.0.2.2', \
                        'sphinx', \
                        'sphinxcontrib-bibtex', \
                        'pandas==0.24.2', \
                        'sphinx_rtd_theme', \
                        'numpydoc', \
                        'matplotlib',\
                        'opencv-python==4.0.1.24', \
                        'open3d-python==0.6.0.0']
)
