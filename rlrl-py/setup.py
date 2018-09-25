from setuptools import setup

setup(name='rlrl_py',
      description='Python codebase with Gym Environments and RL training for AUTh-ARL robots',
      version='0.0.1',
      author='Iason Sarantopoulos',
      author_email='iasons@auth.gr',
      install_requires=['gym', 'mujoco-py', 'tensorflow', 'sphinx', 'sphinxcontrib-bibtex']
)
