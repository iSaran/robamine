from setuptools import setup

setup(name='gym_rlrl',
      description='Gym Environments for AUTh-ARL robots',
      version='0.0.1',
      author='Iason Sarantopoulos',
      author_email='iasons@auth.gr',
      install_requires=['gym', 'mujoco-py', 'baselines']  # And any other dependencies foo needs
)
