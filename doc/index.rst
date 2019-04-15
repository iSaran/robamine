.. robamine documentation master file, created by
   sphinx-quickstart on Tue Sep 25 12:10:46 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Main
====

.. note::
   This package has been tested only in Ubuntu 16.04 and Python 3.

.. warning::
   Be aware that this package is under heavy development, which means that
   besides the code is currently unstable, the documentation may not be
   complete or match the actual functionality.

`Robamine` (the dopamine of robots) is a python package for testing Reinforcement Learning algorithms (RL) for robotic tasks focusing on robotic grasping and manipulation. It uses `OpenAI Gym <http://www.github.com/openai/gym>`_ for using environments and `Tensorflow <http://www.tensorflow.org>`_ for deep learning. Due to the focusing on robotic tasks all the environments are using the `MuJoCo <http://www.mujoco.org>`_ physics engine.

The development of this package aims for implementation of state of the art RL algorithms in order to perform research in robotics. In order to accomplish that the following requirements should be met:

  * The implementations should be reflect the algorithms proposed in the papers.
  * The implementation of an RL algorithm should be seperated from other technical implementation details. This will ensure that the first requirement is met.
  * The code should be clean and well documented.

The packages consists of two main subpackages:

  * ``algo``: Implementation of RL algorithms.
  * ``env``: Definition of different Gym environments.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :caption: Contents:

   self
   install
   examples
   api/algo/main
   api/env/main
   api/utils/main
   zreferences
