..

.. toctree::

Installation
============

Install a Python Virtual Env
----------------------------

.. code-block:: bash

   sudo pip install virtualenv
   mkdir ~/openai-py/env
   virtualenv ~/openai-py/env --python=python3

If the virtualenv inherits the `$PYTHONPATH` from your system add this in the `bin/activate` script:


.. code-block:: bash

   export PYTHONPATH="$VIRTUAL_ENV/lib"

For using Mujoco do:

.. code-block:: bash

   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384

Then, activate the environment (you have to activate it each time you want to use it):

.. code-block:: bash

 source ~/openai-py/env/bin/activate

Install OpenAI baselines
------------------------

.. code-block:: bash

   # Install dependencies
   sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

   # Install Baselines
   cd ~/openai-py
   git clone https://github.com/openai/baselines.git
   cd baselines
   pip install -e .

   # Verify the installation by running the tests
   pip install pytest
   pytest


Install mujoco1.50 in `~/.mujoco/mjpro150` with the licence key and

.. code-block:: bash

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mjpro150/bin

Run examples to test the installation:

.. code-block:: bash

   cd ~/openai-py/baselines/baselines/trpo_mpi
   python -m baselines.trpo_mpi.run_mujoco

Possible Issues
~~~~~~~~~~~~~~~

If no module cv2 found install opencv:

.. code-block:: bash

   pip install opencv-python

If osmesa error occurs

.. code-block:: bash

   sudo apt-get install libosmesa6-dev

Install patchelf if not found:

Download it from https://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.gz and:

.. code-block:: bash

   ./configure
   make
   sudo make install

If you get

.. code-block:: bash

   ERROR: GLEW initalization error: Missing GL version


Run your file with:

.. code-block:: bash

   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so python your_file.py

Install rlrl
------------
Clone the repository and install the package:

.. code-block:: bash

   cd rlrl/rlrl-py
   pip install -e .
