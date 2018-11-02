# Robamine ![version](https://img.shields.io/badge/version-v0.0.0-blue.svg) 

**Under heavy development**

Here you can find the [change log](CHANGELOG.md) for all the releases. We use
the [Semantic Versioning Specifications](http://semver.org/) for release
numbering.

The documentation for the latest release can be found
[here](https://auth-arl.github.io/docs/robamine/latest/index.html). For the
latest master you need to produce it locally with `doxygen Doxyfile` inside the
repository. Then see `docs/html/index.html`.

## Contents

* [robabmine](robamine): Python Gym Environments for our robots and testing RL in MuJoCo

Python code and OpenAI Gym environments for testing RL algorithms with models of ARL's robots.

## Installation

### Install a Python Virtual Env

```bash
sudo pip install virtualenv
mkdir ~/openai-py/env
virtualenv ~/openai-py/env --python=python3
```

If the virtualenv inherits the `$PYTHONPATH` from your system add this in the `bin/activate` script:

```
export PYTHONPATH="$VIRTUAL_ENV/lib"
```

For using Mujoco do:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384
```


Then, activate the environment (you have to activate it each time you want to use it):

```bash
source ~/openai-py/env/bin/activate
```

### Install OpenAI baselines

```bash
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
```

Install mujoco1.50 in `~/.mujoco/mjpro150` with the licence key and

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mjpro150/bin
```

Run examples to test the installation:

```bash
cd ~/openai-py/baselines/baselines/trpo_mpi
python -m baselines.trpo_mpi.run_mujoco
```

If no module cv2 found install opencv:

```bash
pip install opencv-python
```

If osmesa error occurs

```bash
sudo apt-get install libosmesa6-dev
```

Install patchelf if not found:
Download it from https://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.gz and:

```bash
./configure
make
sudo make install
```

If you get

```
ERROR: GLEW initalization error: Missing GL version
```

Run your file with:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so python your_file.py
```

### Install Robamine
Clone the repository and install the package:

```bash
cd robamine/robamine
pip install -e .
```

## Run examples

You can train the grasping of pillbox using DDPG with:

```bash
cd robamine/examples
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so python grasping-pillbox-ddpg-training.py --env-id='Floating-BHand-v0'
```

Plotting the results using the log directory:

```bash
cd baselines/baselines
python results_plotter.py --dirs=/tmp/openai-2018-07-11-14-55-45-282417

```

#  Generate documentation

Useful information can be found in the documentation which can be generated locally by:

Then build the documentation (in will be build in `doc/_build/index.html`):

```bash
cd doc
make html
```
