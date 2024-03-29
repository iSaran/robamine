# Robamine ![version](https://img.shields.io/badge/version-v0.1-blue.svg)

Python code and OpenAI Gym environments for testing RL algorithms with models of ARL's robots.

**Under heavy development**

Here you can find the [change log](CHANGELOG.md) for all the releases. We use
the [Semantic Versioning Specifications](http://semver.org/) for release
numbering.

The documentation for the latest release will be found in the future
[here](https://auth-arl.github.io/docs/robamine/latest/index.html). For now see
at the end of this README on how to generate the docs locally.

## Installation

### Mujoco

In order to work with `mujoco_py` you should have Mujoco version 1.50 with its license in `~/.mujoco` such that the structure of `~/.mujoco` would be:

```
.mujoco
 |-- mjkey.txt
 |-- mjpro150
     |--bin
     |-- doc
     |-- include
     |-- model
     |-- sample
```

### Install a Python Virtual Env
Install `glew` if not installed:

```bash
sudo apt install libglew-dev
```

Install `python3-tk`:

```bash
sudo apt-get install python3-tk
```

```bash
sudo pip install virtualenv
virtualenv ~/robamine-env --python=python3 --prompt='[robamine env] '
```

At the end of the `~/robamine-env/bin/activate` script, add the following lines:

**If you have Nvidia:**

Check with `nvidia-smi` which nvidia driver is installed (e.g. 384) and replace 384 with your nvidia driver in the lines below:
```bash
# If the virtualenv inherits the `$PYTHONPATH` from your system:
export PYTHONPATH="$VIRTUAL_ENV/lib"

# For using Mujoco:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so
```

**If you do not have Nvidia:**

```bash
# If the virtualenv inherits the `$PYTHONPATH` from your system:
export PYTHONPATH="$VIRTUAL_ENV/lib"

# For using Mujoco:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so
```

Then, activate the environment (you have to activate it each time you want to use it):

```bash
source ~/robamine-env/bin/activate
```

### Install Robamine
Activate your virtual environment and then clone the repository and install the package:

```bash
git clone https://github.com/iSaran/robamine.git
cd robamine
pip install -e .
```

### Using ROS environments

In case you want to use the ROS environments (e.g. `LWRROS-v1`) with ROS kinetic you should install the following packages in order for ROS to work with Python 3:

```bash
sudo apt-get install python3-pip python3-yaml
sudo pip3 install rospkg catkin_pkg
```

Remove from your `activate` script of your environment this line:

```bash
export PYTHONPATH="$VIRTUAL_ENV/lib"
```

And finally, uncomment the commented imports of the ROS environments from `robamine/envs/__init__.py`.

## Run

To run with GUI simply run the following script (recommended):

```bash
python run.py
```

To run without GUI:

```bash
python run.py --no-gui --yml=YAML_FILE
```

## Run unit tests

After major changes run every unit test:

```
python run_tests.py
```

## Generate documentation

Useful information can be found in the documentation which can be generated locally by:


```bash
cd doc
make html
```

It will be build in `doc/_build/index.html`.

## Development

Follow the steps below in order to add a new algorithm or environment:

1. Add the Python class of the agent/environment in `algo` or `env`.
2. Add its name in `yaml/available.yml`, its default values as a yaml file in `yaml/defaults` and value constraints in `yaml/constraints`.
