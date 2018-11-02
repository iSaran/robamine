..

.. toctree::

Examples
========

In `robamine/examples` directory you can find some examples to run. For example there for training and evaluating the DDPG algorithm with the Pendulum environment.

.. code-block:: python

   from robamine.algo.ddpg import DDPG

   import tensorflow as tf


   if __name__ == '__main__':
       with tf.Session() as sess:
           agent = DDPG(sess, 'Pendulum-v0', exploration_noise_sigma=0.2, log_dir='/home/iason/robamine_logs/ddpg-pendulum').train(n_episodes=1000, episode_batch_size=25, render=False, episodes_to_evaluate=15, render_eval = True)

This will train DDPG for 1000 episodes and will compile stats for a batch of 25
episodes. Furhermore it will evaluate the learned policy for 15 episodes every
time it collects epidode batch stats (every 25 episodes of training).

If you want to monitor progress during you can use tensorboard in another terminal:

.. code-block:: bash

   tensorboard --logdir=LOGDIR

where `LOGDIR` the log directory printed out by the program during start.

Finally, in the log directory, there are log files for plotting the results as well as a log file of messages (debugs, warnings) for debugging purposes.
