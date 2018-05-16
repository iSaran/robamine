"""
The code was taken and customised from openai/baselines/ddpg/main.py
"""

import os, argparse, time

# Inport Baselines utilities
from baselines import logger, bench
from baselines.common.misc_util import boolean_flag, set_global_seeds

# Import DDPG modules
from baselines.ddpg.noise import *
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
import baselines.ddpg.training as training

from mpi4py import MPI
import gym, gym_rlrl
import tensorflow as tf

def run(evaluation, noise_type, layer_norm, seed, **kwargs):
    """ Runs the DDPG algorithm for the Floating BHand Environment """
    # Configure things
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs
    env = gym.make('Floating-BHand-v0')
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    if evaluation and rank==0:
        eval_env = gym.make(env_id)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Initialize noise objects
    action_noise, param_noise = parse_noise_type(noise_type=noise_type, nb_actions=env.action_space.shape[-1])

    # Initialize replay buffer, and actor critic
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions=env.action_space.shape[-1], layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()

    # Train
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)

    # Terminate the program
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))

def parse_noise_type(noise_type, nb_actions):
    """ Parse the noise type given as an argument and initializes the relevant DDPG noise objects. """
    action_noise = None
    param_noise = None
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('Unknown noise type given as argument "{}". See -h for available options.'.format(current_noise_type))
    return action_noise, param_noise

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Define the parameters

    boolean_flag(parser, 'evaluation',
                 default=False,
                 help='Flag for running evaluation.')

    parser.add_argument('--noise-type',
                        type=str,
                        default='adaptive-param_0.2',
                        help="""The type of parameter noise. For parameter noise read plappert17
                                The suffix xx is the standar deviation of the noise.
                                Choices are:
                                1. adaptive_param_xx: For adaptive noise in parameter space.
                                2. normal_xx: For using gaussian noise on the action space (uncorrelated).
                                3. ou_xx: For using the Ornstein-Uhlenbeck process (correlated) on action space.""")

    boolean_flag(parser, 'layer-norm',
                 default=True,
                 help='Flag if you want to use a normalization layer in the networks of actor and critic for normalizing input data. Useful if your observation have for instance position and velocity, quantities with different scaling.')

    parser.add_argument('--seed',
                         type=int,
                         default=0,
                         help='RNG seed')

    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--num-timesteps', type=int, default=None)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args

if __name__ == '__main__':
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    args = parse_args()
    run(**args)
