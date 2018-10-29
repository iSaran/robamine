from rlrl_py.algo.util import Plotter

plotter = Plotter('/home/iason/rlrl_logs/ddpg-pendulum/rlrl_py_logger_DDPG_Pendulum-v0_2018.10.29.11.17.12.448170', streams = ['eval_episode', 'eval_batch', 'train_episode', 'train_batch'])
plotter.plot()
#plotter.plot_episode()
#plotter.plot_epoch()
