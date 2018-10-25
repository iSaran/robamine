from rlrl_py.algo.util import Plotter

plotter = Plotter('/home/iason/rlrl_logs/ddpg-pendulum/rlrl_py_logger_DDPG_Pendulum-v0_2018.10.25.15.29.23.757280', streams = ['episode', 'batch'])
plotter.plot()
#plotter.plot_episode()
#plotter.plot_epoch()
