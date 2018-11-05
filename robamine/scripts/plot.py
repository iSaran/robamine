from robamine.algo.util import Plotter

plotter = Plotter('/home/iason/robamine_logs/ddpg-sphere-reacher/robamine_logs_2018.11.05.18.51.38.336344/DDPG_SphereReacherShapedReward-v1', streams = ['eval_episode', 'eval_batch', 'train_episode', 'train_batch'])
plotter.plot()
#plotter.plot_episode()
#plotter.plot_epoch()
