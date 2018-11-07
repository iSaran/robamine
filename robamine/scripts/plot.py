from robamine.algo.util import Plotter


if __name__ == '__main__':
    rb_logging.init()
    directory = '/home/iason/robamine_logs/ddpg-pendulum/_robamine_logs_2018.11.07.17.16.55.875154/DDPG_Pendulum-v0'
    Plotter.create_batch_from_stream(directory, 'train', 5)
    plotter = Plotter('/home/iason/robamine_logs/ddpg-pendulum/_robamine_logs_2018.11.07.17.16.55.875154/DDPG_Pendulum-v0', ['train', 'batch_train'])
    plotter.plot()

