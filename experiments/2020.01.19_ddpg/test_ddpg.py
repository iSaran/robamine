from robamine.algo.core import TrainWorld
from robamine.algo.ddpg_torch import DDPG_TORCH
from robamine import rb_logging
import logging
import yaml

def run():
    with open('params.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    rb_logging.init(directory=params['world']['logging_dir'], friendly_name=params['world']['friendly_name'], file_level=logging.INFO)
    trainer = TrainWorld(agent=params['agent'], env='Pendulum-v0', params=params['world']['params'])
    trainer.run()

if __name__ == '__main__':
    run()
