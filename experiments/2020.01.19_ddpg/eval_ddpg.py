from robamine.algo.core import EvalWorld
# from robamine.algo.ddpg_torch import DDPG_TORCH
from robamine.algo.splitddpg import SplitDDPG
from robamine import rb_logging
import logging
import yaml

def run():
    with open('params.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    rb_logging.init(directory=params['world']['logging_dir'], friendly_name=params['world']['friendly_name'], file_level=logging.INFO)
    trainer = EvalWorld.load('/home/mkiatos/robamine/logs/robamine_logs_2020.01.31.11.12.01 ')
    trainer.run()

if __name__ == '__main__':
    run()
