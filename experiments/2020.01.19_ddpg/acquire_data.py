from robamine.algo.core import SampleTransitionsWorld
from robamine.algo.splitddpg import SplitDDPG
from robamine import rb_logging
import logging
import yaml

def run():
    with open('data_acquisition.yml', 'r') as stream:
        params = yaml.safe_load(stream)
    rb_logging.init(directory=params['world']['logging_dir'], friendly_name=params['world']['friendly_name'], file_level=logging.INFO)
    SampleTransitionsWorld(agent='RandomHybrid', env=params['env'], params=params['world']['params'], name='SampleTransitionsWorld').run()

if __name__ == '__main__':
    run()
