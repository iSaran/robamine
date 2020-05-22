import numpy as np

class FeatureBase:
    def __init__(self, name=None):
        self.name = name

    def rotate(self, angle):
        raise NotImplementedError()

    def array(self):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()

class PushActionBase:
    def __init__(self):
        pass

    def get_init_pos(self):
        raise NotImplementedError()

    def get_final_pos(self):
        raise NotImplementedError()

    def get_duration(self, distance_per_sec=0.1):
        return np.linalg.norm(self.get_init_pos() - self.get_final_pos()) / distance_per_sec

    def plot(self):
        raise NotImplementedError()

