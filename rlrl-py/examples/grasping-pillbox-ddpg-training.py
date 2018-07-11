import baselines.ddpg.main as main
import rlrl_py

from mpi4py import MPI
from baselines import logger
import argparse

if __name__ == '__main__':
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    args = main.parse_args()
    main.run(**args)
