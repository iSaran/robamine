from gym.envs.registration import register

register(
    id='Floating-BHand-v0',
    entry_point='rlrl_py.envs:FloatingBHand'
)

register(
    id='BHand-Slide-Pillbox-v0',
    entry_point='rlrl_py.envs:BHandSlidePillbox'
)
