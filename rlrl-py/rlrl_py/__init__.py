from gym.envs.registration import register

register(
    id='Floating-BHand-v0',
    entry_point='rlrl_py.envs:FloatingBHand'
)

register(
    id='BHand-Slide-Pillbox-v0',
    entry_point='rlrl_py.envs:BHandSlidePillbox'
)

register(
    id='BHandSlidePillbox-v2',
    entry_point='rlrl_py.envs:BHandSlidePillbox2',
    max_episode_steps=2000
)
