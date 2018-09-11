from gym.envs.registration import register

# A sphere is placed on a table and tries to reach random position goals on the
# table. The actions are applied forces on the sphere and the state the goal,
# the current position and the current velocity of the sphere.
register(
    id='SphereReacher-v1',
    entry_point='rlrl_py.envs:SphereReacher',
    max_episode_steps=100
)

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

register(
    id='BHandSlidePillbox-v3',
    entry_point='rlrl_py.envs:BHandSlidePillbox3',
    max_episode_steps=2000
)

register(
    id='FingerSlide-v1',
    entry_point='rlrl_py.envs:FingerSlide',
    max_episode_steps=2000
)

register(
    id='SpherePosition-v1',
    entry_point='rlrl_py.envs:SpherePosition',
    max_episode_steps=50
)
