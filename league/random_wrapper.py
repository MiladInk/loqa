import dataclasses

import flax.struct as struct

from league.agent import get_cooperative_action
from utils import rscope

import jax.random as rax

@struct.dataclass
class RandomAgent:
  player: int = struct.field(pytree_node=False)

  def get_action(self, **kwargs):
    rng = kwargs.get('rng')
    return rax.randint(key=rng, minval=0, maxval=4, dtype=int, shape=[]) # TODO: use g_num_actions

@dataclasses.dataclass
class RandomAgentLoader:
  supported_players: list = dataclasses.field(default_factory=lambda: [0, 1])
  name: str = 'Random'

  def load(self, player, *args, **kwargs):
    assert player in self.supported_players
    return RandomAgent(player=player)