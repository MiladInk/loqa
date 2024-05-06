import dataclasses

import flax.struct as struct

from league.agent import get_defect_action
from league._utils import rscope


@struct.dataclass
class AlwaysDefectAgent:
  player: int = struct.field(pytree_node=False)
  hp: dict = struct.field(pytree_node=False)

  def get_action(self, **kwargs):
    rng = kwargs.get('rng')
    episode = kwargs.get('episode')
    t = kwargs.get('t')

    return get_defect_action(episode=episode, t=t, hp=self.hp, rng=rscope(rng, 'cooperative_action_rng'), agent_player=1-self.player, other_player=self.player)


@dataclasses.dataclass
class AlwaysDefectAgentLoader:
  hp: dict
  supported_players: list = dataclasses.field(default_factory=lambda: [0, 1])
  name: str = 'Always Defect'

  def load(self, player, *args, **kwargs):
    assert player in self.supported_players
    return AlwaysDefectAgent(player=player, hp=self.hp)



