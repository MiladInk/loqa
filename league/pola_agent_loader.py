import dataclasses
import pickle

import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jp


class POLAAgent(nn.Module):
  player: int
  hidden_size: int = 64

  @nn.compact
  def __call__(self, obsseq):
    assert obsseq.shape[1:] == (4,3,3)
    T,C,H,W = obsseq.shape
    xs = obsseq.reshape([T,-1])
    hs = nn.relu(nn.Dense(features=self.hidden_size, name="linear1")(xs))
    scancell = nn.scan(nn.GRUCell, in_axes=0, out_axes=0,
                       variable_broadcast="params",
                       split_rngs=dict(params=False))(name="GRUCell")
    carry = nn.GRUCell.initialize_carry(jax.random.PRNGKey(1), (), self.hidden_size)
    carry, hs = scancell(carry, hs)
    zs = nn.Dense(features=4, name="linear_end")(hs)  # 4 actions
    zs = zs[..., (1, 0, 3, 2)]  # permute actions
    return zs


@struct.dataclass
class BoundPOLAAgent:
  params: "Any"
  module: "POLAAgent" = struct.field(pytree_node=False)

  @property
  def player(self):
    return self.module.player

  def __call__(self, obsseq):
    return self.module.apply(dict(params=self.params), obsseq)

  def get_action(self, **kwargs):
    rng = kwargs.get('rng')
    episode = kwargs.get('episode')
    t = kwargs.get('t')

    logits = self(episode['obs'][:, self.module.player])
    return jax.random.categorical(rng, logits[t])

def load_pola_agents(path, player):
  with open(path, "rb") as file:
    paramss = pickle.load(file)
  paramss = jax.tree_util.tree_map(jp.array, paramss)  # TODO does this put them on device?
  # NOTE as of august 2023 we are using the code in the official POLA repo, which does not
  # flip observations for the second player. to avoid mistakes I remove the second agent here.
  paramss = paramss[:1]
  return [BoundPOLAAgent(params["params"], POLAAgent(player=player)) for params in paramss]

def load_pola_agent(path, player: int):
  return load_pola_agents(path, player=player)[0]

@dataclasses.dataclass
class POLAAgentLoader:
  path: str
  name: str
  supported_players: list = dataclasses.field(default_factory=lambda: [0, 1])

  def load(self, player, *args, **kwargs):
    assert player in self.supported_players
    return load_pola_agent(self.path, player)


if __name__ == '__main__':
  pola_agent = load_pola_agent("checkpoints/pola_20230805/agents_t20230805-0823_seed2_update200.pkl", 0)
  print(pola_agent(jp.ones([7,4,3,3])))

