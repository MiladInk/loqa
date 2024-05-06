import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import jax.numpy as jp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCriticMFOS(nn.Module):
    def __init__(self, input_shape, action_dim, n_out_channels, batch_size):
        super(ActorCriticMFOS, self).__init__()
        self.batch_size = batch_size
        self.n_out_channels = n_out_channels
        self.space = n_out_channels

        self.conv_a_0 = nn.Conv2d(input_shape[0], n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.conv_a_1 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.linear_a_0 = nn.Linear(n_out_channels * input_shape[1] * input_shape[2], self.space)

        self.GRU_a = nn.GRU(input_size=self.space, hidden_size=self.space)
        self.linear_a = nn.Linear(self.space, action_dim)

        self.conv_v_0 = nn.Conv2d(input_shape[0], n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.conv_v_1 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.linear_v_0 = nn.Linear(n_out_channels * input_shape[1] * input_shape[2], self.space)

        self.GRU_v = nn.GRU(input_size=self.space, hidden_size=self.space)
        self.linear_v = nn.Linear(self.space, 1)

        self.conv_t_0 = nn.Conv2d(input_shape[0], n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.conv_t_1 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, stride=1, padding="same", padding_mode="circular")
        self.linear_t_0 = nn.Linear(n_out_channels * input_shape[1] * input_shape[2], self.space)
        self.GRU_t = nn.GRU(input_size=self.space, hidden_size=self.space)
        self.linear_t = nn.Linear(self.space, self.space)

        self.reset(None, outer=True)

    def reset(self, memory, outer=False):
        if not outer:
            state_tbs = torch.stack(self.state_traj_bs, dim=0)
            memory.states_traj.append(state_tbs)
            memory.actions_traj.append(torch.stack(self.action_traj_b, dim=0))
            memory.logprobs_traj.append(torch.stack(self.logprob_traj_b, dim=0))
            state_Bs = state_tbs.flatten(end_dim=1)
            x = self.conv_t_0(state_Bs)
            x = F.relu(x)
            x = self.conv_t_1(x)
            x = F.relu(x)
            x = torch.flatten(x, start_dim=1)
            x = self.linear_t_0(x)
            x = F.relu(x)
            x = x.view(len(self.state_traj_bs), self.batch_size, self.space)
            x, _ = self.GRU_t(x)
            x = x[-1]
            x = x.mean(0, keepdim=True).repeat(self.batch_size, 1)
            # ACTIVATION HERE?
            x = self.linear_t(x)
            x = torch.sigmoid(x)
            self.th_bh = x
        else:
            self.th_bh = torch.ones(self.batch_size, self.space).to(device)

        self.ah_obh = torch.zeros(1, self.batch_size, self.space).to(device)
        self.vh_obh = torch.zeros(1, self.batch_size, self.space).to(device)

        self.state_traj_bs = []
        self.action_traj_b = []
        self.logprob_traj_b = []

    def forward(self):
        raise NotImplementedError

    def forward_a(self, state_bs):
        x = self.conv_a_0(state_bs)
        x = F.relu(x)
        x = self.conv_a_1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_a_0(x)
        x = F.relu(x)
        x, self.ah_obh = self.GRU_a(x.unsqueeze(0), self.ah_obh)
        x = x.squeeze(0)
        x = F.relu(x)
        x = self.th_bh * x

        return F.softmax(self.linear_a(x), dim=-1)

    def act(self, state_bs):
        action_probs_ba = self.forward_a(state_bs)
        dist = Categorical(action_probs_ba)
        action_b = dist.sample()

        self.state_traj_bs.append(state_bs)
        self.action_traj_b.append(action_b)
        self.logprob_traj_b.append(dist.log_prob(action_b))
        return action_b

    def evaluate(self, state_Ttbs, action_Ttb):

        state_Bs = state_Ttbs.transpose(0, 1).flatten(end_dim=2)

        x = self.conv_t_0(state_Bs)
        x = F.relu(x)
        x = self.conv_t_1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_t_0(x)
        x = F.relu(x)
        x = x.view(state_Ttbs.size(1), state_Ttbs.size(0) * self.batch_size, self.space)
        x, _ = self.GRU_t(x)  # tBh
        x = x[-1]  # Bh
        x = x.view(state_Ttbs.size(0), self.batch_size, self.space)  # Tbh
        x = x.mean(1, keepdim=True).repeat(1, self.batch_size, 1)  # Tbh
        # ACTIVATION HERE?
        x = self.linear_t(x)  # Tbh
        x = torch.sigmoid(x)
        th_Tbh = torch.cat((torch.ones(1, self.batch_size, self.space).to(device), x[:-1]), dim=0)

        x = self.conv_a_0(state_Bs)
        x = F.relu(x)
        x = self.conv_a_1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_a_0(x)
        x = F.relu(x)
        x = x.view(state_Ttbs.size(1), state_Ttbs.size(0) * self.batch_size, self.space)  # tBh
        x, _ = self.GRU_a(x)  # tBh
        x = x.view(state_Ttbs.size(1), state_Ttbs.size(0), self.batch_size, self.space)  # tTbh
        x = x.transpose(0, 1)  # Ttbh
        x = F.relu(x)
        x = th_Tbh.unsqueeze(1) * x
        action_probs_Ttba = F.softmax(self.linear_a(x), dim=-1)
        dist = Categorical(action_probs_Ttba)
        action_logprobs = dist.log_prob(action_Ttb)
        dist_entropy = dist.entropy()

        x = self.conv_v_0(state_Bs)
        x = F.relu(x)
        x = self.conv_v_1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.linear_v_0(x)
        x = F.relu(x)
        x = x.view(state_Ttbs.size(1), state_Ttbs.size(0) * self.batch_size, self.space)  # tBh
        x, _ = self.GRU_v(x)
        x = x.view(state_Ttbs.size(1), state_Ttbs.size(0), self.batch_size, self.space)  # tTbh
        x = x.transpose(0, 1)  # Ttbh
        x = F.relu(x)
        x = th_Tbh.unsqueeze(1).detach() * x
        state_value = self.linear_v(x).squeeze(-1)

        return action_logprobs, state_value, dist_entropy

class BoundMFOSAgent():
    def __init__(self, path, player, grid_size):
        self.player = player
        self.grid_size = grid_size
        batch_size = 512
        self.batch_size = batch_size

        with open(path, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))

        state_dim = [7, grid_size, grid_size]
        action_dim = 4
        n_latent_var = 16


        self.mfos = ActorCriticMFOS(state_dim, action_dim, n_latent_var, batch_size)
        self.mfos.load_state_dict(state_dict['actor_critic'])

    def _prep_state(self, env_state, rewards_inner, rewards_outer, dones_inner):
        rewards_inner_tiled = torch.tile(rewards_inner[None, None].T, [1, self.grid_size, self.grid_size])[:, None]
        rewards_outer_tiled = torch.tile(rewards_outer[None, None].T, [1, self.grid_size, self.grid_size])[:, None]
        dones_inner_tiled = torch.tile(dones_inner[None, None].T, [1, self.grid_size, self.grid_size])[:, None]

        return torch.cat([env_state, rewards_inner_tiled, rewards_outer_tiled, dones_inner_tiled], axis=1)

    def get_action(self, **kwargs):
        episode = kwargs.get('episode')
        rewards = episode['rew']
        player_rewards = rewards[:, self.player]
        opponent_rewards = rewards[:, 1 - self.player]
        dones = torch.zeros(1)
        t = kwargs.get('t')
        obs = episode['obs'][t, self.player]
        #obs = obs[(1, 0, 3, 2), ...] # swapping observation as m-fos codebase uses different ordering
        obs = obs[(0, 1, 2, 3), ...] # swapping observation as m-fos codebase uses different ordering
        import numpy as np
        def torchify(x):
            return torch.from_numpy(np.array(x))
        def jaxify(x):
            return jp.array(np.array(x))
        obs = torchify(obs).unsqueeze(0)
        player_reward_t = torchify(player_rewards[t-1]) if t > 0 else torch.zeros(1)
        opponent_reward_t = torchify(opponent_rewards[t-1]) if t > 0 else torch.zeros(1)
        state = self._prep_state(obs, player_reward_t, opponent_reward_t, dones)
        state = state.repeat(self.batch_size, 1, 1, 1)
        actions = self.mfos.act(state)
        chosen_action = int(actions[0])
        map_from_mfos_to_loqa = {0: 1, 1: 0, 2: 3, 3: 2}
        action = map_from_mfos_to_loqa[chosen_action]
        return action

@dataclasses.dataclass
class MFOSAgentLoader:
  path: str
  name: str
  supported_players: list = dataclasses.field(default_factory=lambda: [0, 1])

  def load(self, player, **kwargs):
    assert player in self.supported_players
    grid_size = kwargs.get('grid_size')
    return BoundMFOSAgent(path=self.path, player=player, grid_size=grid_size)


if __name__ == '__main__':
    mfos_agent = BoundMFOSAgent(path="/Users/miladaghajohari/PycharmProjects/loqa/league/checkpoints/mfos/1000.pth", player=0, grid_size=7)

