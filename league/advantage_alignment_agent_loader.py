import dataclasses
import pickle
from coin_train import GRUCoinAgent, CoinAgent


# it is a mere copy paste of the advantage alignment agent loader, as it uses the exact same architecture and checkpointing logic
def load_advantage_alignment_agent(path, player_id: int):
    with open(path, 'rb') as f:
        minimal_state = pickle.load(f)
    hp = minimal_state['hp']
    agent_params = minimal_state[f'agent0']['params']
    agent_module = GRUCoinAgent(hidden_size_actor=hp['actor']['hidden_size'],
                                hidden_size_qvalue=hp['qvalue']['hidden_size'],
                                layers_before_gru_actor=hp['actor']['layers_before_gru'],
                                layers_before_gru_qvalue=hp['qvalue']['layers_before_gru'], )
    # IMPORTANT: this is a hack to change player_id, but it works because the agent just sees the obsseq with GRU, and we just use inference, so it does not matter
    return CoinAgent(params=agent_params, model=agent_module, player=player_id)


@dataclasses.dataclass
class AdvantageAlignmentAgentLoader:
    path: str
    name: str
    supported_players: list = dataclasses.field(default_factory=lambda: [0, 1])

    def load(self, player, *args, **kwargs):
        assert player in self.supported_players
        return load_advantage_alignment_agent(self.path, player)


if __name__ == '__main__':
    advantage_alignment_agent = load_advantage_alignment_agent("/Users/miladaghajohari/PycharmProjects/loqa/league/checkpoints/advantage_alignment_seeds/minimal_state_25400_1", 0)
