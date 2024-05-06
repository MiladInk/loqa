import jax
# add to sys.path
# print working directory
import os
print('working directory', os.getcwd())
import sys
sys.path.append('.') # for the league import to work
from league.run_league import get_all_agent_loaders, get_hp, evaluate_these_agent_combinations
import matplotlib.animation as animation
from IPython.display import HTML
import jax
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import io
import fire


hp = get_hp(debug_mode=False, batch_size=1, trace_length=50)
_, named_agent_loaders = get_all_agent_loaders(hp)
print('keys', named_agent_loaders.keys())
clean_agent_name_dict = {
    'loqa_2i4vsulp': 'LOQA',
    'Always Defect': 'AD',
    'Always Cooperate': 'AC',
}

def buffer_plot_and_get(fig):
    t = time.time()
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', dpi=80)
    plt.close()
    buf.seek(0)
    image = Image.open(buf)
    print(f'opening took {time.time() - t}')
    return image

def arrow_end_coords(start_x, start_y, action, arrow_length):
    end_x, end_y = start_x, start_y

    if action == 0:  # LEFT
        end_x -= arrow_length
    elif action == 1:  # RIGHT
        end_x += arrow_length
    elif action == 2:  # UP
        end_y -= arrow_length
    elif action == 3:  # DOWN
        end_y += arrow_length

    return end_x, end_y

def plot_game(game, ax,  frame: int, prev_ret_1, prev_ret_2, info=None):
    t = time.time()
    width = game['games'].WIDTH
    height = game['games'].HEIGHT
    coin_position = game['coin_pos'][0], game['coin_pos'][1]
    coin_owner = game['coin_owner']
    player1_pos = game['player1_pos'][0], game['player1_pos'][1]
    player2_pos = game['player2_pos'][0], game['player2_pos'][1]
    # set the position of the coin
    coin_x = coin_position[1]
    coin_y = coin_position[0]
    player1_y = player1_pos[0]
    player1_x = player1_pos[1]
    player2_y = player2_pos[0]
    player2_x = player2_pos[1]
    player1_action = game['act'][0]
    player2_action = game['act'][1]

    # create a figure and axis object
    ax.clear()

    # loop over the board and add the squares to the plot
    for i in range(height):
        for j in range(width):
            ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, color='black', linewidth=3.))

    # add the coin to the plot
    if coin_owner == 0:
        ax.add_patch(plt.Circle((coin_x + 0.8, coin_y + 0.8), 0.2, fill=True, color='red'))
    elif coin_owner == 1:
        ax.add_patch(plt.Circle((coin_x + 0.8, coin_y + 0.8), 0.2, fill=True, color='blue'))

    # load the player images
    player1_img = mpimg.imread('assets/player_1.png') # run this script from the root project directory for this to work
    player2_img = mpimg.imread('assets/player_2.png')

    # red player
    ax.imshow(player1_img, extent=[player1_x, player1_x + 0.4, player1_y, player1_y + 0.4], alpha=1.0)
    # blue player
    ax.imshow(player2_img, extent=[player2_x + 0.6, player2_x + 1.0, player2_y, player2_y + 0.4], alpha=1.0)
    if frame % 2 == 1:
        # red player reward
        ax.text(player1_x + 0.2, player1_y + 0.6, int(game['rew'][0]), color="black", ha="center", va="center", fontsize=30)
        # blue player reward
        ax.text(player2_x + 0.8, player2_y + 0.6, int(game['rew'][1]), color="black", ha="center", va="center", fontsize=30)
        # Draw arrow for player 1's action
        arrow_length = 0.2
        arrow_width = 0.05
        arrow_start_x = player1_x + 0.2
        arrow_start_y = player1_y + 0.3
        arrow_end_x, arrow_end_y = arrow_end_coords(arrow_start_x, arrow_start_y, player1_action, arrow_length)

        ax.arrow(arrow_start_x, arrow_start_y, arrow_end_x - arrow_start_x, arrow_end_y - arrow_start_y,
                 head_width=arrow_width, head_length=arrow_width, fc='red', ec='purple', linewidth=3.)

        # Draw arrow for player 2's action
        arrow_start_x = player2_x + 0.8
        arrow_start_y = player2_y + 0.3
        arrow_end_x, arrow_end_y = arrow_end_coords(arrow_start_x, arrow_start_y, player2_action, arrow_length)

        ax.arrow(arrow_start_x, arrow_start_y, arrow_end_x - arrow_start_x, arrow_end_y - arrow_start_y,
                 head_width=arrow_width, head_length=arrow_width, fc='blue', ec='purple', linewidth=3.)

    # set the limits of the plot
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # hide the ticks and axis labels
    ax.set_xticks([])
    ax.set_yticks([])

    if info:
        time_step = info['time_step']
        agent1_name = info['agent1']
        agent2_name = info['agent2']
        # ax.set_title(f'ًًRed:{agent1_name} vs. Blue:{agent2_name} | t: {time_step}',
        #      fontsize=24, color='#34495e', fontweight='bold', fontname='Arial',
        #      backgroundcolor='white', pad=10)
        x_center = 0.5
        y_position = 1.05
        ret_1 = int(game['ret'][0] if frame % 2 == 1 else prev_ret_1)
        ret_2 = int(game['ret'][1] if frame % 2 == 1 else prev_ret_2)
        ax.text(x_center - 0.2, y_position, f'{agent1_name}({ret_1})', color='red', ha='right', va='bottom', transform=ax.transAxes, size=20)
        ax.text(x_center + 0.0, y_position, 'vs. ', color='black', ha='center', va='bottom', transform=ax.transAxes, size=20)
        ax.text(x_center + 0.2, y_position, f'{agent2_name}({ret_2})', color='blue', ha='left', va='bottom', transform=ax.transAxes, size=20)
        #ax.text(x_center + 0.2, y_position, f'| t : {time_step}', color='black', ha='left', va='bottom', transform=ax.transAxes, size=20)

    print(f'took {time.time() - t}.')
    return ax

# Set up the figure
def main(agent1_name, agent2_name): # for example agent1_name='loqa_2i4vsulp', agent2_name='Always Defect'
    agent1_loader = named_agent_loaders[agent1_name]
    agent2_loader = named_agent_loaders[agent2_name]
    results, episode_logs = evaluate_these_agent_combinations(combinations=[(agent1_loader, agent2_loader)],
                                                              batch_size=1,
                                                              rng=jax.random.PRNGKey(42),
                                                              hp=hp,
                                                              for_instead_of_vmap=False, )
    episodes = episode_logs[(agent1_name, agent2_name)][0]

    episode = jax.tree_map(lambda x: x[0], episodes)
    episode['ret'] = jax.numpy.cumsum(episode['rew'], axis=0)
    fig, ax = plt.subplots(figsize=(4, 4))
    def update(frame):
        t = frame // 2
        game = jax.tree_map(lambda x: x[t], episode)
        ax.clear()
        plot_game(game, ax=ax, info={'time_step': t,
                                     'agent1': clean_agent_name_dict[agent1_name],
                                     'agent2': clean_agent_name_dict[agent2_name]},
                  frame=frame,
                  prev_ret_1=episode['ret'][t - 1][0] if t > 0 else 0,
                  prev_ret_2=episode['ret'][t - 1][1] if t > 0 else 0,
                  )
        return ax

    ani = animation.FuncAnimation(fig, update, frames=range(102), blit=False)

    # To display the animation in a Jupyter notebook
    HTML(ani.to_jshtml())

    # Or to save the animation as a video file
    ani.save(f'./animations/{agent1_name}vs{agent2_name}_animation.mp4', writer='ffmpeg', fps=1)


if __name__ == '__main__':
    fire.Fire()

