from abc import ABC, abstractmethod

import datetime
import os

import numpy
import torch

import asyncio
import time

from .abstract_game import AbstractGame

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from snivyplayer import SnivyPlayer
from poke_env.environment.move_category import MoveCategory
from poke_env.player.mu_player import MuPlayer
from poke_env.player_configuration import _create_player_configuration_from_player
from poke_env.player_configuration import PlayerConfiguration
from poke_env.player_configuration import _create_muplayer_configuration

class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (1, 1, 5)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(5))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 42  # Maximum number of moves if game is not finished before
        self.num_simulations = 200  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 42  # Number of game moves to keep for every batch element
        self.td_steps = 42  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None,playeronenum=0,playertwonum=0):
        self.env = PokemonGame(seed, playeronenum,playertwonum)

    async def step(self, action1, action2):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        dummyhold = await asyncio.gather(self.env.step(action1,action2))
        observation1, reward1, done1, observation2, reward2, done2 = dummyhold[0]
        return observation1, reward1 * 10, done1, observation2, reward2 * 10, done2

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()


    def legal_actions(self, playernum):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions(playernum)


    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the column to play for the player {self.to_play()}: ")
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter another column : ")
        return int(choice)

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return 1#self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return f"Play column {action_number + 1}"

class PokemonGame:
    """
    First version of pokemon game for muzero
    Spaghetti code
    Spaghetti files
    Looks like it was made by someone who hasn't even graduated high school
    """
    #acc1 and acc2 are mu_player
    #def __init__(self, seed=None, acc1=None, acc2=None):
    def __init__(self, seed=None,playeronenum=0,playertwonum=0):
        #print("creation of pokemon game class. one: ", playeronenum, ". two: ",playertwonum)
        self.player1 = MuPlayer(player_configuration= _create_muplayer_configuration(playeronenum))
        self.player2 = MuPlayer(player_configuration= _create_muplayer_configuration(playertwonum))
        #for a in range(10):
        #    snivy_player = SnivyPlayer(battle_format="gen8randombattle")
        self.game1 = None
        self.game2 = None
        self.create_game()


    async def step(self, action1, action2):
        """
        Blindly attempts to step for both players
        """
        print("pokemongame step")
        self.attempt_move(self.player1,move,self.game1)
        self.attempt_move(self.player2,move,self.game2)

        done = self.have_winner() or len(self.legal_actions()) == 0

        rewardone, rewardtwo = self.have_winner()
        return self.get_observation(1), rewardone, done, self.get_observation(2), rewardtwo, done

    def get_observation(self, player):
        if player == 1:
            obsdata = self.player1.battle_state(self.game1)
        elif player ==2:
            obsdata = self.player2.battle_state(self.game2)
        else:
            raise NotImplementedError("Only players 1 & 2 have observations")
        return [
            [onefield*1]*1
        ]

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return 1
        raise NotImplementedError("Both players must move each turn")
        return self.player

    def legal_actions(self, playernum):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        if playernum == 1:
            return self.player1.get_moves()
        elif playernum == 2:
            return self.player2.get_moves()
        else:
            return "legal actions got player num not 1 or 2"

    def have_winner(self):
        if self.game1.won():
            if self.game1._won_by(self.player1.username):
                return 1, -1
            else: #player 2 must have won
                return -1, 1
        else:
            return 0, 0

    async def reset(self):
        await self.player1.mu_message("Archlei", "I am player 1. I am reseting battle")
        await self.player2.mu_message("Archlei", "I am player 2. Same with player 1")
        #self.player1.complete_current_battle()
        #self.player2.complete_current_battle()
        self.create_game()
        return self.get_observation(1), self.get_observation(2)

    async def close(self):
        await self.player1.die()
        await self.player2.die()

    def render(self):
        #render account's games
        print(self.player1.battle_state(self.game1),self.player2.battle_state(self.game2))

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the action to play for the player {self.to_play()}: ")
        while int(choice) not in self.legal_actions():
            choice = input("Ilegal action. Enter another action : ")
        return int(choice)

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        raise NotImplementedError

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return str(action_number)

    async def helpermethodsbelow(self):
        pass

    async def create_game(self):# Error is here. The battle against does not return the battle object. The battle object is needed
        await self.player1.battle_against(self.player2, n_battles=1)#doesn't return anything
        self.game1 = player1._current_battle
        self.game2 = player2._current_battle
        print("This is gameobject1: ", self.game1)

    async def attempt_move(self, player, action):
        """
        Attempts moves. If a side's move fails, choose max damage move, then random
        """
        if player == 1:
            self.player1.choose_move(action, self.game1)
            #if move fails, max move automatically picks max then random
            if False:
                self.player1.choose_max_move(self.game1)
        elif player == 2:
            self.player2.choose_move(action, self.game2)
            if False:
                self.player2.choose_max_move(self.game2)
