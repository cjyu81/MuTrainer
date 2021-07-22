# -*- coding: utf-8 -*-
"""This module defines a player class exposing the Open AI Gym API.
"""
import asyncio
import numpy as np  # pyre-ignore
import time

from abc import ABC, abstractmethod, abstractproperty

from queue import Queue
from threading import Thread

from typing import Any, Callable, List, Optional, Tuple, Union

from poke_env.environment.battle import Battle
from poke_env.player.player import Player
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.utils import to_id_str
from self_play import MCTS, Node, GameHistory, MinMaxStats
from poke_env.pokeconfig import MuZeroConfig
from poke_env.data import POKEDEX  # if you are not playing in gen 8, you might want to do import GENX_POKEDEX instead
from poke_env.data import MOVES

import models
import torch

class MuPlayer(Player, ABC):  # pyre-ignore
    """Player with local gamehistory of own perspective.
    Chooses move using and MCTS search of local model."""
    #gottem
    MAX_BATTLE_SWITCH_RETRY = 10000
    PAUSE_BETWEEN_RETRIES = 0.001
    _ACTION_SPACE = list(range(4 * 4 + 6))

    SPECIES_TO_DEX_NUMBER = {}
    idx = 0
    for species, values in POKEDEX.items():
        SPECIES_TO_DEX_NUMBER[species] = idx
        idx += 1
        if 'otherFormes' in  values:
            for other_form in values['otherFormes']:
                SPECIES_TO_DEX_NUMBER[to_id_str(other_form)] = idx
                idx += 1
    MOVE_TO_NUM = {}
    a=1
    for moves, values in MOVES.items():
        MOVE_TO_NUM[moves] = a
        a+=1


    def __init__(
        self,
        player_configuration: Optional[PlayerConfiguration] = None,
        *,
        avatar: Optional[int] = None,
        battle_format: str = "gen8randombattle",
        log_level: Optional[int] = None,
        server_configuration: Optional[ServerConfiguration] = None,
        start_listening: bool = True,
        team: Optional[Union[str, Teambuilder]] = None,
        initcheckpoint = None,
        mu_config: MuZeroConfig,
    ):
        #summary deleted
        super(MuPlayer, self).__init__(
            player_configuration=player_configuration,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            max_concurrent_battles=1,
            server_configuration=server_configuration,
            start_listening=start_listening,
            team=team,
        )
        #self._actions = {}
        self._current_battle: Battle
        #self._observations = {}
        #self._reward_buffer = {}
        self._start_new_battle = False
        self.laction = 0
        self.gh = GameHistory()#this will be replaced by self_play's version
        self.config = mu_config
        #network initialization
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initcheckpoint["weights"])
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()
        self._ACTION_SPACE = list(range(4 * 4 + 6))
        self.temp = 0
        self.temp_thresh = 0
        self.lroot = 0


    def battle_once(self, opponent: Player):
        """
        Does not have team preview functionality for built team battles
        """
        #print("mu player 63 battle_once called",self.gh.observation_history)
        #challenge, accept by opponent, get battle, choose moves until end, game history
        self._start_new_battle = True
        async def launch_battles(player: MuPlayer, opponent: Player):
            battles_coroutine = asyncio.gather(
                player.send_challenges(
                    opponent=to_id_str(opponent.username),
                    n_challenges=1,
                    to_wait=opponent.logged_in,
                ),
                opponent.accept_challenges(
                    opponent=to_id_str(player.username), n_challenges=1
                ),
            )
            await battles_coroutine
            #self._current_battle = await battles_coroutine
        """
        def env_algorithm_wrapper(player, kwargs):
            #env_algorithm(player, **kwargs)#This should be a control function and not be necessary
            player._start_new_battle = False
            while True:
                try:
                    player.complete_current_battle()
                    player.reset()
                except OSError:
                    break
        loop = asyncio.get_event_loop()
        env_algorithm_kwargs=None#shouldnt be necessary so initialized as None
        if env_algorithm_kwargs is None:
            env_algorithm_kwargs = {}"""
        loop = asyncio.get_event_loop()#duplicated
        thread = Thread()
        thread.start()
        while self._start_new_battle:
            loop.run_until_complete(launch_battles(self, opponent))
        thread.join()
        """while True:#no idea what this does
            try:
                player.complete_current_battle()
                player.reset()
            except OSError:
                break
        ###
        loop = asyncio.get_event_loop()
        while self._start_new_battle:
            loop.run_until_complete(launch_battles(self, opponent))
        """

        #append game history stuff to local gh attribute
        #no return

    async def mu_message(self, player, message) -> None:
        await self.player_message(player, message)

    async def mu_room_message(self, messagein, battlein) -> None:
        await self._send_message(message = messagein, battle=battlein)

    def get_moves(self) -> Any:
        a = self.stripmoves(self._current_battle.avaliable_moves)
        #return a
        return [1,2,3,4]

    def battle_summary(self, perspective):
        return "I didn't finish this method b/c no purpose yet"

    def battle_state(self, battle: Battle):
        """
        Much of the data in a pokemon battle is not ordinal or easily represented
        This method will encode a variety of information naively using values from 0-1
        2D array with battle/field attributes in first row.
        Next 12 rows are player's and opponent's pokemon
        """
        battle = self._current_battle
        assert battle != None, "battle_state received None instead of Battle object"
        state = [[0]* 13]*13# 1+6+6 1 field, 6 mons, 6 opmons
        properties = 13#13 pokemon traits
        substate = [0]*properties #substate is one pokemon
        substate[0] = self.statetonum(battle.fields,12)#Set[Field]
        substate[1] = self.statetonum(battle.opponent_side_conditions,13)#Set(SideCondition)
        substate[2] = self.statetonum(battle.side_conditions,13)#Set(SideCondition)
        substate[3] = int(battle.maybe_trapped)#bool
        substate[4] = int(battle.trapped)#bool
        substate[5] = self.weathertonum(battle.weather)#Optional[Weather]
        substate[6] = int(battle.can_dynamax)
        substate[7] = 0
        substate[8] = 0
        substate[9] = 0
        substate[10] = 0
        substate[11] = 0
        substate[12] = 0
        state[0] = substate
        monindex = 1
        for mon in battle.team.values():
            substate[0] = 0#mon.species#
            substate[1] = mon.base_stats["spe"]/500
            substate[2] = self.typetonum(mon,18)#
            substate[3] = 0#mon.ability#####
            substate[4] = mon.level/100
            substate[5] = mon.current_hp_fraction
            substate[6] = self.statetonum(mon.effects,162)
            substate[7] = self.statustonum(mon.status,7)
            substate[8] = 0#mon.item#####
            substate[9], substate[10], substate[11], substate[12] = self.stripmoves(mon.moves)
            state[monindex] = substate
            monindex += 1
        opponent_prop = properties
        opss = [0] * opponent_prop
        for opmon in battle.opponent_team.values():
            opss[0] = 0#opmon.species#####
            opss[1] = opmon.base_stats["spe"]/500
            opss[2] = self.typetonum(battle.active_pokemon,18)
            opss[3] = 0#opmon.ability#####
            opss[4] = opmon.level/100
            opss[5] = opmon.current_hp_fraction
            opss[6] = self.statetonum(opmon.effects,162)
            opss[7] = self.statustonum(opmon.status,7)
            opss[8] = 0#opmon.item#####
            opss[9] = 0#will be it's known moves
            opss[10] = 0
            opss[11] = 0
            opss[12] = 0
            state[monindex] = substate
            monindex += 1
            #moves not implemented yet
        return [state]

    def empty_state(self):
        return [[[0]*13]*13]

    def stripmoves(self, moves):
        """
        moves parameter is array of 4 moves.
        Returns 4 integers representing the moves.
        These values are assigned with the constant MOVE_TO_NUM dictionary
        """
        intmoves = [0]*4
        counter = 0
        for move in moves:
            intmoves[counter]=self.MOVE_TO_NUM[move]
            counter+=1

        return intmoves[0], intmoves[1], intmoves[2], intmoves[3]

    def statetonum(self, states, statenum):
        #Generates a unique corresponding int ID for each field combination
        if not states or states is None:
            return 0
        a=0
        num = 0
        if(type(states) == 'Weather'):
            print("STATES IS ")
            print(states)
            raise ValueError("States is weather; please review errors")
        for state in states:
            num+=state.value*(statenum**a)
            a+=1
        return num

    def weathertonum(self, weather):
        if not weather or weather is None:
            return 0
        return weather.value

    def typetonum(self, pokemon, typenum):
        #same function as statetonum but functions for tuples instead of sets
        num=0
        num += pokemon._type_1.value
        if pokemon._type_2:
            num+= pokemon._type_2.value * typenum
        return num

    def statustonum(self, status, statusnum):
        if not status or status is None:
            return 0
        return status.value

    def mysit(self, instring):
        """
        My String To Int (mysit)
        """
        sum = 0
        for a in range(0, len(instring)):
            sum += ord(instring[a]) - 97
        return sum

    def myone(self,value,min,max):
        """
        My integer to 0-1 (myone)
        """
        return (value-min)/(max-min)

    def check_win(self, battle: Battle):
        if battle.won:
            return 1
        elif battle.lost:
            return -1
        return 0

    def printname(self):
        print("Mu Player's name is ", self._username)

    def _action_to_move(self, action: int, battle: Battle) -> str:
        """Abstract method converting elements of the action space to move orders.
        """

    def _battle_finished_callback(self, battle: Battle) -> None:
        print("battle has completed")
        #self._observations[battle].put(self.embed_battle(battle))

    def _init_battle(self, battle: Battle) -> None:
        self._current_battle = battle#added

    def choose_move(self, battle: Battle) -> str:
        #print("choose move method, obs history length below")
        #print(len(self.gh.observation_history), len(self.gh.observation_history[0]), len(self.gh.observation_history[0][0]),len(self.gh.observation_history[0][0]))
        #self.printname()
        self._init_battle(battle)
        temperature = self.temp
        temperature_threshold = self.temp_thresh
        print("choose move contents: ")
        print(self.config.stacked_observations)
        stacked_observations = self.gh.get_stacked_observations(
            -1,
            self.config.stacked_observations,
        )
        print(stacked_observations)
        root, mcts_info = MCTS(self.config).run(
            self.model,
            stacked_observations,
            self._ACTION_SPACE,
            1,
            True,
        )
        action = self.select_action(
            root,
            temperature
            if not temperature_threshold
            or len(gh.action_history) < temperature_threshold
            else 0,
        )
        self.laction = action
        self.lroot = root
        #step()
        #gamehistory appends are normally here
        return self._action_to_move(action, battle)

    #check move's results
    def check_move(self, battle: Battle, from_teampreview_request: bool = False, maybe_default_order=False):
        root = self.lroot
        self.gh.store_search_statistics(root, self.config.action_space)
        self.gh.action_history.append(self.laction)
        self.gh.observation_history.append(self.battle_state(battle))
        self.gh.reward_history.append(self.check_win(battle))
        self.gh.to_play_history.append(1)#technically should be to_play

    def choose_max_move(self, battle: Battle):
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

    def close(self) -> None:
        """Unimplemented. Has no effect."""

    def complete_current_battle(self) -> None:
        """Completes the current battle by performing random moves."""
        done = self._current_battle.finished
        while not done:
            _, _, done, _ = self.step(np.random.choice(self._ACTION_SPACE))

    def compute_reward(self, battle: Battle) -> float:
        """Returns a reward for the given battle.

        The default implementation corresponds to the default parameters of the
        reward_computing_helper method.

        :param battle: The battle for which to compute the reward.
        :type battle: Battle
        :return: The computed reward.
        :rtype: float
        """
        return self.reward_computing_helper(battle)

    #@abstractmethod
    def embed_battle(self, battle: Battle) -> Any:
        return self.battle_state(battle)
        """Abstract method for embedding battles.

        :param battle: The battle whose state is being embedded
        :type battle: Battle
        :return: The computed embedding
        :rtype: Any
        """

    def reset(self) -> Any:
        """Resets the internal environment state. The current battle will be set to an
        active unfinished battle.

        :return: The observation of the new current battle.
        :rtype: Any
        :raies: EnvironmentError
        """
        print("resetted 339 muplayer")
        #self._current_battle = battles[0]


    def render(self, mode="human") -> None:
        """A one line rendering of the current state of the battle.
        """
        print(
            "  Turn %4d. | [%s][%3d/%3dhp] %10.10s - %10.10s [%3d%%hp][%s]"
            % (
                self._current_battle.turn,
                "".join(
                    [
                        "⦻" if mon.fainted else "●"
                        for mon in self._current_battle.team.values()
                    ]
                ),
                self._current_battle.active_pokemon.current_hp or 0,
                self._current_battle.active_pokemon.max_hp or 0,
                self._current_battle.active_pokemon.species,
                self._current_battle.opponent_active_pokemon.species,  # pyre-ignore
                self._current_battle.opponent_active_pokemon.current_hp  # pyre-ignore
                or 0,
                "".join(
                    [
                        "⦻" if mon.fainted else "●"
                        for mon in self._current_battle.opponent_team.values()
                    ]
                ),
            ),
            end="\n" if self._current_battle.finished else "\r",
        )

    def seed(self, seed=None) -> None:
        """Sets the numpy seed."""
        np.random.seed(seed)

    def step(self, action: int) -> Tuple:
        """Performs action in the current battle.

        :param action: The action to perform.
        :type action: int
        :return: A tuple containing the next observation, the reward, a boolean
            indicating wheter the episode is finished, and additional information
        :rtype: tuple
        """
        self._actions[self._current_battle].put(action)
        observation = self._observations[self._current_battle].get()
        return (
            observation,
            self.compute_reward(self._current_battle),
            self._current_battle.finished,
            {},
        )

    def die():
        self.stop_listening()

    def action_space(self) -> List:
        """Returns the action space of the player. Must be implemented by subclasses."""
        pass

    def _action_to_move(self, action: int, battle: Battle) -> str:
        """Converts actions to move orders.

        The conversion is done as follows:

        0 <= action < 4:
            The actionth available move in battle.available_moves is executed.
        4 <= action < 8:
            The action - 4th available move in battle.available_moves is executed, with
            z-move.
        8 <= action < 12:
            The action - 8th available move in battle.available_moves is executed, with
            mega-evolution.
        8 <= action < 12:
            The action - 8th available move in battle.available_moves is executed, with
            mega-evolution.
        12 <= action < 16:
            The action - 12th available move in battle.available_moves is executed,
            while dynamaxing.
        16 <= action < 22
            The action - 16th available switch in battle.available_switches is executed.

        If the proposed action is illegal, a random legal move is performed.

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        if (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif (
            not battle.force_switch
            and battle.can_z_move
            and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)
        ):
            return self.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action - 8], mega=True)
        elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action - 12], dynamax=True)
        elif 0 <= action - 16 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 16])
        else:
            return self.choose_random_move(battle)

    @property
    def action_space(self) -> List:
        """The action space for gen 7 single battles.

        The conversion to moves is done as follows:

            0 <= action < 4:
                The actionth available move in battle.available_moves is executed.
            4 <= action < 8:
                The action - 4th available move in battle.available_moves is executed,
                with z-move.
            8 <= action < 12:
                The action - 8th available move in battle.available_moves is executed,
                with mega-evolution.
            12 <= action < 16:
                The action - 12th available move in battle.available_moves is executed,
                while dynamaxing.
            16 <= action < 22
                The action - 16th available switch in battle.available_switches is
                executed.
        """
        return self._ACTION_SPACE

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = np.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

def main():
    start = time.time()

    # We create two players.
    max_damage_player = MaxDamagePlayer(battle_format="gen8randombattle")
    mu_player = MuPlayer(battle_format="gen8randombattle")


    # Now, let's evaluate our player
    mu_player.battle_against(max_damage_player, n_battles=5)

    print(
        "Max damage player won %d / 100 battles [this took %f seconds]"
        % (max_damage_player.n_won_battles, time.time() - start)
    )


if __name__ == "__main__":
    main()
    #asyncio.get_event_loop().run_until_complete(main())
