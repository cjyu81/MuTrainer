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


class MuPlayer(Player, ABC):  # pyre-ignore
    """Player with local gamehistory of own perspective.
    Chooses move using and MCTS search of local model."""
    #gottem
    MAX_BATTLE_SWITCH_RETRY = 10000
    PAUSE_BETWEEN_RETRIES = 0.001
    _ACTION_SPACE = list(range(4 * 4 + 6))

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
        self.gh = GameHistory()

    async def battle_once(self, opponent: Player):
        print("mu plaer 63 battle_once called")
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

        def env_algorithm_wrapper(player, kwargs):
            #env_algorithm(player, **kwargs)

            player._start_new_battle = False
            while True:
                try:
                    player.complete_current_battle()
                    player.reset()
                except OSError:
                    break

        loop = asyncio.get_event_loop()
        env_algorithm_kwargs=None
        if env_algorithm_kwargs is None:
            env_algorithm_kwargs = {}

        thread = Thread(
            target=lambda: env_algorithm_wrapper(self, env_algorithm_kwargs)
        )
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
        #battle initialzation observations
        self.gh.action_history.append(0)
        self.gh.observation_history.append(self.battle_state(self._current_battle))
        self.gh.reward_history.append(0)
        self.gh.to_play_history.append(1)
        #append game history stuff to local gh attribute
        #no return

    async def mu_message(self, player, message) -> None:
        await self.player_message(player, message)

    async def mu_room_message(self, messagein, battlein) -> None:
        await self._send_message(message = messagein, battle=battlein)

    def _action_to_move(self, action: int, battle: Battle) -> str:
        """Abstract method converting elements of the action space to move orders.
        """

    def get_moves(self) -> Any:
        a = self.stripmoves(self._current_battle.avaliable_moves)
        #return a
        return [1,2,3,4]

    def battle_summary(self, perspective):
        return "I didn't finish this method b/c no purpose yet"

    def battle_state(self, battle: Battle):
        battle = self._current_battle
        assert battle != None, "battle_state received None instead of Battle object"
        state = [0]* 8
        state0 = [0]*6
        state0[0] = battle.fields#Set[Field]
        state0[1] = battle.opponent_side_conditions#Set(SideCondition)
        state0[2] = battle.side_conditions#Set(SideCondition)
        state0[3] = battle.maybe_trapped#bool
        state0[4] = battle.trapped#bool
        state0[5] = battle.weather#Optional[Weather]
        state[0] = state0

        properties = 13
        substate = [None]*properties #substate is one pokemon
        mon1 = battle.active_pokemon
        substate[0] = mon1.species
        substate[1] = mon1.base_stats
        substate[2] = mon1.types
        substate[3] = mon1.ability
        substate[4] = mon1.can_dynamax#level
        substate[5] = mon1.current_hp_fraction
        substate[6] = mon1.effects
        substate[7] = mon1.status
        substate[8] = mon1.item
        intmovefour = self.stripmoves(mon1.moves)
        substate[9] = intmovefour[0]
        substate[10] = intmovefour[1]
        substate[11] = intmovefour[2]
        substate[12] = intmovefour[3]
        state[1] =substate
        monindex = 2
        for mon in battle.team.values():
            substate[0] = mon.species
            substate[1] = mon.base_stats
            substate[2] = mon.types
            substate[3] = mon.ability
            substate[4] = mon.level
            substate[5] = mon.current_hp_fraction
            substate[6] = mon.effects
            substate[7] = mon.status
            substate[8] = mon.item
            intmovefour = self.stripmoves(mon1.moves)
            substate[9] = intmovefour[0]
            substate[10] = intmovefour[1]
            substate[11] = intmovefour[2]
            substate[12] = intmovefour[3]
            state[monindex] = substate
            monindex += 1
        opponent_prop = 2


        return state
        """
        field weather
        terrain
        state
        hazards
        dynamax available

        species
        base stats
        types
        ability
        level
        hp fraction
        effects
        status
        moves and PP
        item
        dynamaxed(only for active)
        boosts(only for active)
        end
        """

    def stripmoves(self, moves):
        """
        moves paramter is array of 4 moves.
        Returns 4 integers representing the moves.
        These values are abritrarily determined by move id.
        """
        intmoves[4] = [0]*4
        counter = 0
        for moof in moves:
            intmoves[counter]=self.mysit(moof.id)
            counter+=1
        return intmoves

    def mysit(self, instring):
        """
        My String To Int (mysit)
        """
        sum = 0
        for a in range(0, len(instring)):
            sum += ord(instring[a]) - 97
        return sum

    def check_win(self, battle: Battle):
        if battle.won:
            return 1
        elif battle.lost:
            return -1
        return 0

    def _battle_finished_callback(self, battle: Battle) -> None:
        print("battle has completed")
        #self._observations[battle].put(self.embed_battle(battle))

    def _init_battle(self, battle: Battle) -> None:
        self._observations[battle] = Queue()
        self._actions[battle] = Queue()
        self._current_battle = battle#added

    def choose_move(self, action, battle: Battle) -> str:
        if battle not in self._observations or battle not in self._actions:
            self._init_battle(battle)
        stacked_observations = game_history.get_stacked_observations(
            -1,
            self.config.stacked_observations,
        )
        root, mcts_info = MCTS(self.config).run(
            self.model,
            stacked_observations,
            self.game.legal_actions(1),
            self.game.to_play(),#shouldnt exist
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
        #step()
        #gamehistory appends are normally here
        return self._action_to_move(action, battle)

    #check move's results
    def check_move(self, battle: Battle, from_teampreview_request: bool = False, maybe_default_order=False):
        gh.store_search_statistics(root, self.config.action_space)
        gh.action_history.append(self.laction)
        gh.observation_history.append(self.battle_state(battle))
        gh.reward_history.append(self.check_win(battle))
        gh.to_play_history.append(self.to_play())#supposed to be to_play but to_play is not in synchrnous

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
        for _ in range(self.MAX_BATTLE_SWITCH_RETRY):
            battles = dict(self._actions.items())
            battles = [b for b in battles if not b.finished]
            if battles:
                self._current_battle = battles[0]
                observation = self._observations[self._current_battle].get()
                return observation
            time.sleep(self.PAUSE_BETWEEN_RETRIES)
        else:
            raise EnvironmentError("User %s has no active battle." % self.username)

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
    """
    def play_against(
        self, env_algorithm: Callable, opponent: Player, env_algorithm_kwargs=None
    ):
        Executes a function controlling the player while facing opponent.

        The env_algorithm function is executed with the player environment as first
        argument. It exposes the open ai gym API.

        Additional arguments can be passed to the env_algorithm function with
        env_algorithm_kwargs.

        Battles against opponent will be launched as long as env_algorithm is running.
        When env_algorithm returns, the current active battle will be finished randomly
        if it is not already.

        :param env_algorithm: A function that controls the player. It must accept the
            player as first argument. Additional arguments can be passed with the
            env_algorithm_kwargs argument.
        :type env_algorithm: callable
        :param opponent: A player against with the env player will player.
        :type opponent: Player
        :param env_algorithm_kwargs: Optional arguments to pass to the env_algorithm.
            Defaults to None.

        self._start_new_battle = True

        async def launch_battles(player: EnvPlayer, opponent: Player):
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

        def env_algorithm_wrapper(player, kwargs):
            env_algorithm(player, **kwargs)

            player._start_new_battle = False
            while True:
                try:
                    player.complete_current_battle()
                    player.reset()
                except OSError:
                    break

        loop = asyncio.get_event_loop()

        if env_algorithm_kwargs is None:
            env_algorithm_kwargs = {}

        thread = Thread(
            target=lambda: env_algorithm_wrapper(self, env_algorithm_kwargs)
        )
        thread.start()

        while self._start_new_battle:
            loop.run_until_complete(launch_battles(self, opponent))
        thread.join()
    """
    async def die():
        await self.stop_listening()

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


async def main():
    start = time.time()

    # We create two players.
    max_damage_player = MaxDamagePlayer(battle_format="gen8randombattle")
    mu_player = MuPlayer(battle_format="gen8randombattle")


    # Now, let's evaluate our player
    await mu_player.battle_against(max_damage_player, n_battles=5)

    print(
        "Max damage player won %d / 100 battles [this took %f seconds]"
        % (max_damage_player.n_won_battles, time.time() - start)
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
