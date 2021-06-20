# -*- coding: utf-8 -*-
import numpy as np
import asyncio
import time

from poke_env.player.mu_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class TepigPlayer(Gen8EnvSinglePlayer):
    def choose_move(self, battle):
        #Access as much of the opponent's team's information as possible to test accessbility
        #_opponent_team
        #opponent_active_pokemon
        #opponent_team
        #_teampreview_opponent_team
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            #await self.mu_room_message("message", "Lobby")
            return self.create_order(best_move)
        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )

class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

async def main():
    start = time.time()

    # We create two players.
    random_player = RandomPlayer(battle_format="gen8randombattle")
    tepig_player = TepigPlayer(battle_format="gen8randombattle")

    # Now, let's evaluate our player
    await tepig_player.battle_against(random_player, n_battles=3)

    print(
        "Tepig damage player won %d / 3 battles [this took %f seconds]"
        % (tepig_player.n_won_battles, time.time() - start)
    )

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
