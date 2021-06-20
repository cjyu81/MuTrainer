# -*- coding: utf-8 -*-
import asyncio
import time

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from max_damage_player import MaxDamagePlayer
from poke_env.environment.move_category import MoveCategory
from poke_env.player.env_player import Gen8EnvSinglePlayer


class SnivyPlayer(Gen8EnvSinglePlayer):
    def choose_move(self, battle):
        #If at full health, set up
        # If the player can attack, it will
        if battle.available_moves:
            # Picks first status move to spam
            if battle.active_pokemon.current_hp_fraction == 1:
                for moof in battle.available_moves:
                    if(moof.category == MoveCategory.STATUS):
                        return self.create_order(moof)
                #No status moves to choose from
                return self.choose_random_move(battle)
            else:
                best_move = max(battle.available_moves, key=lambda move: move.base_power)
                return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

    def attempt_move(self, battle):
        #If at full health, set up
        # If the player can attack, it will
        if battle.available_moves:
            # Picks first status move to spam
            if battle.active_pokemon.current_hp_fraction == 1:
                for moof in battle.available_moves:
                    if(moof.category == MoveCategory.STATUS):
                        return self.create_order(moof)
                #No status moves to choose from
                return self.choose_random_move(battle)
            else:
                best_move = max(battle.available_moves, key=lambda move: move.base_power)
                return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

    async def snivy_message(
        self, player, message
    ) -> None:
        await self.player_message(player, message)
        #await self._challenge("Archlei", "gen8randombattle")
        #USE /pm COMMAND TO MESSAGE ON SHOWDOWN

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100 # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    #battle.opponent_active_pokemon.moves,#type_1,
                    #battle.opponent_active_pokemon.item,#type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [moves_base_power, moves_dmg_multiplier, [remaining_mon_team, remaining_mon_opponent]]
        )
    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )

async def main():
    start = time.time()

    # We create two players.
    max_damage_player = MaxDamagePlayer()
    #mdp2 = MaxDamagePlayer()
    #await mdp2.battle_against(max_damage_player, n_battles=5)
    snivy_player = SnivyPlayer()

    # Now, let's evaluate our player
    await snivy_player.battle_against(max_damage_player, n_battles=5)
    #await snivy_player.snivy_message("Archlei","Hello world. I am Snivy")
    #for a in range(10):
    #    snivy_player = SnivyPlayer(battle_format="gen8randombattle")
    #    await snivy_player.snivy_message("Archlei","Hello world. I am Snivy")

    #await snivy_player.player_message(f"/challenge {name}, {format_}", "lobby")
    """print(
        "Snivy player won %d / 100 battles [this took %f seconds]"
        % (snivy_player.n_won_battles, time.time() - start)
    )"""


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
