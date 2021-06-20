# -*- coding: utf-8 -*-
import asyncio
import time

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from max_damage_player import MaxDamagePlayer
from poke_env.environment.move_category import MoveCategory


class SweeperPlayer(Player):
    def choose_move11(self, battle):
        pass
        #translator input
        #translator output

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


async def main():
    start = time.time()

    # We create two players.
    max_damage_player = MaxDamagePlayer(battle_format="gen8randombattle")
    sweeper_player = SweeperPlayer(battle_format="gen8randombattle")

    # Now, let's evaluate our player
    await sweeper_player.battle_against(max_damage_player, n_battles=100)

    print(
        "Sweeper player won %d / 100 battles [this took %f seconds]"
        % (sweeper_player.n_won_battles, time.time() - start)
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
