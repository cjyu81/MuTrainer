# -*- coding: utf-8 -*-
import asyncio
import time

from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from sweeper import SweeperPlayer
from max_damage_player import MaxDamagePlayer
from poke_env.environment.move_category import MoveCategory


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
