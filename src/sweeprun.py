# -*- coding: utf-8 -*-
import asyncio
import time
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from sweeper import SweeperPlayer
from max_damage_player import MaxDamagePlayer
from poke_env.environment.move_category import MoveCategory

class sweepclassrun:
    async def sweeperrun(self):
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
async def main():
    sw = sweepclassrun()
    await(sw.sweeperrun())


if __name__ == "__main__":
        asyncio.get_event_loop().run_until_complete(main())
