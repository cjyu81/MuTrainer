class SnivyGame():
    """
    Was gonna use as portal between pokemongame.py and mu_player.py but instead
    added all the methods to pokemongame.py as helper methods
    """
    #if move fails, re-call decision making process
    #get account battle's legal actions
    #get account objects and terminate current battles?
    #issue new battle challenge
    #render account's battles
    def __init__(self, seed=None):
        self.currentgame = 0
        self.player1 = 1
        self.player2 = 2


    def get_observation(self,player):
        return [
            [self.truefield*1]*1
        ]

    def legal_actions(self, playernum):
        """
        Return legal actions of player. No faint switches or zero PP moves.

        Returns:
            An array of integers, subset of the action space.
        """
        if(playernum == 1):
            return [0,1,2,3,4]
        else:
            return [0,1,2,3,4]

    def have_winner(self):
        """
        Check battle conditions. Return appropriate value
        """
        return -1,1
        return 1,-1
        return 0, 0

    async def reset(self):
        #get account objects and terminate current battles?
        #issue new battle challenge
        print("Truefield vision")
        print(self.truefield)
        self.truefield = numpy.zeros((5), dtype="int32")
        self.player = 1
        return self.get_observation(1), self.get_observation(2)

    def close(self):
        #no idea how to use this or reset
        pass

    def render(self):
        """Render game through terminal with string data"""
        print("This is supposed to be printing battle string data")
