import unittest
from ClashRoyaleHandler import ClashRoyaleHandler
from BattleAgent import BattleAgent
from BattleModel import BattleModel
from ClashRoyalebot import ClashRoyaleBot

class Tester(unittest.TestCase):

    def setUp(self) -> None:
        self.env = ClashRoyaleHandler()
        self.agent = BattleAgent(load=False)

    def testAgentTraining(self):
        for num in range(10):
            state = self.env.get_state()
            state_ = self.env.get_state()
            self.agent.experience(state, 3, state_, 1, False)
        self.agent.remember(5)
    
    def testBot(self):
        pass
    
unittest.main()