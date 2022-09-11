from CRBot import CRBot

if __name__ == '__main__':
    bot = CRBot()

    episodes = 10
    load = False
    save = False
    bot.play(episodes, load, save)

    