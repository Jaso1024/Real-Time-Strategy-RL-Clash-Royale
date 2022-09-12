from CRBot import CRBot

if __name__ == '__main__':
    bot = CRBot()

    episodes = 100
    load = True
    save = True
    bot.play(episodes, load, save)

