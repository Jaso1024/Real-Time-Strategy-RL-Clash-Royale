import os
import time

import cv2
import keyboard as kb
import numpy as np
import pandas as pd
import pyautogui as ui
import win32con
import win32com.client
import win32gui
from PIL import Image

from Resources.Models.ElixirModel import ElixirModel
from collections import OrderedDict


class Handler:
    """A class used to interact with a bluestacks instance of Clash Royale"""
    def __init__(self, spells=False):
        def create_elixir_model():
            self.elixir_model = ElixirModel()
            self.elixir_model.load_model()

        create_elixir_model()

        self.top_right = (self.get_window_dimensions()[0], self.get_window_dimensions()[1])
        self.scalars = self.get_window_scalars()
        self.spells = spells

    def get_state(self):
        """
        Returns the current state of the Clash Royale arena.

        :return: state (dict): A dictionary containing data about the field, elixir-count, card, choices
        relevant to the current state of the game
        """
        frame = self.get_frame()

        field_data = self.gen_field_data(frame)

        elixir_data = self.predict_elixir(frame)

        choice_data, card_data = self.gen_choice_data(frame)

        state = {
            "field_data": field_data,
            "elixir_data": elixir_data,
            "choice_data": choice_data,
            "card_data": card_data
        }

        return state

    # Essential
    def get_frame(self):
        """
        Returns a rescaled image (244x419) of game.

        :return: window_image_rescaled (np.array): A numpy array of an image of the Clash Royale window
        """
        window_dimensions = self.get_window_dimensions()
        screenshot = cv2.cvtColor(np.asarray(ui.screenshot()), cv2.COLOR_BGR2RGB)
        screenshot = Image.fromarray(np.asarray(screenshot))
        window_image = screenshot.crop((window_dimensions))
        window_image = np.asarray(window_image)
        window_image_rescaled = cv2.resize(window_image, (244, 419), interpolation=cv2.INTER_AREA)
        window_image_rescaled = np.array(cv2.cvtColor(window_image_rescaled, cv2.COLOR_BGR2RGB))
        return window_image_rescaled

    def get_window_dimensions(self):
        """
        Returns the dimensions of the Clash Royale window (not including the title bar)
        in the format - (left, top, right, bottom).

        :return: window_dimensions (tuple): A tuple containing the dimensions of the Clash Royale window
        """
        window = win32gui.FindWindow(None, "BlueStacks App Player")
        win32gui.ShowWindow(window, win32con.SW_RESTORE)
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys("%")
        win32gui.SetForegroundWindow(window)

        time.sleep(0.05)
        window_dimensions = win32gui.GetWindowRect(window)
        window_dimensions = (window_dimensions[0], window_dimensions[1] + 45, *window_dimensions[2:])
        return window_dimensions

    def get_window_scalars(self):
        """
        Returns the length and width scalars of the current window size given the standard window size (244,419).

        :return: scalars (tuple): A tuple containing 2 float values
        """
        window_dimensions = self.get_window_dimensions()
        current_dimensions = (window_dimensions[3] - window_dimensions[1], window_dimensions[2] - window_dimensions[0])
        scalars = (current_dimensions[1] / 244, current_dimensions[0] / 419)
        return scalars

    # Visualization
    def save_current_frame(self):
        """
        Saves the current frame of the Clash Royale window to the "Visualizations" folder.

        :return: None
        """
        current_frame = cv2.cvtColor(np.asarray(self.get_frame()), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"Visualizations/Image.png", current_frame)

    # Interaction
    def battle(self):
        """
        Clicks the "Battle" button in the Clash Royale window.

        :return: None
        """
        self.scalars = self.get_window_scalars()
        window_dimensions = self.get_window_dimensions()
        battle_button_location = (78 * self.scalars[0] + window_dimensions[0],
                                  278 * self.scalars[1] + window_dimensions[1]
                                  )
        ui.click(battle_button_location)
        time.sleep(0.5)

    def start_training_game(self):
        """
        Starts a training game in the Clash Royale window.

        :return: None
        """
        self.scalars = self.get_window_scalars()
        window_dimensions = self.get_window_dimensions()
        options_button_location = (225 * self.scalars[0] + window_dimensions[0],
                                   39 * self.scalars[1] + window_dimensions[1]
                                   )
        ui.click(options_button_location)
        time.sleep(0.1)

        start_game_button_location = (152 * self.scalars[0] + window_dimensions[0],
                                      146 * self.scalars[1] + window_dimensions[1]
                                      )
        ui.click(start_game_button_location)
        time.sleep(0.1)

        ok_button_location = (167 * self.scalars[0] + window_dimensions[0],
                              249 * self.scalars[1] + window_dimensions[1]
                              )
        ui.click(ok_button_location)
        time.sleep(4.5)

    def act(self, choice):
        """
        Executes the action correlating to the choice given.

        :param choice: A dictionary containing data about an executable action
        :return: None
        """
        self.scalars = self.get_window_scalars()
        if choice is None:
           return

        key_mappings = {1: "a", 2: "s", 3: "d", 4: "f"}
        card_num = choice['card_number']
        if card_num is None:
            return None
        card_key = key_mappings[card_num+1]
        location = choice["tile_screen_location"]

        kb.press(card_key)
        time.sleep(0.01)
        kb.release(card_key)
        try:
            ui.click(location)
        except Exception as e:
            print("Please stop moving the mouse", e)
        ui.click(self.top_right[0] + 5, self.top_right[1] + 5)

    def leave_game(self):
        """
        Leaves a finished game in the Clash Royale window.

        :return: None
        """
        self.scalars = self.get_window_scalars()
        window_dimensions = self.get_window_dimensions()
        ok_button_location = (
            121 * self.scalars[0] + window_dimensions[0],
            358 * self.scalars[1] + window_dimensions[1]
        )
        ui.click(ok_button_location)

    def acknowledge_reward_limit_reached(self):
        """
        Clicks the "Reward limit reached" button if it pops up, clicks a space and has no effect otherwise.

        :return: None
        """
        self.scalars = self.get_window_scalars()
        window_dimensions = self.get_window_dimensions()
        ok_button_location = (
            123 * self.scalars[0] + window_dimensions[0],
            260 * self.scalars[1] + window_dimensions[1]
        )
        ui.click(ok_button_location)
        time.sleep(0.5)

    def ignore_new_reward(self):
        """
        Ignores new reward if available, otherwise does nothing.

        :return: None
        """
        if self.check_for_new_reward():
            self.scalars = self.get_window_scalars()
            window_dimensions = self.get_window_dimensions()
            ok_button_location = (
                120 * self.scalars[0] + window_dimensions[0],
                390 * self.scalars[1] + window_dimensions[1]
            )
            ui.click(ok_button_location)
            time.sleep(0.5)

    # Verification
    def match_to_template(self, image, template, threshold):
        """
        Standard template matching.

        :param image: A 2D iterable (must be smaller than template)
        :param template: A 2D iterable (must be bigger than image)
        :param threshold: A float (should be 0.0<=x<=1.0) that determines how well the image must match the template
        :return: A boolean representing whether the image matches the template at the given threshold
        """
        matches = cv2.matchTemplate(np.asarray(image), template, cv2.TM_CCOEFF_NORMED)
        _, m, _, _ = cv2.minMaxLoc(matches)
        return m >= threshold

    def training_game_over(self):
        """
        Checks if there is a training game ongoing in the Clash Royale window.

        :return: A boolean representing whether a training match is ongoing or not
        """

        frame = Image.fromarray(np.array(self.get_frame()))
        rectangle = np.asarray(frame.crop((205, 0, 242, 7)))
        rectangle = cv2.cvtColor(np.asarray(rectangle), cv2.COLOR_BGR2RGB)

        matches = []
        for size in ("XS", "S", "M", "L", "XL"):
            template = np.asarray(Image.open(f"Resources/Templates/OngoingGame{size}.png"))
            matches.append(self.match_to_template(rectangle, template, 0.30))

        return not any(matches)

    def game_is_over(self):
        """
        Checks if a competitive game is ongoing in the Clash Royale window.

        :return: A boolean representing whether a competitive game is ongoing or not
        """
        frame = Image.fromarray(np.array(self.get_frame()))
        rectangle = np.asarray(frame.crop((10, 25, 13, 28)))
        rectangle = cv2.cvtColor(np.asarray(rectangle), cv2.COLOR_BGR2RGB)

        cv2.imwrite("Resources/Data/EpisodialImageData/game_is_over.png", rectangle)
        image = np.asarray(Image.open("Resources/Data/EpisodialImageData/game_is_over.png"))

        matches = []
        for size in ("XS", "S", "M", "L", "XL", "O"):
            template = np.asarray(Image.open(f"Resources/Templates/OngoingBattle{size}.png"))
            matches.append(self.match_to_template(image, template, 0.70))

        return not any(matches)

    def at_home_screen(self):
        """
        Checks that the current state of the Clash Royale window is the home screen.

        :return: A boolean representing whether the game window is currently at the home screen
        """
        frame = Image.fromarray(np.array(self.get_frame()))
        rectangle = np.asarray(frame.crop((108, 390, 136, 399)))

        cv2.imwrite("Resources/Data/EpisodialImageData/HomeScreen.png", rectangle)
        image = np.asarray(Image.open("Resources/Data/EpisodialImageData/HomeScreen.png"))

        template = np.asarray(Image.open("Resources/Templates/HomeScreen.png"))

        return self.match_to_template(image, template, 0.7)

    def check_reward_limit_reached(self):
        """
        Checks if the reward limit is reached.

        :return: A boolean representing whether the reward limit has been reached
        """
        frame = Image.fromarray(np.array(self.get_frame()))
        rectangle = np.asarray(frame.crop((85, 155, 153, 214)))

        template = np.asarray(Image.open("Resources/Templates/RewardLimitReached.png"))

        return self.match_to_template(rectangle, template, 0.6)

    def check_for_new_reward(self):
        """
        Checks if there is a new reward.

        :return: A boolean representing whether there is a new reward or not
        """
        frame = Image.fromarray(np.array(self.get_frame()))
        rectangle = np.asarray(frame.crop((85, 387, 163, 410)))
        cv2.imwrite("Resources/Data/EpisodialImageData/NewReward.png", rectangle)

        image = np.asarray(Image.open("Resources/Data/EpisodialImageData/NewReward.png"))
        template = np.asarray(Image.open("Resources/Templates/NewReward.png"))

        return self.match_to_template(image, template, 0.6)

    # Observations
    def get_player_crowns(self):
        """
        Returns the amount of crowns the player won.

        :return: An integer corresponding to the amount of crowns the player has won
        """
        frame = Image.fromarray(np.array(self.get_frame()))
        crown_1 = cv2.cvtColor(np.asarray(frame.crop((60, 200, 72, 210))), cv2.COLOR_BGR2RGB)
        crown_2 = cv2.cvtColor(np.asarray(frame.crop((116, 193, 128, 207))), cv2.COLOR_BGR2RGB)
        crown_3 = cv2.cvtColor(np.asarray(frame.crop((170, 199, 182, 210))), cv2.COLOR_BGR2RGB)

        crowns = [crown_1, crown_2, crown_3]
        templates = []
        battle_templates = []

        for num in range(1, 4):
            cv2.imwrite(f"Resources/Data/EpisodialImageData/Player_{num}_crown.png", crowns[num - 1])
            templates.append(np.asarray(Image.open(f"Resources/Templates/Player_{num}_crown.png")))
            battle_templates.append(np.asarray(Image.open(f"Resources/Templates/Player_{num}_battle_crown.png")))

        crowns = []
        for num in range(1, 4):
            crowns.append(np.asarray(Image.open(f"Resources/Data/EpisodialImageData/Player_{num}_crown.png")))

        for num, (crown, template, battle_template) in enumerate(zip(crowns, templates, battle_templates)):
            if self.match_to_template(crown, template, 0.6) or self.match_to_template(crown, battle_template, 0.6):
                continue
            else:
                return num

        return 3

    def get_enemy_crowns(self):
        """
        Returns the amount of crowns the enemy won.

        :return: An integer corresponding to the amount of crowns the enemy has won
        """
        frame = Image.fromarray(np.array(self.get_frame()))
        crown_1 = cv2.cvtColor(np.asarray(frame.crop((59, 86, 69, 93))), cv2.COLOR_BGR2RGB)
        crown_2 = cv2.cvtColor(np.asarray(frame.crop((114, 79, 124, 88))), cv2.COLOR_BGR2RGB)
        crown_3 = cv2.cvtColor(np.asarray(frame.crop((172, 85, 181, 93))), cv2.COLOR_BGR2RGB)

        crowns = [crown_1, crown_2, crown_3]
        templates = []

        for num in range(1, 4):
            cv2.imwrite(f"Resources/Data/EpisodialImageData/Enemy_{num}_crown.png", crowns[num - 1])
            templates.append(np.asarray(Image.open(f"Resources/Templates/Enemy_{num}_crown.png")))

        crowns = []
        for num in range(1, 4):
            crowns.append(np.asarray(Image.open(f"Resources/Data/EpisodialImageData/Enemy_{num}_crown.png")))

        for num, (crown, template) in enumerate(zip(crowns, templates)):
            if self.match_to_template(crown, template, 0.6):
                continue
            else:
                return num
        return 3

    # Elixir
    def get_elixir_image(self, frame):
        """
        Returns an image of the elixir bar.

        :param frame: A 2D iterable image of the current frame
        :return: elixir_image (np.array): A numpy array of the elixir bar image
        """
        frame = Image.fromarray(frame)
        elixir_image = frame.crop((66, 408, 233, 409))  # 167x1
        elixir_image = np.array(elixir_image)
        return elixir_image

    def fit_model_elixir(self, elixir_image, elixir_count):
        """
        Trains elixir model.

        :param elixir_image: A 2D iterable representing an image of the elixir bar
        :param elixir_count: An integer of representing the approximate amount of elixir displayed in the image
        :return: None
        """
        self.elixir_model.fit(elixir_image, elixir_count)

    def predict_elixir(self, frame):
        """
        Returns a prediction of the amount of the elixir currently available.

        :param frame: A 2D iterable representing an image of the Clash Royale window
        :return: prediction (float): The prediction of the elixir count
        """
        elixir_image = self.get_elixir_image(frame)
        prediction = self.elixir_model.predict(elixir_image)[0][0]

        return prediction

    def elixir_model_mass_training(self):
        """
        Trains the elixir model (requires user interaction)
        :return: None
        """
        images = []
        while True:
            x = input("take photo?:")
            if x == 'no':
                break
            images.append(self.get_elixir_image(self.get_frame()))

        for image in images:
            cv2.imshow('fff', cv2.resize(np.array(image), (800, 200)))
            cv2.waitKey(0)
            x = int(input('enter elixir:'))
            self.elixir_add_to_training(image, x)

    def elixir_add_to_training(self, image, elixir):
        """
        Adds sample to elixir model training data.

        :param image: A 2D iterable representing an image of the elixir bar
        :param elixir: An integer or float representing the approximate amount of elixir available
        :return: None
        """
        training_data = pd.read_pickle("Resources/Models/TrainingData/ElixirData.pkl")
        new_data = pd.DataFrame({"Elixir": [elixir], "image": [np.asarray(image)]})
        training_data = pd.concat([training_data, new_data], sort=False)
        training_data.to_pickle("Resources/Models/TrainingData/ElixirData.pkl")

    # Field
    def gen_field_data(self, frame):
        """
        Returns data representing the current state of the field.

        :param frame: A 2D iterable representing an image of the Clash Royale window
        :return: field_data (dict): containing data about the field of the battle
        """
        full_image = np.asarray(cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2GRAY))
        field_dimensions = OrderedDict({
            "enemy_side_dimensions": (20, 36, 226, 164),  # 206x128: 26,368px
            "left_bridge_dimensions": (50, 162, 68, 186),  # 18x24: 432px
            "right_bridge_dimensions": (175, 162, 193, 186),  # 18x24: 432px
            "player_side_dimensions": (20, 182, 226, 314),  # 206x132: 26,368px
        })

        field_data = {}
        field_shapes = [(206,128), (18,24), (18,24), (206,132)]
        for (place, dimensions), shape in zip(field_dimensions.items(), field_shapes):
            image = np.asarray(Image.fromarray(full_image).crop(dimensions))
            image = np.asarray(cv2.resize(image, (int(shape[0]/4), int(shape[1]/4))))
            field_data[place] = image

        return field_data

    # Choices
    def gen_choice_data(self, frame):
        """
        Returns a dictionary of the action choices currently available.

        :param frame: A 2D iterable representing an image of the Clash Royale window
        :param spells: A boolean indicating if there are spells in the deck being used
        :return: A tuple (list[dict,dict,...], list[np.array,np.array]) representing the action choices and cards - respectively - available
        """
        window_dimensions = self.get_window_dimensions()

        window_bottom_left = (window_dimensions[0], window_dimensions[3])
        bottom_left_tile_location = (
            window_bottom_left[0] + (24 * self.scalars[0]),
            window_bottom_left[1] - (111 * self.scalars[1])
        )

        # 1tile - 24x20px
        screen_tile_size = (11 * self.scalars[0], 8 * self.scalars[1])
        choice_data = []
        card_choices = self.get_cards(frame)
        for x in range(0, 18):
            for y in range(0, 14):
                tile_coordinates = np.array([x, y])
                tile_screen_location = ((bottom_left_tile_location[0] + (x * screen_tile_size[0])),
                                            (bottom_left_tile_location[1] - (y * screen_tile_size[1])))
                    
                choice_data.append({"tile_coordinates": tile_coordinates,
                                    "tile_screen_location": tile_screen_location,
                                    })

        if self.spells:
            extra_choice_data = self.gen_north_of_bridge_choice_data(card_choices)
            choice_data.extend(extra_choice_data)
        return choice_data, card_choices

    def gen_north_of_bridge_choice_data(self, card_choices):
        """
        Returns a dictionary of the action choices currently available from north of the bridge.

        :param card_choices: A list of numpy arrays representing the available card options
        :return: Choice_data (list): A list of dictionaries containing data about the action choices available north of the bridge
        """
        window_dimensions = self.get_window_dimensions()
        window_bottom_left = (window_dimensions[0], window_dimensions[3])
        screen_tile_size = (11 * self.scalars[0], 8 * self.scalars[1])

        choice_data = []

        tile_to_gen_from = (
            window_bottom_left[0] + (25 * self.scalars[0]),
            window_bottom_left[1] - (258 * self.scalars[1])
        )

        for x in range(18):
            for y in range(14):
                for choice in card_choices:
                    tile_coordinates = (x, y)
                    tile_screen_location = ((tile_to_gen_from[0] + (x * screen_tile_size[0])),
                                            (tile_to_gen_from[1] - (y * screen_tile_size[1])))
                    if choice[1][0] != 1:
                        choice_data.append({"tile_coordinates": tile_coordinates,
                                            "tile_screen_location": tile_screen_location,
                                            "card_number": choice[0],
                                            "card": choice[1]
                                            })
                    else:
                        choice_data.append(None)

        return choice_data

    def get_cards(self, frame):
        """
        Returns a list of the available cards.

        :param frame: A 2D iterable representing an image of the Clash Royale window
        :return: playable_cards (list): A list of numpy arrays representing the cards currently available
        """
        full_image = Image.fromarray(frame)

        # all card images are 35x43
        card1 = full_image.crop((58, 350, 93, 393))
        card2 = full_image.crop((105, 350, 140, 393))
        card3 = full_image.crop((150, 350, 185, 393))
        card4 = full_image.crop((195, 350, 230, 393))

        cards = (card1, card2, card3, card4)
        card_images = [
            cv2.cvtColor(np.array(Image.open(f"Resources/Cards/{card}"), dtype=np.float32), cv2.COLOR_BGR2GRAY) for card
            in os.listdir("Resources/Cards")]

        playable_cards = []
        valids = []
        identity = np.identity(9)
        for num in range(1, 5):
            card = cards[num - 1]
            card = np.array(card, dtype=np.float32)
            card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
            added = False
            for c_num in range(len(card_images)):
                cc = card_images[c_num]
                matches = cv2.matchTemplate(card[10:30,10:40] / 255, cc / 255, cv2.TM_CCOEFF_NORMED)
                _, m, _, _ = cv2.minMaxLoc(matches)
                if m > 0.7:
                    playable_cards.append([num, identity[c_num + 1]])
                    valids.append(1)
                    added = True
            if not added:
                playable_cards.append([num, identity[0]])
        
        return playable_cards



