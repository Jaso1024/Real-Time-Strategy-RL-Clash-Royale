from collections import OrderedDict

import win32gui
import numpy as np
import cv2
from PIL import Image
import time
import pandas as pd
import pyautogui as ui
import keyboard as kb
import os
import absl

from Resources.Models.ElixirModel import ElixirModel
from Resources.Models.EnemyTowerModel import EnemyTowerHealthModel





absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ClashRoyaleHandler:
    def __init__(self):
        def create_elixir_model():
            self.elixir_model = ElixirModel()
            self.elixir_model.load_model()

        def create_enemy_towers_model():
            self.enemy_tower_model = EnemyTowerHealthModel()
            self.enemy_tower_model.load_model()

        create_elixir_model()

        self.top_right = (self.get_window_dimensions()[0], self.get_window_dimensions()[1])
        self.scalars = self.get_window_scalars()

    def get_state(self):
        frame = self.get_frame()

        field_data = self.gen_field_data(frame)

        elixir_data = self.predict_elixir(frame)

        choice_data = self.gen_choice_data(frame)

        state_data = OrderedDict({"field_data": field_data,
                                  "elixir_data": elixir_data,
                                  "choice_data": choice_data,
                                  })

        return state_data

    # Essential
    def get_frame(self):
        window_dimensions = self.get_window_dimensions()

        screenshot = cv2.cvtColor(np.asarray(ui.screenshot()), cv2.COLOR_BGR2RGB)
        screenshot = Image.fromarray(np.asarray(screenshot))
        window_image = screenshot.crop((window_dimensions))
        window_image = np.asarray(window_image)

        window_image_rescaled = cv2.resize(window_image, (244, 419), interpolation= cv2.INTER_AREA)
        window_image_rescaled = cv2.cvtColor(window_image_rescaled, cv2.COLOR_BGR2RGB)
        return window_image_rescaled

    def get_window_dimensions(self):
        window = win32gui.FindWindow(None, "BlueStacks App Player")
        # win32gui.SetForegroundWindow(window)
        time.sleep(0.05)
        window_dimensions = win32gui.GetWindowRect(window)
        window_dimensions = (window_dimensions[0], window_dimensions[1] + 45, *window_dimensions[2:])
        return window_dimensions

    def get_window_scalars(self):
        window_dimensions = self.get_window_dimensions()

        current_dimensions = (window_dimensions[3] - window_dimensions[1], window_dimensions[2] - window_dimensions[0])
        return current_dimensions[1] / 244, current_dimensions[0] / 419

    # Visualization
    def save_current_frame(self):
        current_frame = cv2.cvtColor(np.asarray(self.get_frame()), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"Visualizations/Image.png", current_frame)

    # Interaction
    def battle(self):
        self.scalars = self.get_window_scalars()
        window_dimensions = self.get_window_dimensions()
        battle_button_location = (78 * self.scalars[0] + window_dimensions[0],
                                  278 * self.scalars[1] + window_dimensions[1]
                                  )
        ui.click(battle_button_location)
        time.sleep(0.5)

    def start_training_game(self):
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
        self.scalars = self.get_window_scalars()
        if choice is None:
            return

        key_mappings = {1: "a", 2: "s", 3: "d", 4: "f"}
        card_key = key_mappings[choice['card_number']]
        location = choice["tile_screen_location"]

        kb.press(card_key)
        time.sleep(0.01)
        kb.release(card_key)
        ui.click(location)
        ui.click(self.top_right[0] + 5, self.top_right[1] + 5)

    def leave_game(self):
        self.scalars = self.get_window_scalars()
        window_dimensions = self.get_window_dimensions()
        ok_button_location = (
            121 * self.scalars[0] + window_dimensions[0],
            358 * self.scalars[1] + window_dimensions[1]
        )
        ui.click(ok_button_location)

    def acknowledge_reward_limit_reached(self):
        self.scalars = self.get_window_scalars()
        window_dimensions = self.get_window_dimensions()
        ok_button_location = (
            123 * self.scalars[0] + window_dimensions[0],
            260 * self.scalars[1] + window_dimensions[1]
        )
        ui.click(ok_button_location)
        time.sleep(0.5)
        return True

    def ignore_new_reward(self):
        if self.check_for_new_reward():
            self.scalars = self.get_window_scalars()
            window_dimensions = self.get_window_dimensions()
            ok_button_location = (
                120 * self.scalars[0] + window_dimensions[0],
                390 * self.scalars[1] + window_dimensions[1]
            )
            ui.click(ok_button_location)
            time.sleep(0.5)
            return True
        else:
            return False



    # Verification
    def match_to_template(self, image, template, threshold):
        res = cv2.matchTemplate(np.asarray(image), template, cv2.TM_CCOEFF_NORMED)
        min, max, thing1, thing2 = cv2.minMaxLoc(res)
        return max >= threshold

    def training_game_over(self):
        frame = Image.fromarray(np.array(self.get_frame()))
        rectangle = np.asarray(frame.crop((205, 0, 242, 7)))
        rectangle = cv2.cvtColor(np.asarray(rectangle), cv2.COLOR_BGR2RGB)

        matches = []
        for size in ("XS", "S", "M", "L", "XL"):
            template = np.asarray(Image.open(f"Resources/Templates/OngoingGame{size}.png"))
            matches.append(self.match_to_template(rectangle, template, 0.30))

        return not any(matches)

    def game_is_over(self):
        frame = Image.fromarray(np.array(self.get_frame()))
        rectangle = np.asarray(frame.crop((50, 400, 60, 415)))
        rectangle = cv2.cvtColor(np.asarray(rectangle), cv2.COLOR_BGR2RGB)

        cv2.imwrite("Resources/Data/EpisodialImageData/game_is_over.png", rectangle)
        image = np.asarray(Image.open("Resources/Data/EpisodialImageData/game_is_over.png"))

        matches = []
        for size in ("XS", "S", "M", "L", "XL"):
            template = np.asarray(Image.open(f"Resources/Templates/OngoingBattle{size}.png"))
            matches.append(self.match_to_template(image, template, 0.60))

        return not any(matches)

    def at_home_screen(self):
        frame = Image.fromarray(np.array(self.get_frame()))
        rectangle = np.asarray(frame.crop((108, 390, 136, 399)))

        cv2.imwrite("Resources/Data/EpisodialImageData/HomeScreen.png", rectangle)
        image = np.asarray(Image.open("Resources/Data/EpisodialImageData/HomeScreen.png"))

        template = np.asarray(Image.open("Resources/Templates/HomeScreen.png"))

        return self.match_to_template(image, template, 0.7)

    def check_reward_limit_reached(self):
        frame = Image.fromarray(np.array(self.get_frame()))
        rectangle = np.asarray(frame.crop((85, 155, 153, 214)))

        cv2.imwrite("Resources/Templates/RewardLimitReached.png", rectangle)
        template = np.asarray(Image.open("Resources/Templates/RewardLimitReached.png"))

        return self.match_to_template(rectangle, template, 0.6)

    def check_for_new_reward(self):
        frame = Image.fromarray(np.array(self.get_frame()))
        rectangle = np.asarray(frame.crop((85, 387, 163, 410)))
        cv2.imwrite("Resources/Data/EpisodialImageData/NewReward.png", rectangle)

        image = np.asarray(Image.open("Resources/Data/EpisodialImageData/NewReward.png"))
        template = np.asarray(Image.open("Resources/Templates/NewReward.png"))

        return self.match_to_template(image, template, 0.6)
    # Observations
    def determine_winner(self):
        player_crowns = self.get_player_crowns()
        enemy_crowns = self.get_enemy_crowns()
        return player_crowns > enemy_crowns, enemy_crowns > player_crowns

    def get_player_crowns(self):
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
        frame = Image.fromarray(frame)
        elixir_image = frame.crop((66, 408, 233, 409))  # 167x1
        elixir_image = np.array(elixir_image)
        return elixir_image

    def fit_model_elixir(self, elixir_image, elixir_count):
        self.elixir_model.fit(elixir_image, elixir_count)

    def predict_elixir(self, frame):
        elixir_image = self.get_elixir_image(frame)
        prediction = self.elixir_model.predict(elixir_image)[0][0]

        return prediction

    def elixir_model_mass_fitting(self):
        images = []
        while True:
            x = input("take photo?:")
            if x == 'no':
                break
            images.append(env.get_elixir_image(env.get_frame()))

        for image in images:
            cv2.imshow('fff', cv2.resize(np.array(image), (800, 200)))
            cv2.waitKey(0)
            x = int(input('enter elixir:'))
            self.elixir_add_to_training(image, x)

    def elixir_add_to_training(self, image, elixir):
        training_data = pd.read_pickle("Resources/Models/TrainingData/ElixirData.pkl")
        new_data = pd.DataFrame({"Elixir": [elixir], "image": [np.asarray(image)]})
        training_data = pd.concat([training_data, new_data], sort=False)
        training_data.to_pickle("Resources/Models/TrainingData/ElixirData.pkl")

    # Field
    def gen_field_data(self, frame):
        full_image = np.asarray(cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2GRAY))
        field_dimensions = OrderedDict({
            "enemy_side_dimensions": (20, 36, 226, 164),  # 206x128: 26,368px
            "left_bridge_dimensions": (50, 162, 68, 186),  # 18x24: 432px
            "right_bridge_dimensions": (175, 162, 193, 186),  # 18x24: 432px
            "player_side_dimensions": (20, 182, 226, 314),  # 206x132: 26,368px
        })

        field_data = OrderedDict()
        for place, dimensions in field_dimensions.items():
            image = np.asarray(Image.fromarray(full_image).crop(dimensions))
            field_data[place] = image

        return field_data

    def add_field_to_encoder_data(self):
        data_path = "Resources/Models/TrainingData"
        field_data = self.gen_field_data(self.get_frame())
        ordered_field_data = []
        for key in field_data.values():
            ordered_field_data.append(key)
        frame = pd.read_pickle(f"{data_path}/EncoderData.pkl")

        new_frame = pd.DataFrame({
            "enemy":[ordered_field_data[0]], "left":[ordered_field_data[1]],
            "right":[ordered_field_data[2]], "player":[ordered_field_data[3]]
        })
        frame = pd.concat([frame, new_frame])
        frame.to_pickle(f"{data_path}/EncoderData.pkl")


    # Choices
    def gen_choice_data(self, frame):
        window_dimensions = self.get_window_dimensions()

        window_bottom_left = (window_dimensions[0], window_dimensions[3])
        bottom_left_tile_location = (
            window_bottom_left[0] + (24 * self.scalars[0]),
            window_bottom_left[1] - (111 * self.scalars[1])
        )

        # 1tile - 24x20px
        screen_tile_size = (11 * self.scalars[0], 8 * self.scalars[1])
        choice_data = [None,]
        card_choices = self.get_cards(frame)
        for x in range(0, 18):
            for y in range(0, 14):
                for choice in card_choices:
                    tile_coordinates = np.array([x, y])
                    tile_screen_location = ((bottom_left_tile_location[0] + (x * screen_tile_size[0])),
                                            (bottom_left_tile_location[1] - (y * screen_tile_size[1])))
                    choice_data.append({"tile_coordinates": tile_coordinates,
                                        "tile_screen_location": tile_screen_location,
                                        "card_number": choice[0],
                                        "card_image": choice[1]
                                        })

        extra_choice_data = self.gen_north_of_bridge_choice_data(card_choices)
        choice_data.extend(extra_choice_data)
        return choice_data

    def gen_north_of_bridge_choice_data(self, card_choices):
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
                    choice_data.append({"tile_coordinates": tile_coordinates,
                                        "tile_screen_location": tile_screen_location,
                                        "card_number": choice[0],
                                        "card_image": choice[1]
                                        })
        return choice_data

    def get_cards(self, frame):
        def visualize_cards():
            num = 1
            for card in (card1, card2, card3, card4):
                card = cv2.cvtColor(np.asarray(card), cv2.COLOR_BGR2RGB)
                cv2.imshow("fdafa", np.array(card))
                cv2.waitKey(0)
                num += 1

        full_image = Image.fromarray(frame)

        # all card images are 35x43
        card1 = full_image.crop((58, 350, 93, 393))
        card2 = full_image.crop((105, 350, 140, 393))
        card3 = full_image.crop((150, 350, 185, 393))
        card4 = full_image.crop((195, 350, 230, 393))

        cards = (card1, card2, card3, card4)
        playable_cards = []
        for num in range(1, 5):
            card = cards[num - 1]
            card = np.asarray(card)
            card = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
            playable_cards.append([num, np.asarray(card)])

        return playable_cards

    # Experimental

    # Tower Health
    def gen_tower_data(self, frame):
        enemy_princess_tower_data = self.get_enemy_princess_tower_health(frame)
        enemy_king_tower_health = self.get_enemy_king_tower_health(enemy_princess_tower_data, frame)
        enemy_tower_data = (*enemy_princess_tower_data, enemy_king_tower_health)
        return enemy_tower_data

    def get_enemy_king_tower_health(self, enemy_tower_data, frame):
        if all(tower > 0 for tower in enemy_tower_data):
            return 100
        else:
            full_image = Image.fromarray(frame)
            full_image = full_image.crop((223, 28, 297, 32))

            full_image = cv2.resize(np.asarray(full_image), (51, 4), interpolation= cv2.INTER_AREA)
            full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)

            prediction = self.enemy_tower_model.predict(full_image)[0][0]
            prediction = np.round(prediction * 100, 0)
            return min(prediction, 100)

    def get_left_enemy_tower_image(self, frame):
        full_image = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
        full_image = Image.fromarray(full_image)
        return np.asarray(full_image.crop((102, 128, 153, 132)))

    def get_right_enemy_tower_image(self, frame):
        full_image = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
        full_image = Image.fromarray(full_image)
        return np.asarray(full_image.crop((364, 128, 415, 132)))

    def get_enemy_princess_tower_health(self, frame):
        left_prediction = int(
            np.round(self.enemy_tower_model.predict(self.get_left_enemy_tower_image(frame)) * 100, 0)[0])
        right_prediction = int(
            np.round(self.enemy_tower_model.predict(self.get_right_enemy_tower_image(frame)) * 100, 0)[0])
        return left_prediction, right_prediction

    def fit_enemy_princess_tower_model(self, tower_image, percentage):
        self.enemy_tower_model.fit(tower_image, percentage)

    def add_to_training_enemy_tower(health_image, percentage):
        training_data = pd.read_pickle("Resources/TrainingData/EnemyTowerHealthData.pkl")
        new_frame = pd.DataFrame({"health_percentages": [percentage], "health_images": [np.asarray(health_image)]})
        training_data = pd.concat([training_data, new_frame], sort=False)
        # training_data = training_data[training_data["health_percentages"] != 1022/1512]
        training_data.to_pickle("Resources/TrainingData/EnemyTowerHealthData.pkl")
        print(len(training_data))

    def mass_fit_enemy_princess_towers(self):
        env.save_current_frame()
        images = []
        while True:
            x = input("take photo?:")
            if x == 'no':
                break
            images.append(np.asarray(env.get_left_enemy_tower_image(env.get_frame())))
            images.append(np.asarray(env.get_right_enemy_tower_image(env.get_frame())))

        for image in images:
            cv2.imshow('fff', np.array(image))
            cv2.waitKey(0)
            x = int(input('enter health:'))
            image = image.reshape((1, 4, 51, 3))
            env.fit_enemy_princess_tower_model(image, np.array([x]))


if __name__ == '__main__':
    env = ClashRoyaleHandler()

    print(len(env.gen_choice_data(env.get_frame())))
    #print(env.get_window_dimensions()) #good window dimensions - (5, 48, 543, 997) or 538x949
    # env.start_training_game()
    # env.leave_game()
    # print(env.training_game_over())
    # print(env.determine_winner())
    # print(env.get_player_crowns())
    # print(env.get_enemy_crowns())
    # print(env.determine_winner())
    # print(len(env.gen_choice_data(env.get_frame())))
    # print(env.determine_winner())
    # print(env.get_player_crowns())
    # print(env.game_is_over())
    # print(env.check_reward_limit_reached())
    # print(env.check_for_new_reward())
    # print(env.training_game_over())
    # print(env.get_enemy_crowns())
    # print(env.get_player_crowns())
    # env.elixir_model_mass_fitting()
    # print(env.determine_winner())  355x617 - 1:1.73
    # print(env.get_window_dimensions()) # 590x1080 - 1:1.75
    # print(env.gen_tower_data(env.get_frame())) # 244x419 1:1.72
    # print(env.predict_elixir(env.get_frame()))
    # print(len(env.gen_choice_data(env.get_frame())))
