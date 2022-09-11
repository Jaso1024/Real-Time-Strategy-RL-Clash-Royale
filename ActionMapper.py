import numpy as np
import tensorflow as tf


class ActionMapper:
    def get_origin_square_locations(self):
        locations = []
        for x in range(1, 18-1, 2):
            for y in range(1, 14-1, 2):
                locations.append((x,y))
        return locations
    
    def get_tile(self, tile_of_nine):
        tile_mappings = {1:(-1,-1), 2:(-1,0), 3:(-1,1),
                         4:(0,-1), 5:(0,0), 6:(0,1),
                         7:(1,-1), 8:(1,0), 9:(1,1)}
        return tile_mappings[tile_of_nine+1]

    def make_action(self, tile, card_num):
        if type(tile) == dict:
            action = {}
            for key, value in tile.items():
                if key == 'card':
                    continue
                else:
                    action[key] = value
            
            action['card_number'] = card_num
        else:
            action = tile
        return action
    
    def transform_card_data(self, card_data):
        cards = []
        for elem in card_data:
            cards.append(elem[1])

        return cards

    def get_action(self, action_components, choices, card_data):
        action = {}

        card_data = self.transform_card_data(card_data)

        origin_squares_data = []
        tile_matrix = self.to_matrix(choices)

        for x, y in self.get_origin_square_locations():
            origin_squares_data.append([x, y])

        square_num = tf.squeeze(action_components[0])
        square_num = int(tf.get_static_value(square_num))
        
        if square_num == 48:
            return None
            
        origin_tile_location = origin_squares_data[square_num]

        tile_component = tf.squeeze(action_components[1])
        tile_component = int(tf.get_static_value(tile_component))
        tile_component = self.get_tile(tile_component)
        tile_location = (origin_tile_location[0] + tile_component[0], origin_tile_location[1] + tile_component[1])
        tile = tile_matrix[tile_location]

        card_num = tf.squeeze(action_components[2])
        card_num = int(tf.get_static_value(card_num))
        if card_data[card_num].flatten()[0] == 1:
            card_num = None
        action = self.make_action(tile, card_num)
        return action
        
    def to_matrix(self, choices):
        current_idx = 0
        choice_matrix = []
        for x in range(18):
            choice_vector = []
            for y in range(14):
                choice_vector.append(choices[current_idx])
                current_idx += 1
            choice_matrix.append(choice_vector)
        return np.array(choice_matrix)