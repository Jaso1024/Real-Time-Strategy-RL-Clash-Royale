# ClashRoyaleBotV3
This project contains code that trains several neural networks, with the end goal of playing against and defeating the opponent of a Training match in Supercell's Clash Royale. The technique used is [Proximal-Policy-Optimization](https://arxiv.org/abs/1707.06347) combined with [multi-agent reinforcement learning](https://arxiv.org/abs/1911.10635) and also makes use of both [state abstraction](https://ala2021.vub.ac.be/papers/ALA2021_paper_50.pdf) and [action-space factoring](https://arxiv.org/abs/1705.07269).

#### NOTE: Changing the code in order to use this bot for competetive Clash Royale games is a violation of Supercell's TOS. 

# Methodology

## Overview
In this project, there are 3 agents that work together to interact with the environment. One chooses the cards, another chooses 1 of 48 tiles, and the last one chooses 1 of 9 tiles. Each of these 3 agents are comprised of an actor and critic and is trained using PPO.

## Inputs
At each timestep t, the current state of the game will be broken into 9 parts:
- An image of the top part of the field downscaled to 51x33
- An image of the bottom part of the field downscaled to 51x32
- 2 images, the left and right bridges respectively, both downscaled to 4x6
- An estimate of the current amount of elixir the player has (provided by another neural network).
- 4 one-hot encoded vectors, each corresponding to a card currently in the players hand.

these 9 parts are passed through the encoding half of an autoencoder, which returns a single vector, which will be given to each of the 3 agents.

## Actions
At every timestep t, each of the 3 agent's actors will choose an action, these individual actions will be combined and a single action will be excecuted in the environment.

### Cards
The agent which chooses the cards will return a softmax distribution, each probability cooresponding to a card currently available to the player.

### Tiles
In Clash Royale there are hundreds of locations in which a card can be placed, in order to reduce the amount of training time needed, this project splits the location in which the chosen card will be placed, into 2 parts, origin and shell.

#### Origin
In order for a tile to be considered an origin tile it must meet the following requirements:

1. The tile is not on the edge of the grid
2. The tile is not next to another origin tile

#### Shell 
In order for a tile to be considered a shell tile it must be directly touching the side or corner of an origin tile

2 examples of the tile types are shown below, the origin tiles are marked in red, shell tiles are marked in white, other tiles are black and are not included in the action space.

![3x3im](https://user-images.githubusercontent.com/107654508/189499330-8d94b262-8a3e-4c7d-a4df-eb41675d40da.png)
![imageedit_24_3998630696_320x320](https://user-images.githubusercontent.com/107654508/189499669-e552f3ec-446e-4f0c-afa7-da29b7b30272.png)

## Reward function
The agent gets the following rewards:
- Step reward: `+0.1`
- Win reward: `+300`
- Loss reward: `-200`
### Crowns
The rewards for crowns lost and crowns won follow the function below:

`(4.9*log(4.8*(number of crowns player has won) + 0.75) + 1.4) - (4.9*log(4.8*(number of crowns player has lost) + 0.75) + 1.4)`


This function gives the agents a larger reward the bigger the difference between the number of crowns the player has won and lost is, and results in a maximum reward of `+15` and a minimum reward of `-15`.
 
## Model architecture
Overview of the network:



## Installation

## Dependencies
- numpy v1.23.2
- pandas v1.4.3
- tensorflow v2.9.1
- keras v2.9.0
- cv2 v4.6.0
- PIL v9.2.0
- keyboard v0.13.5
- pywin32 v304
- pyautogui v0.9.53
- absl-py v1.2.0

## Installing Bluestacks
1: Navigate to the Bluestacks website - https://www.bluestacks.com/

2: Download and follow the installation steps for Bluestacks 5

####Note: Bluestacks 10 will not work with this program

## Installing Clash Royale
1: After launching the BlueStacks App Player, navigate to the play store

2: Once at the Play Store, search for and install Clash Royale


# Setup
Before running the program, certain modifications must be made to the configuration of the BlueStacks App Player and game

Side Bar: Once the App Player has been launched a thin bar may be seen on the window's right side, this bar can be identified by the 2 arrows at the top of it
make sure to collapse this bar by clicking the 2 arrows pointing left on the upper-right hand corner.

Deck: Make sure that the deck being used is the same as the one below

![image](https://user-images.githubusercontent.com/107654508/189466078-d9dd5956-696c-4fc8-8bd7-32d270113b9d.png)


Window Size: This program works better with bigger window sizes, make sure it is enlarged to a reasonable size

Window Location: This program uses snapshots of the computers screen in order to get each frame of the game, please make sure the BlueStacks App Player window is completely visible


# Running the agent







