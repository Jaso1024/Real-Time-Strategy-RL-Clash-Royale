
import numpy as np
import pandas as pd
import tensorflow
import keras
import cv2
import PIL
import keyboard as kb
import win32con
import win32com
import win32gui
import pyautogui
import absl

imports = [np, pd, tensorflow, keras, cv2, PIL, kb, win32con, win32com, win32gui, pyautogui, absl]
for impo in imports:
    try:
        print(f'- {impo.__name__} {impo.__version__}')
    except:
        print("- " + impo.__name__)
        continue

