import pyautogui as pg
import random as rnd

UP_BUTTON = 'w'
LEFT_BUTTON = 'a'
DOWN_BUTTON = 's'
RIGHT_BUTTON = 'd'
JUMP_BUTTON = 'space'
LIGHT_BUTTON = 'j'
HEAVY_BUTTON = 'k'
DODGE_BUTTON = 'l'


def move_left_and_attack(x, y):
    print("L", x, y)
    pg.keyDown(LEFT_BUTTON)
    pg.press(RIGHT_BUTTON)
    r = rnd.randint(0, 1)
    if r == 0:
        pg.press(LIGHT_BUTTON)
    elif r == 1:
        pg.press(HEAVY_BUTTON)


def move_right_and_attack(x, y):
    print("R", x, y)
    pg.keyDown(RIGHT_BUTTON)
    pg.press(LEFT_BUTTON)
    r = rnd.randint(0, 1)
    if r == 1:
        pg.press(LIGHT_BUTTON)
    elif r == 0:
        pg.press(HEAVY_BUTTON)


def jump_to_stage_and_recovery(x, y, left, right):
    print("J", x, y)
    pg.keyDown(UP_BUTTON)
    pg.press(JUMP_BUTTON)
    if x <= left:  # jump right
        pg.keyDown(RIGHT_BUTTON)
        pg.press(LEFT_BUTTON)
        pg.press(DODGE_BUTTON)
    elif x >= right:  # jump left
        pg.keyDown(LEFT_BUTTON)
        pg.press(RIGHT_BUTTON)
        pg.press(DODGE_BUTTON)
    pg.press(HEAVY_BUTTON)
    pg.keyUp(UP_BUTTON)


def dsig(x, y):
    print('dsig', x, y)
    pg.keyUp(RIGHT_BUTTON)
    pg.keyUp(LEFT_BUTTON)
    pg.keyDown(DOWN_BUTTON)
    pg.press(HEAVY_BUTTON)
