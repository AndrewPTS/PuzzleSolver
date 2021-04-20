import cv2
from jigsaw import cut_puzzle, scramble_pieces
from solver import solve

img = cv2.imread("rose.jpg")
img = cv2.resize(img, (int(img.shape[1] * .6), int(img.shape[0] * .6)))
cut_puzzle(img, 150, "rose")
scramble_pieces("rose")
solve("rose")