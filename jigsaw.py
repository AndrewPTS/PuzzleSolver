import cv2
import numpy as np
import math
import random
from pathlib import Path
import os
import glob
from shutil import copyfile

def divide_puzzle(img, scale_factor):
    cv2.imshow("preview", img)
    cv2.waitKey()
    height, width, _ = img.shape
    coord_height, coord_width = math.ceil(height/scale_factor), math.ceil(width/scale_factor)
    if coord_height%2 == 0:
        coord_height += 1
    if coord_width%2 == 0:
        coord_width += 1
    intersect_coords = np.zeros((coord_height, coord_width, 2), dtype=int)

    # first set intersection points in a rectagular grid
    for i in range(0, coord_width):
        if i * scale_factor >= width or i == coord_width-1:
            intersect_coords[0][i][0] = int(width-1)
        else:
            intersect_coords[0][i][0] = int(i * scale_factor)
    for i in range(1, coord_height):
        # first copy row to have same x coords
        intersect_coords[i] = intersect_coords[0]
        for j in range(0, coord_width):
            if i * scale_factor >= height or i == coord_height-1:
                intersect_coords[i][j][1] = int(height-1)
            else:
                intersect_coords[i][j][1] = int(i * scale_factor)


    straight_lines = img.copy()
    for i in range(0, coord_height):
        for j in range(0, coord_width):
            if i+1 < coord_height:
                lines = cv2.line(straight_lines, tuple(intersect_coords[i][j]), tuple(intersect_coords[i+1][j]),
                                (255, 255, 255), 2)
            if j+1 < coord_width:
                lines = cv2.line(straight_lines, tuple(intersect_coords[i][j]), tuple(intersect_coords[i][j+1]),
                                 (255, 255, 255), 2)
    cv2.imshow("grid", lines)
    cv2.waitKey()

    # now randomize locations in set area
    # if either index is odd, it is meant to be a m/f point
    random_factor = math.floor(scale_factor/2)
    offset_min = math.ceil(random_factor/4)
    for i in range(0, coord_height):
        for j in range(0, coord_width):
            if not(i%2==0 and j%2==0):
                if 0 < i < coord_height-1:
                    sign = random.randint(1, 2)
                    if i % 2 == 1:
                        # offset = (-1**sign)*random.randint(0, offset_min)
                        offset = 0
                    else:
                        offset = (-1**sign)*random.randint(offset_min, random_factor)
                    intersect_coords[i][j][1] = intersect_coords[i][j][1] + offset

                if 0 < j < coord_width-1:
                    sign = random.randint(1, 2)
                    if j % 2 == 1:
                        # offset = (-1**sign)*random.randint(0, offset_min)
                        offset = 0
                    else:
                        offset = (-1**sign)*random.randint(offset_min, random_factor)
                    intersect_coords[i][j][0] = intersect_coords[i][j][0] + offset

    lines = img.copy()
    for i in range(0, coord_height):
        for j in range(0, coord_width):
            if i+1 < coord_height:
                lines = cv2.line(lines, tuple(intersect_coords[i][j]), tuple(intersect_coords[i+1][j]),
                                (255, 0, 255), 2)
            if j+1 < coord_width:
                lines = cv2.line(lines, tuple(intersect_coords[i][j]), tuple(intersect_coords[i][j+1]),
                                 (255, 0, 255), 2)
    cv2.imshow("pieces", lines)
    cv2.waitKey()
    return intersect_coords

def cut_pieces(img, intersect_coords, title):
    path = title + "/pieces"
    Path(path).mkdir(parents=True, exist_ok=True)
    remove_old_files(path)

    coord_height, coord_width, _ = intersect_coords.shape
    for row in range(0, coord_height-2, 2):
        for col in range(0, coord_width-2, 2):
            points = np.array([intersect_coords[row][col], intersect_coords[row][col+1],
                              intersect_coords[row][col+2], intersect_coords[row+1][col+2],
                              intersect_coords[row+2][col+2], intersect_coords[row+2][col+1],
                              intersect_coords[row+2][col], intersect_coords[row+1][col]])
            rect = cv2.boundingRect(points)
            x,y,w,h = rect
            cropped = img[y:y+h, x:x+w].copy()
            points = points - points.min(axis=0)
            mask = np.zeros(cropped.shape[:2], np.uint8)
            cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
            piece = cv2.bitwise_and(cropped, cropped, mask=mask)
            # add background
            bg_color = [255, 255, 255]
            bg = np.full(piece.shape, bg_color, np.uint8)
            bg_mask = cv2.bitwise_not(mask)
            fg_masked = cv2.bitwise_and(piece,piece, mask=mask)
            bg_masked = cv2.bitwise_and(bg, bg, mask=bg_mask)
            piece = cv2.bitwise_or(fg_masked, bg_masked)
            # border makes edge detection a lot more accurate around edges of the piece
            piece = cv2.copyMakeBorder(piece, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=bg_color)

            # save piece to puzzle directory
            filename = path+"/"+str(row) + "_" + str(col)+".png"
            cv2.imwrite(filename, piece)

def cut_puzzle(img, scale_factor, title):
    cut_pieces(img, divide_puzzle(img, scale_factor), title)

def scramble_pieces(title):
    piece_path = title+"/pieces"
    if not os.path.exists(piece_path):
        print("the puzzle pieces have not been made yet!")
        return

    path = title+"/scrambled"
    Path(path).mkdir(parents=True, exist_ok=True)
    remove_old_files(path)

    pieces = glob.glob(piece_path+"/*")
    random.shuffle(pieces)

    i = 0
    for piece in pieces:
        filename = path + "/" + str(i) + ".png"
        copyfile(piece, filename)
        i = i + 1

def remove_old_files(path):
    old_files = glob.glob(path+"/*")
    for file in old_files:
        os.remove(file)