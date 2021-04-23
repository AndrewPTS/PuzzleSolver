import cv2
from jigsaw import cut_puzzle, scramble_pieces
from solver import solve
import sys
import getopt


def main():

    name = None
    filename = None
    g = 10
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "f:n:g:")
    except:
        print("Correct usage: create_puzzle.py -f <filename> -n <name> -g <grid size>")

    for opt, arg in opts:

        if opt in ['-n']:
            name = arg
        elif opt in ['-f']:
            filename = arg
        elif opt in ['-g']:
            g = int(arg)

    print(filename)
    img = cv2.imread(filename)
    img = cv2.resize(img, (round(img.shape[1]), round(img.shape[0])))

    res = None
    while res is None:
        cut_puzzle(img, g, name)
        scramble_pieces(name)
        res = solve(name)


main()