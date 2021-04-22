from solver import solve
import sys
import getopt


def main():

    name = None
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "n:")
    except:
        print("Correct usage: solve_puzzle.py -n <name>")

    for opt, arg in opts:
        if opt in ['-n']:
            name = arg

    print(name)
    solve(name)


main()
