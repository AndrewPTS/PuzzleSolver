# PuzzleSolver

###Introdution  
PuzzleSolver can solve simplified 'jigsaw puzzles' by generating information about every piece
and use it to find the adjacent pieces. It also does not need a reference image! 
At this stage of the development, it uses simplified pieces
that are not curved and do not have round pegs/holes.

Some sample puzzles have been included, along with a generator for simplified jigsaw puzzles is included. To distribute
these pieces to other people, make sure to share the directory with the name of the puzzle.
It is possible to delete the pieces folder in said directory, but the scrambled
pieces must stay in the 'scrambled' directory.

###Usage  
####Required Libraries:
NumPy
PIL
scikit-learn

####create_puzzle: create_puzzle.py -f <filename> -n <name> -g <grid size>
-f: filename of image  
-n: name of the puzzle (does not need to match filename)  
-g: grid size 

Typically, grid size roughly correlates to double the number of 
pieces along the x axis. For a ~1000 pixel wide image, a value
around 10-20 works well. Lower numbers have better results, but 
obviously fewer pieces.
  
####solve_puzzle: solve_puzzle.py -n <name>
-n: name of the puzzle

The name should correspond to the directory containing a 'scrambled' 
directory containing the scrambled pieces.
