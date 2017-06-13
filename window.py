import numpy as np

class Transformation:
    # States of windows
    ORIGINAL = 'original'
    FLIP_VERTICALLY = 'flip_vertically'
    FLIP_HORIZONTALLY = 'flip_horizontally'
    ROTATE90 = 'rotate90'
    ROTATE180 = 'rotate180'
    ROTATE270 = 'rotate270'
    FLIP_DIAGONAL1 = 'flip_diagonal1'
    FLIP_DIAGONAL2 = 'flip_diagonal2'

    # Transformation functions
    do = {
    ORIGINAL:             lambda window: window,
    FLIP_VERTICALLY:      lambda window: window[::-1],
    FLIP_HORIZONTALLY:    lambda window: window[:,::-1],
    ROTATE90:             lambda window: np.rot90(window),
    ROTATE180:            lambda window: window[:,::-1][::-1],
    ROTATE270:            lambda window: np.rot90(window[:,::-1][::-1]),
    FLIP_DIAGONAL1:       lambda window: np.rot90(window[::-1]),
    FLIP_DIAGONAL2:       lambda window: np.rot90(window[:,::-1])
    }


class Window:

    def __init__(self, grid, startPoint=(0,0), size=4):
        """
            startPoint, height and width define the window from initial grid.
            The window will be size * size cells.
            If dimensions exceed the grid, the exception will be raised.
        """
        gridHeight, gridWidth = grid.shape
        startPointHeight, startPointWidth = startPoint
        endPointHeight = startPointHeight + size
        endPointWidth = startPointWidth + size

        # Check if dimensions of window exceed the grid
        if endPointWidth > gridHeight or endPointWidth > gridWidth:
            raise Exception("Window dimensions exceed the grid")

        # Create array from the grid with specified dimensions
        self.size = size
        self.data = grid[startPointHeight:endPointHeight, startPointWidth:endPointWidth]

    def toHash(self):
        """
            Calculate the hash of the window and how it was rotated to obtain this hash.
            Return in a form (transformation, hash)
        """

        # Hash window represents a window with same dimentions as current window with values 1, 2, 4, 8...
        hashWindow = np.array([2 ** x for x in range(self.size ** 2)]).reshape(self.size, self.size)

        # Create dictionary with all possible transformations of current window
        allStates = {state: func(self.data) for state, func in Transformation.transform}

        # Calculate each hash value for transformated windows
        hashes = {state: sum(window * self.hashWindow) for state, window in allStates.items()}

        # Return minimum hash from all possible transformations
        return min(hashes.items(), key=lambda p: p[1])
