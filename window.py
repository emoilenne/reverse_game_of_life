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
    ROTATE90:             lambda window: np.array([window[w,h] for h in range(len(window)) for w in range(len(window) - 1, -1, -1)]).reshape(window.shape),
    ROTATE180:            lambda window: window[:,::-1][::-1],
    ROTATE270:            lambda window: np.array([window[w,h] for h in range(len(window) - 1, -1, -1) for w in range(len(window))]).reshape(window.shape),
    FLIP_DIAGONAL1:       lambda window: np.array([window[w,h] for h in range(len(window)) for w in range(len(window))]).reshape(window.shape),
    FLIP_DIAGONAL2:       lambda window: np.array([window[w,h] for h in range(len(window) - 1, -1, -1) for w in range(len(window) - 1, -1, -1)]).reshape(window.shape),
    }


class Window:

    def __init__(self, grid, startPoint, size):
        """
            startPoint, height and width define the window from initial grid.
            The window will be size * size cells.
            If dimensions exceed the grid, the exception will be raised.
        """
        gridHeight, gridWidth = grid.shape
        startPointHeight, startPointWidth = startPoint
        endPointHeight = startPointHeight + size
        endPointWidth = startPointWidth + size

        self.size = size
        self.data = grid[startPointHeight:endPointHeight, startPointWidth:endPointWidth]

    def toHash(self):
        """
            Calculate the hash of the window and how it was rotated to obtain this hash.
            Return in a form (transformation, hash)
        """

        hashWindow = np.array([2 ** x for x in range(self.size ** 2)]).reshape(self.size, self.size)
        allStates = {state: func(self.data) for state, func in Transformation.do.items()}
        hashes = {state: sum(sum(window * hashWindow)) for state, window in allStates.items()}
        return min(hashes.items(), key=lambda p: p[1])
