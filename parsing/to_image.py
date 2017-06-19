import csv
from sys import argv

CELLS_WIDTH = 20
CELLS_HEIGHT = 20
ALIVE_COLOR = '\x1b[5;30;43m'
NOCOLOR = '\x1b[0m'

if len(argv) != 2:
    print "Usage: python %s file_name" % argv[0]
    exit(1)

try:
    with open(argv[1], 'r') as outputCSV:
        outputCSVreader = csv.reader(outputCSV)
        outputCSVreader.next()
        color = lambda x: ALIVE_COLOR if x == '1' else NOCOLOR
        # id 1, steps 1
        data = outputCSVreader.next()[2:]
        for i in range(CELLS_HEIGHT):
            row = ''
            for j in range(CELLS_WIDTH):
                row += color(data[i * CELLS_HEIGHT + j]) + "  " + NOCOLOR

            print row

except Exception as e:
    print "Failed to open the csv file: " + str(e)
    exit(1)
