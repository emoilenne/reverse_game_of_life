import csv
from sys import argv

CELLS_WIDTH = 20
CELLS_HEIGHT = 20

if len(argv) != 3:
    print "Usage: python %s file_name" % argv[0]
    exit(1)

try:
    with open(argv[1], 'r') as outputCSV:
        outputCSVreader = csv.reader(outputCSV)
        for skip in range(int(argv[2])):
            outputCSVreader.next()
        format_cell = lambda x: '[X]' if x == '1' else '   '
        data = outputCSVreader.next()[1:]
        train_file = len(data) == 1 + 2 * CELLS_HEIGHT * CELLS_WIDTH
        test_file = len(data) == 1 + CELLS_HEIGHT * CELLS_WIDTH
        if not train_file and not test_file:
            raise Exception("The board should be 20x20")
        if train_file:
            print "Start board:"

        for i in range(CELLS_HEIGHT):
            row = ''
            for j in range(CELLS_WIDTH):
                row += format_cell(data[1 + i * CELLS_WIDTH + j])
            print row

        if train_file:
            print "Stop board in %s steps:" % data[0]
            for i in range(CELLS_HEIGHT):
                row = ''
                for j in range(CELLS_WIDTH):
                    row += format_cell(data[1 + (i + CELLS_HEIGHT) * CELLS_WIDTH + j])
                print row


except Exception as e:
    print "An error occured: " + str(e)
    exit(1)
