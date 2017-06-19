from sys import argv, stdin
import os, csv
from PIL import Image
import numpy as np

CELLS_WIDTH = 20
CELLS_HEIGHT = 20
# for terminal image: (255,190,38)
ALIVE_COLOR = (255,255,0)

if len(argv) != 2:
    print "Usage: python %s file_name" % argv[0]
    exit(1)

# Read image
try:
    with Image.open(argv[1], 'r') as image:
        rgb_image = image.convert('RGB')
        imageWidth, imageHeight = image.size
        cellHeight = imageHeight / CELLS_HEIGHT
        cellWidth = imageWidth / CELLS_WIDTH
except Exception as e:
    print "Failed to open the image: " + str(e)
    exit(1)

# Create output file
outputFilename = os.path.splitext(argv[1])[0] + '.csv'

try:
    with open(outputFilename, 'w+') as outputCSV:
        outputCSVwriter = csv.writer(outputCSV)
        outputCSVwriter.writerow(['id', 'delta'] + ['stop.' + str(i + 1) for i in range(CELLS_HEIGHT * CELLS_WIDTH)])
        color = lambda x: 1 if x == ALIVE_COLOR else 0
        # id 1, steps 1
        data = [1, 1]
        for i in range(CELLS_HEIGHT):
            for j in range(CELLS_WIDTH):
                pixel_h = int(cellHeight * (i + 0.5))
                pixel_w = int(cellWidth * (j + 0.5))
                pixel = rgb_image.getpixel((pixel_w, pixel_h))
                data += [color(pixel)]
        outputCSVwriter.writerow(data)
except Exception as e:
    print "Failed to create the csv file: " + str(e)
    exit(1)
