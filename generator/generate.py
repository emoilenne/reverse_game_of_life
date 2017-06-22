import numpy as np
from map import Map
import csv, time

MAPS = 1000000
HEIGHT = 20
WIDTH = 20
FILE = '../csv/gtrain.csv'

TOTAL = HEIGHT * WIDTH

mapgen = Map(HEIGHT, WIDTH)
with open(FILE, 'w+') as output:
    outputWriter = csv.writer(output)
    outputWriter.writerow(['id', 'delta'] + ['start.' + str(i + 1) for i in range(TOTAL)] +
                                            ['stop.' + str(i + 1) for i in range(TOTAL)])
    totalTimeStart = time.time()
    for id in range(MAPS):
        timeStart = time.time()
        while True:
            row = [str(id + 1)]
            mapgen.generate()
            steps = mapgen.getSteps()
            row.append(str(steps))
            row.extend(mapgen.getValues())
            mapgen.step(steps)
            row.extend(mapgen.getValues())
            if mapgen.aliveCells() != 0:
                break
        outputWriter.writerow(row)
        timeEnd = time.time()
        print("Generating map #%d took %.3f ms" % (id, (timeEnd - timeStart) * 1000.))
    totalTimeEnd = time.time()
    print("Generating maps took %.3f s or %.3f min" % (id, timeEnd - timeStart, (timeEnd - timeStart) / 60.))
