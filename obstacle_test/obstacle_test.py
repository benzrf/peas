import peas.methods.neat as neat
import peas.networks.rnn as rnn
import numpy as np
import random
from itertools import *
from math import sqrt, floor
from PIL import Image

def magnitude(v):
    x, y = v
    return sqrt(x**2 + y**2)

class ObstacleCourse:
    x, y  = 500, 0
    steps = 0

    def __init__(self, grid, net):
        self.grid, self.net = grid, net

    def tick(self):
        inputs = np.array(self.dists)
        out = self.net.feed(inputs)
        vx, vy = out[-2:]
        self.x = min(999, max(0, self.x + vx))
        self.y = min(999, max(0, self.y + vy))
        self.steps += 1

    def collides(self):
        return 0 in self.dists

    @property
    def dists(self):
        x, y = floor(self.x), floor(self.y)
        return self.grid[x][y]

class ObstacleCourseTask:
    def __init__(self, grids):
        self.grids = grids

    def evaluate(self, net):
        if not isinstance(net, neat.NeuralNetwork):
            net = neat.NeuralNetwork(net)
        solved, scores = True, []
        for grid in self.grids:
            course = ObstacleCourse(grid, net)
            while course.steps < 1000 and course.y < 999 and not course.collides():
                course.tick()
            if course.y != 999:
                solved = False
            scores.append(course.y)
        scores.sort()
        weights = range(len(scores), 0, -1)
        avg = sum(w * s for w, s in zip(weights, scores)) / sum(weights)
        return {'fitness': avg, 'solved': solved}

    def solve(self, net):
        return self.evaluate(net)['solved']

def loadppm(f):
    if not hasattr(f, 'read'):
        with open(f, 'rb') as f_:
            return loadppm(f_)
    header_lines = 0
    while header_lines < 3:
        if f.readline()[0] != 35: # b'#'[0] == 35
            header_lines += 1
    grid = []
    for _ in range(1000):
        row = []
        for _ in range(1000):
            row.append(f.read(3) == b'\x00\x00\x00')
        grid.append(row)
    return np.array(grid).T.tolist()

DIRS = [(0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1)]

def calcdists(grid):
    newgrid = [[[9999] * 8 for _ in range(1000)] for _ in range(1000)]
    for ix, (dx, dy) in enumerate(DIRS):
        starts_rows = {0: [],
                       1: product([0], range(1000)),
                       -1: product([999], range(1000))}[dx]
        starts_cols = {0: [],
                       1: product(range(1000), [0]),
                       -1: product(range(1000), [999])}[dy]
        for x, y in chain(starts_rows, starts_cols):
            dist = 0
            while 0 <= x < 1000 and 0 <= y < 1000:
                if grid[x][y]:
                    dist = 0
                else:
                    dist += 1
                newgrid[x][y][ix] = dist
                x += dx
                y += dy
    return newgrid

def do_evo(gens):
    files = 'course1.ppm course2.ppm course3.ppm course4.ppm course5.ppm'.split()
    grids = [calcdists(loadppm(f)) for f in files]
    task = ObstacleCourseTask(grids)
    factory = lambda: neat.NEATGenotype(inputs=8, outputs=2)
    pop = neat.NEATPopulation(factory)
    try:
        results = pop.epoch(generations=gens, evaluator=task, solution=task)
    except KeyboardInterrupt:
        results = None
    return (results, pop)

def trace(net, f, steps):
    if not isinstance(net, neat.NeuralNetwork):
        net = neat.NeuralNetwork(net)
    grid = calcdists(loadppm(f))
    course = ObstacleCourse(grid, net)
    im = Image.open(f)
    pix = im.load()
    for _ in range(steps):
        course.tick()
        if course.collides():
            break
        pix[course.x, course.y] = (255, 0, 0)
    return im

