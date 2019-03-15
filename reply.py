#!/usr/local/bin/python3

import numpy as np
import collections
import heapq
import tqdm


def path_to_str(path):
    result = ''
    for i in range(len(path) - 1):
        diff = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if diff == (0, 1):
            result += 'D'
        if diff == (0, -1):
            result += 'U'
        if diff == (1, 0):
            result += 'R'
        if diff == (-1, 0):
            result += 'L'

    return result


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class Grid:
    def __init__(self, grid):
        self.grid = np.array(grid)
        self.width = self.grid.shape[1]
        self.height = self.grid.shape[0]

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.height and 0 <= y < self.width

    def not_lava(self, id):
        (x, y) = id
        return self.grid[x, y] != '#'

    def cost(self, id):
        x, y = id
        if self.grid[x, y] == '~':
            return 800
        if self.grid[x, y] == '*':
            return 200
        if self.grid[x, y] == '+':
            return 150
        if self.grid[x, y] == 'X':
            return 120
        if self.grid[x, y] == '_':
            return 100
        if self.grid[x, y] == 'H':
            return 70
        if self.grid[x, y] == 'T':
            return 50
        else:
            return 0

    def neighbors(self, id):
        # Returns a list of neighbors
        (x, y) = id
        jumps = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        results = [(x + j[0], y + j[1]) for j in jumps]
        results = filter(self.in_bounds, results)
        results = filter(self.not_lava, results)
        return list(results)


def dijkstra(graph, start, goal):
        # Search
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            value = graph.grid[next[0]][next[1]]
            if (value == 'C' or value == 'R') and next != goal:
                continue
            new_cost = cost_so_far[current] + graph.cost(next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current

    # Reconstruction
    if current == goal:
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    return [0, 0]


def main():
    n, m, c, r = list(map(int, f.readline().rsplit()))
    print(n, m, c, r)
    customers = []
    for _ in range(c):
        x, y, reward = list(map(int, f.readline().rsplit()))
        customers.append((y, x, reward))

    matrix = []

    for _ in range(m):
        matrix.append(list(f.readline().rstrip()))
    graph = Grid(matrix)

    for c in customers:
        graph.grid[c[0]][c[1]] = 'C'

    customers = sorted(customers, key=lambda x: x[2], reverse=True)
    print(customers)
    replies = []
    for i in range(0, r):
        neighbors = graph.neighbors(customers[i][0:2])
        if len(neighbors) > 0:
            neighbors = [n for n in neighbors if graph.grid[n[0]][n[1]] != 'C' and graph.grid[n[0]][n[1]] != 'R']
            replies.append(neighbors[0])
            graph.grid[c[0]][c[1]] = 'R'

    for customer in customers:
        for reply in replies:
            path = dijkstra(graph, reply, customer[0:2])
            print(reply[0], reply[1], path_to_str(path))
            output.write(f'{reply[0]} {reply[1]} {path_to_str(path)}\n')


if __name__ == '__main__':
    inputs = ['2_himalayas', '4_manhattan', '1_victoria_lake', '3_budapest', '5_oceania']
    for i in inputs:
        f = open(i + '.txt', 'r')
        output = open(i + '.out', 'w')
        main()
