import heapq
import time
import argparse
import numpy as np
from math import sqrt
from random import randint
from typing import Tuple, Set, Dict, List, FrozenSet
import matplotlib.pyplot as plt

INF = 99999


#   euclidean distance between two cities
def distance(start: Tuple[int, int], arrive: Tuple[int, int]) -> float:
    return sqrt((arrive[0] - start[0]) ** 2 + (arrive[1] - start[1]) ** 2)


#  a state is defined by a set of visited cities and the current city
class State:
    def __init__(self, visited_cities: FrozenSet[Tuple[int, int]], current_city: Tuple[int, int]):
        self.visited_cities: FrozenSet[Tuple[int, int]] = visited_cities
        self.current_city: Tuple[int, int] = current_city

    def __eq__(self, other):
        if type(other) is type(self):
            return (self.visited_cities == other.visited_cities) and (self.current_city == other.current_city)
        else:
            return False

    def __hash__(self):
        return hash((self.visited_cities, self.current_city))


class Node:
    def __init__(self, state: State, parent, path_cost: float):
        self.state: State = state
        self.parent: Node = parent
        self.path_cost = path_cost


class Problem:
    def __init__(self, map_size: int, num_cities: int, cities: List[Tuple[int, int]]):
        if num_cities > map_size ** 2:
            raise ValueError("Too many cities for the map size")

        self.map_size = map_size
        self.num_cities = num_cities

        #  initialize random cities
        self.cities_indexes: Dict[Tuple[int, int], int] = dict()
        self.initial_city: Tuple[int, int] = (-1, -1)
        if len(cities) == 0:
            self.cities: FrozenSet[Tuple[int, int]] = self.init_random_cities()
        else:
            self.cities: FrozenSet[Tuple[int, int]] = self.init_cities(cities=cities)

        self.adjacency_matrix: np.ndarray = self.init_adjacency_matrix(cities=self.cities)

        self.initial_state = State(visited_cities=frozenset({self.initial_city}), current_city=self.initial_city)

        # the goal state is reached when we have visited all cities and the current city is the initial city
        self.goal_state = State(visited_cities=self.cities, current_city=self.initial_state.current_city)

    def init_random_cities(self):
        tmp: Set[Tuple[int, int]] = set()
        city_index = 0
        while len(tmp) < self.num_cities:
            x = randint(0, self.map_size)
            y = randint(0, self.map_size)
            #  check if a city in position (x,y) already exists
            if not (x, y) in tmp:
                if city_index == 0:
                    self.initial_city = (x, y)
                tmp.add((x, y))
                self.cities_indexes[(x, y)] = city_index
                city_index += 1
        res = frozenset(tmp)
        return res

    def init_cities(self, cities: List[Tuple[int, int]]):
        tmp: Set[Tuple[int, int]] = set()
        for i, city in enumerate(cities):
            if i == 0:
                self.initial_city = city
            tmp.add(city)
            self.cities_indexes[city] = i
        res = frozenset(tmp)
        return res

    def init_adjacency_matrix(self, cities: [(int, int)]) -> np.ndarray:
        res = np.zeros(shape=(len(self.cities), len(self.cities)), dtype=float)
        for i, start_city in enumerate(cities):
            for j, arrive_city in enumerate(cities):
                if res[i, j] == 0 and i != j:
                    d = distance(start=start_city, arrive=arrive_city)
                    res[i, j] = d
                    res[j, i] = d
        return res

    def isGoal(self, state: State):
        if self.goal_state.current_city == state.current_city and self.goal_state.visited_cities == state.visited_cities:
            return True
        else:
            return False

    def cost(self, start_state: State, arrive_state: State):
        return distance(start=start_state.current_city, arrive=arrive_state.current_city)

    #  returns all the states reachable from a given state, this is as a helper method for 'expand'
    def available_cities(self, state: State) -> Set[State]:
        res: Set[State] = set()
        #  if we have already visited all cities we must return to the initial city
        if state.visited_cities == self.cities:
            res.add(State(visited_cities=state.visited_cities, current_city=self.initial_state.current_city))
        else:
            visitable_cities = self.cities - state.visited_cities
            for city in visitable_cities:
                res.add(State(visited_cities=state.visited_cities.union({city}), current_city=city))
        return res

    def expand(self, node: Node) -> Set[Node]:
        res: Set[Node] = set()
        available_states = self.available_cities(state=node.state)
        for state in available_states:
            cost = node.path_cost + self.cost(start_state=node.state, arrive_state=state)
            res.add(Node(state=state, parent=node, path_cost=cost))
        return res

    #  helper method for building mst

    def min_distances(self, distances, mst_set, unvisited_cities_indexes) -> [int, float]:
        min_distance = INF
        min_index = -1
        for v in unvisited_cities_indexes:
            if distances[v] < min_distance and mst_set[v] is False:
                min_distance = distances[v]
                min_index = v
        return min_index, min_distance

    #  the heuristic function returns the weight of the mst formed by unvisited cities
    #  plus the minimum distance from unvisited cities to the initial city
    def heuristic(self, node: Node) -> float:

        visited_cities = node.state.visited_cities
        unvisited_cities = self.cities - visited_cities
        if (len(unvisited_cities)) == 0:
            return 0

        #  ordered list of the unvisited cities' indexes
        unvisited_cities_indexes: List[int] = []
        for unvisited_city in unvisited_cities:
            unvisited_cities_indexes.append(self.cities_indexes[unvisited_city])
        unvisited_cities_indexes.sort()

        #  index of the problem's initial city
        initial_city_index: int = self.cities_indexes[self.initial_state.current_city]

        #  minimum distance from unvisited cities to the initial city
        min_dist_to_initial_city = INF
        for unvisited_city_index in unvisited_cities_indexes:
            d = self.adjacency_matrix[unvisited_city_index][initial_city_index]
            if d < min_dist_to_initial_city:
                min_dist_to_initial_city = d

        #  initialize the set of nodes in mst
        mst_set = [False for _ in range(len(self.cities))]

        #  initialize distances of nodes from current mst
        distances = [INF for _ in range(len(self.cities))]

        #  select node at index 0 as the first node of mst
        distances[0] = 0

        mst_weight = 0
        for _ in range(len(unvisited_cities)):
            #  u is the node with minimum distance from current mst
            u, dist = self.min_distances(distances=distances, mst_set=mst_set,
                                         unvisited_cities_indexes=unvisited_cities_indexes)

            #  insert the node in mst and update mst weight
            mst_set[u] = True
            mst_weight += dist

            # updates the distances from the current mst of nodes that are not in mst yet
            # and that can be reached by u at a cost less than the cost in memory
            for v in unvisited_cities_indexes:
                if mst_set[v] is False and self.adjacency_matrix[u, v] < distances[v] \
                        and self.adjacency_matrix[u, v] != 0:
                    distances[v] = self.adjacency_matrix[u, v]
        return mst_weight + min_dist_to_initial_city

    def print_cities(self):
        for city in self.cities:
            print(city)
        print('\n')


class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.entry_count = 0

    def empty(self) -> bool:
        return not self.elements

    def put(self, node: Node, priority: float):
        self.entry_count += 1
        heapq.heappush(self.elements, [priority, self.entry_count, node])

    def get(self) -> Node:
        return heapq.heappop(self.elements)[2]


def a_star_search(problem: Problem):
    initial_node = Node(state=problem.initial_state, parent=None, path_cost=0)
    frontier = PriorityQueue()
    reached: Dict[State, Node] = {}
    frontier.put(node=initial_node, priority=0)
    reached[initial_node.state] = initial_node

    while not frontier.empty():
        node = frontier.get()
        if problem.isGoal(node.state):
            return node
        h = problem.heuristic(node=node)
        for child in problem.expand(node=node):
            s = child.state
            if s not in reached or child.path_cost < reached.get(s).path_cost:
                reached[s] = child
                g = child.path_cost
                f = g + h
                frontier.put(node=child, priority=f)
    return None


def show_result(num_cities, map_size, cities: List[Tuple[int, int]]):
    for i in range(len(cities)):
        cities[i] = tuple(cities[i])
    num_c = num_cities[0]
    map_s = map_size[0]

    problem = Problem(map_size=map_s, num_cities=num_c, cities=cities)

    print("List of cities:")
    problem.print_cities()
    start_time = time.time()
    solution: Node = a_star_search(problem=problem)
    end_time = time.time()
    print("Initial city:", problem.initial_state.current_city, '\n')
    print("Solution:\n", ' -Cost:', solution.path_cost, '\n', ' -Path:', end=' ')
    n = solution
    for _ in range(len(problem.cities) + 1):
        print(n.state.current_city, end=' ')
        n = n.parent
    print("\n\nExecution time: ", end_time - start_time, '\n')

    X = []
    Y = []
    n = solution
    for _ in range(len(problem.cities) + 1):
        X.append(n.state.current_city[0])
        Y.append(n.state.current_city[1])
        n = n.parent
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, map_s + 2, 1))
    ax.set_yticks(np.arange(0, map_s + 2, 1))
    plt.xlim(-1, map_s + 1)
    plt.ylim(-1, map_s + 1)
    plt.grid()
    for i in range(len(X) - 2):
        plt.plot([X[i], X[i + 1]], [Y[i], Y[i + 1]], marker='o')
    plt.plot([X[len(X) - 2], X[len(X) - 1]], [Y[len(X) - 2], Y[len(X) - 1]], marker='o', linestyle='--')
    plt.show()


def pair(arg):
    return [int(x) for x in arg.split(',')]


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--num_cities", nargs="*", type=int, default=0)
    CLI.add_argument("--map_size", nargs="*", type=int, default=0)
    CLI.add_argument('--cities', type=pair, nargs='+', default=[])
    args = CLI.parse_args()

    num_cities = args.num_cities
    map_size = args.map_size
    cities = args.cities
    show_result(num_cities=num_cities, map_size=map_size, cities=cities)
