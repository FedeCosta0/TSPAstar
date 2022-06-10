import sys
import heapq
import time
import numpy as np
from math import sqrt
from random import randint
from typing import Tuple, Set, Dict, List, FrozenSet
import matplotlib.pyplot as plt


INF = 99999
NUM_CITIES = 10
MAP_SIZE = 10
NUM_RUNS = 1


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
    def __init__(self, map_size: int, num_cities: int):
        if num_cities > map_size ** 2:
            raise ValueError("Too many cities for the map size")

        self.map_size = map_size
        self.num_cities = num_cities

        #  initialize random cities
        self.cities_indexes: Dict[Tuple[int, int], int] = dict()
        self.cities: FrozenSet[Tuple[int, int]] = self.init_cities()

        self.adjacency_matrix: np.ndarray = self.init_adjacency_matrix(cities=self.cities)

        #  the city for the initial state is selected from the set of cities with 'min' to not remove it with 'pop'
        self.initial_state = State(visited_cities=frozenset({min(self.cities)}), current_city=min(self.cities))

        # the goal state is reached when we have visited all cities and the current city is the initial city
        self.goal_state = State(visited_cities=self.cities, current_city=self.initial_state.current_city)

    def init_cities(self):
        tmp: Set[Tuple[int, int]] = set()
        city_index = 0
        while len(tmp) < self.num_cities:
            x = randint(0, self.map_size)
            y = randint(0, self.map_size)
            #  check if a city in position (x,y) already exists
            if not (x, y) in tmp:
                tmp.add((x, y))
                self.cities_indexes[(x, y)] = city_index
                city_index += 1
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
    @staticmethod
    def min_distances(distances, mst_set, num_vertices) -> [int, float]:
        min_distance = INF
        min_index = -1
        for v in range(num_vertices):
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
        mst_set = [False for _ in range(len(unvisited_cities))]

        #  initialize distances of nodes from current mst
        distances = [INF for _ in range(len(unvisited_cities))]

        #  select node at index 0 as the first node of mst
        distances[0] = 0

        mst_weight = 0
        for _ in range(len(unvisited_cities)):
            #  u is the node with minimum distance from current mst
            u, dist = self.min_distances(distances=distances, mst_set=mst_set, num_vertices=len(unvisited_cities))

            #  insert the node in mst and update mst weight
            mst_set[u] = True
            mst_weight += dist

            # updates the distances from the current mst of nodes that are not in mst yet
            # and that can be reached by u at a cost less than the cost in memory
            for i, v in enumerate(unvisited_cities_indexes):
                if mst_set[i] is False and self.adjacency_matrix[u, v] < distances[i]\
                        and self.adjacency_matrix[u, v] != 0:
                    distances[i] = self.adjacency_matrix[u, v]
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


#  test and show result
def test():
    total_execution_time = 0
    for _ in range(NUM_RUNS):
        problem = Problem(map_size=MAP_SIZE, num_cities=NUM_CITIES)
        start_time = time.time()
        a_star_search(problem=problem)
        end_time = time.time()
        total_execution_time += end_time - start_time
    average_execution_time = total_execution_time / NUM_RUNS
    average_execution_time = round(average_execution_time, 3)
    print("Average execution time with " + str(NUM_CITIES) + " cities and " + str(MAP_SIZE) + " map size: " + str(
        average_execution_time) + "s")
    print("Executed " + str(NUM_RUNS) + " times")


def show_result(num_cities: int, map_size: int):
    problem = Problem(map_size=map_size, num_cities=num_cities)
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
    ax.set_xticks(np.arange(0, map_size + 2, 1))
    ax.set_yticks(np.arange(0, map_size + 2, 1))
    plt.xlim(-1, map_size + 1)
    plt.ylim(-1, map_size + 1)
    plt.grid()
    for i in range(len(X) - 2):
        plt.plot([X[i], X[i + 1]], [Y[i], Y[i + 1]], marker='o')
    plt.plot([X[len(X) - 2], X[len(X) - 1]], [Y[len(X) - 2], Y[len(X) - 1]], marker='o', linestyle='--')
    plt.show()


if __name__ == "__main__":
    num_cities = int(sys.argv[1])
    map_size = int(sys.argv[2])
    show_result(num_cities=num_cities, map_size=map_size)
