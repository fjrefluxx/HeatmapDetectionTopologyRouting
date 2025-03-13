import math

import numpy as np


class DivideConquerDirectedRouting:
    """
    Dynamic programming approach for a shortest path problem with negative edge weights without negative cycles.

    See, for example:
        E. N. Mortensen, B. Morse, W. Barrett, and J. Udupa, 'Adaptive boundary
        detection using live-wire two-dimensional dynamic programming',
        Computers in Cardiology, pp. 635–635, 1992.

    Original code written in Java by Benjamin Becker, TU Darmstadt, Germany.
    Adapted and ported to Python.
    """

    @staticmethod
    def path(matrix: np.array,
             start: (int, int),
             end: (int, int)) -> list[(int, int)]:
        """
        Given a matrix, find the best path from start to end within.

        :start: indices starting cell
        :end: indices destination cell
        :return: summed costs; list of traversal cell indices
        """
        assert not np.array_equal(start, end)
        cost, route = DivideConquerDirectedRouting.__shortest_path_dynamic_programming(np.array(matrix), start, end)
        return route

    @staticmethod
    def __shortest_path_dynamic_programming(matrix: np.array,
                                            start: (int, int),
                                            end: (int, int)) -> (float, list[(int, int)]):
        """
        Dynamic programming approach: find the best route in a 2D-grid from start to end.
        Best is defined as the shortest path with the highest summarized edge weights.

        :start: indices starting cell
        :end: indices destination cell
        :return: summed costs; list of traversal cell indices
        """
        assert not np.array_equal(start, end)

        x1, y1 = start
        x2, y2 = end

        dx = x2 - x1
        dy = y2 - y1

        # rotate and mirror (if necessary)
        mirrored = False
        rotations_counter = 0

        if math.copysign(1, dx) == math.copysign(1, dy):
            if abs(dx) < abs(dy):
                matrix = DivideConquerDirectedRouting.__mirror(matrix)
                x1 = matrix.shape[0] - 1 - x1
                x2 = matrix.shape[0] - 1 - x2
                dx = x2 - x1
                mirrored = True
        else:
            if abs(dx) > abs(dy):
                matrix = DivideConquerDirectedRouting.__mirror(matrix)
                x1 = matrix.shape[0] - 1 - x1
                x2 = matrix.shape[0] - 1 - x2
                dx = x2 - x1
                mirrored = True

        # in case of a diagonal/straight line, exactly one of the two tested paths must be mirrored
        if (x1 > x2) or (x1 == x2 and y1 < y2):
            matrix = DivideConquerDirectedRouting.__mirror(matrix)
            x1 = matrix.shape[0] - 1 - x1
            x2 = matrix.shape[0] - 1 - x2
            dx = x2 - x1
            mirrored = not mirrored

        while dx <= 0 or dy < 0:
            rotations_counter += 1
            matrix = DivideConquerDirectedRouting.__rotate90(matrix)
            index_x = x1
            x1 = y1
            y1 = matrix.shape[1] - 1 - index_x
            index_x = x2
            x2 = y2
            y2 = matrix.shape[1] - 1 - index_x
            dx = x2 - x1
            dy = y2 - y1

        ### restrict directions
        matrix[x2][y2] = 0  # to make the results of two runs with flipped start and end comparable
        matrix[x1][y1] = 0  # to make the results of two runs with flipped start and end comparable

        # vertical mask with inf
        if x1 > 0:
            x = x1 - 1
            for y in range(y1, matrix.shape[1]):
                matrix[x][y] = float("inf")

        # diagonal mask with inf
        if y1 > 0:
            x = x1 - 1
            for y in range(y1, -1, -1):
                if x >= matrix.shape[0]:
                    break
                if x >= 0:
                    matrix[x][y] = float("inf")
                    if y > 0:
                        matrix[x][y - 1] = float("inf")
                x += 1

        # step until goal is reached
        directions = np.zeros(matrix.shape, dtype=int)
        reached = False
        index_x = x1


        # length of shortest path
        result: float = 0

        while not reached:
            x = index_x
            # vertical
            for y in range(y1, matrix.shape[1]):
                if x == x1 and y == y1:
                    matrix[x][y] = 0
                else:
                    if directions[x][y] == 0:
                        cost, direction = DivideConquerDirectedRouting.__best_direction(matrix, x, y)
                        matrix[x][y] = -matrix[x][y] + cost
                        directions[x][y] = direction

                if x == x2 and y == y2:
                    reached = True
                    result = matrix[x][y]

            # diagonal
            for y in range(y1, -1, -1):
                if x >= matrix.shape[0]:
                    break

                if y != y1 and directions[x][y] == 0:
                    cost, direction = DivideConquerDirectedRouting.__best_direction(matrix, x, y)
                    matrix[x][y] = -matrix[x][y] + cost
                    directions[x][y] = direction

                if x == x2 and y == y2:
                    reached = True
                    result = matrix[x][y]

                x += 1
            index_x += 1
        # end while

        steps: list = list()
        stepx = x2
        stepy = y2

        while x1 != stepx or y1 != stepy:
            d = directions[stepx][stepy]
            steps.append(d)
            match d:
                case 1:
                    stepy -= 1
                    continue
                case 2:
                    stepx -= 1
                    stepy -= 1
                    continue
                case 3:
                    stepx -= 1
                    continue
                case 4:
                    stepx -= 1
                    stepy += 1
                    continue
                case _:
                    raise KeyError()

        # rotate and mirror back
        while rotations_counter % 4 > 0:
            rotations_counter += 1
            for i in range(0, len(steps)):
                if steps[i] < 7:
                    steps[i] = steps[i] + 2
                else:
                    if steps[i] == 8:
                        steps[i] = 2
                    if steps[i] == 7:
                        steps[i] = 1

        if mirrored:
            for i in range(0, len(steps)):
                if steps[i] != 5 and steps[i] != 1:
                    steps[i] = 10 - steps[i]

        # reverse order of steps to put it correctly as start->end
        steps.reverse()

        # move through all step commands and retrieve the matching indices
        indices: list[(int, int)] = list()
        indices.append(start)

        x, y = start
        for s in steps:
            if s == 1:
                y += 1
            if s == 2:
                x += 1
                y += 1
            if s == 3:
                x += 1
            if s == 4:
                x += 1
                y -= 1
            if s == 5:
                y -= 1
            if s == 6:
                x -= 1
                y -= 1
            if s == 7:
                x -= 1
            if s == 8:
                x -= 1
                y += 1

            indices.append((x, y))

        return result, indices

    @staticmethod
    def __mirror(matrix: np.array) -> np.array:
        """
        Mirror the input matrix
        """
        mirrored = np.zeros(matrix.shape)
        for x in range(0, matrix.shape[0]):
            for y in range(0, matrix.shape[1]):
                mirrored[matrix.shape[0] - x - 1][y] = matrix[x][y]
        return mirrored

    @staticmethod
    def __rotate90(matrix: np.array) -> np.array:
        """
        Rotate the input matrix
        """
        rotated = np.zeros((matrix.shape[1], matrix.shape[0]))
        for x in range(0, matrix.shape[0]):
            for y in range(0, matrix.shape[1]):
                rotated[y][matrix.shape[0] - x - 1] = matrix[x][y]
        return rotated

    @staticmethod
    def __best_direction(matrix: np.array, x: int, y: int) -> (float, int):
        """
        :return: cost, direction identifier
        """

        # left
        if y - 1 == -1:
            a1 = float("inf")
        else:
            a1 = matrix[x][y - 1] + 1

        # left, up (diagonal)
        if y - 1 == -1 or x - 1 == -1:
            a2 = float("inf")
        else:
            a2 = matrix[x - 1][y - 1] + math.sqrt(2)

        # up
        if x - 1 == -1:
            a3 = float("inf")
        else:
            a3 = matrix[x - 1][y] + 1

        # up, right (diagonal)
        if x - 1 == -1 or y + 1 == matrix.shape[1]:
            a4 = float("inf")
        else:
            a4 = matrix[x - 1][y + 1] + math.sqrt(2)

        directions: list = [a1, a2, a3, a4]

        return np.min(directions), np.argmin(directions) + 1  # +1 = a mathematician's adaptation

