from __future__ import annotations

import math


class LinkedHeatmapCell:
    """
    Cell within a heatmap, linked to potential neighboring cells.
    A cell is defined by two points ('upper left' and 'lower right').
     X-----------
     |          |
     |          |
     |          |
     -----------X
    Each cell holds counters to track measurements in the heatmap.
    """

    def __init__(self, p0: (float, float), p1: (float, float)):
        assert p0[0] < p1[0] and p0[1] < p1[1]

        self.p0 = p0
        self.p1 = p1

        self.left_neighbor: LinkedHeatmapCell | None = None
        self.right_neighbor: LinkedHeatmapCell | None = None
        self.top_neighbor: LinkedHeatmapCell | None = None
        self.bottom_neighbor: LinkedHeatmapCell | None = None

        self.fastDegradationCounter: float = 0
        self.slowDegradationCounter: float = 0
        self.counter_upperbound: float = 100
        self.increment_impact: float = 1.0

        self.halftime_fast: float = 0
        self.halftime_slow: float = 0

        self.activeMeasurement: bool = False

    def center(self) -> (float, float):
        """
        Center of the cell.
        """
        return (self.p0[0] + 0.5 * (self.p1[0] - self.p0[0]),
                self.p0[1] + 0.5 * (self.p1[1] - self.p0[1]))

    def distance_centroid(self, location: (float, float)) -> float:
        """
        Euclidic distance between the cell's center and the given position.
        """
        c1 = self.center()
        return math.sqrt((c1[0] - location[0]) ** 2 + (c1[1] - location[1]) ** 2)

    def contains(self, position: (float, float)) -> bool:
        """
        Check if a position is inside the cell.
        INCLUSIVE of the top and left border.
        EXCLUSIVE of the bottom and right border.
        """
        if (self.p0[0] <= position[0] < self.p1[0]
                and self.p0[1] <= position[1] < self.p1[1]):
            return True
        return False

    def adjacent_cells(self) -> list[LinkedHeatmapCell]:
        """
        List of all adjacent cells, starting from the top neighbor going clockwise

        Indexing:
            7-0-1
            6-X-2
            5-4-3

        :return: List of adjacent cells. Empty, if no neighbors linked.
        """
        neighbors: list = list()

        neighbors.append(self.top_neighbor)
        if self.top_neighbor is not None:
            neighbors.append(self.top_neighbor.right_neighbor)
        neighbors.append(self.right_neighbor)
        if self.right_neighbor is not None:
            neighbors.append(self.right_neighbor.bottom_neighbor)
        neighbors.append(self.bottom_neighbor)
        if self.bottom_neighbor is not None:
            neighbors.append(self.bottom_neighbor.left_neighbor)
        neighbors.append(self.left_neighbor)
        if self.left_neighbor is not None:
            neighbors.append(self.left_neighbor.top_neighbor)

        return neighbors

    def increment_counter(self, position: (float, float)) -> bool:
        """
        Increment the internal counters due to a given position measurement.

        :return: True if position is inside the cell and the counter is incremented. False otherwise.
        """
        if not self.contains(position):
            return False

        if self.slowDegradationCounter < self.counter_upperbound:
            self.slowDegradationCounter += self.increment_impact * (1 - (self.slowDegradationCounter / 100))

        if self.fastDegradationCounter < self.counter_upperbound:
            self.fastDegradationCounter += 1 * self.increment_impact

        self.activeMeasurement = True
        return True

    def degrade_counter(self, interval: float):
        """
        Degrade the internal counters once, using the given interval (i.e., time since the last degradation)
        to scale the size of the counter degradation.
        """
        if self.fastDegradationCounter > 0:
            self.fastDegradationCounter *= math.exp(-interval * (math.log(2) / self.halftime_fast))

            if self.fastDegradationCounter < 0.1:
                self.fastDegradationCounter = 0

        if self.slowDegradationCounter > 0:
            self.slowDegradationCounter *= math.exp(-interval * (math.log(2) / self.halftime_slow))

            if self.slowDegradationCounter < 0.1:
                self.slowDegradationCounter = 0

        self.activeMeasurement = False

    def __str__(self):
        return "HeatmapCell [" + str(self.p0) + "/" + str(self.p1) + "]"
