import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from simpy import Environment

from GaussianKernel import GaussianKernel
from LinkedHeatmapCell import LinkedHeatmapCell


class Heatmap:
    MINUTE = 60.0
    HOUR = MINUTE * 60.0

    degradationInterval: float = MINUTE

    def __init__(self, p0: (float, float), p1: (float, float),
                 cell_size: float, kernel_one_sigma_distance: float):
        """
        Heatmap is defined by the two points p0 (top-left) and p1 (bottom-right).

         X----------
        |          |
        |          |
        |          |
         ----------X

        :param p0: "top left" point of the heatmap
        :param p1: "bottom right" point of the heatmap
        :param cell_size: Size of the quadratic cells in the heatmap. Must divide the width and height of the heatmap without remainder.
        :param kernel_one_sigma_distance: One-sigma distance of the kernel. Set to expected cluster size.
        """
        assert p0[0] < p1[0] and p0[1] < p1[1]

        self.p0 = p0
        self.p1 = p1
        self.cell_size = cell_size

        self.heatmap: NDArray[LinkedHeatmapCell] = self.__create_heatmap()
        self.kernel = GaussianKernel.create_kernel(kernel_one_sigma_distance, cell_size)

        self.running_degradation = False

        self.env = None
        self.measurement_impact = None
        self.measurement_upperbound = None

    def __create_heatmap(self) -> NDArray[LinkedHeatmapCell]:  # list[list[LinkedHeatmapCell]]:
        '''
        Populate the heatmap with linked cells.
        '''
        w = self.p1[0] - self.p0[0]
        h = self.p1[1] - self.p0[1]

        if w % self.cell_size != 0 or h % self.cell_size != 0:
            raise AssertionError("Width and height of the heatmap do not match the given cell size")

        num_cells_x: int = int(w / self.cell_size)
        num_cells_y: int = int(h / self.cell_size)

        # populate heatmap
        heatmap = np.empty(shape=(num_cells_x, num_cells_y), dtype=LinkedHeatmapCell)
        for y in range(0, num_cells_y):
            index_y = self.p0[1] + y * self.cell_size
            for x in range(0, num_cells_x):
                index_x = self.p0[0] + x * self.cell_size
                cell = LinkedHeatmapCell((index_x, index_y), (index_x + self.cell_size, index_y + self.cell_size))

                # link with left neighbor
                if x > 0:
                    cell.left_neighbor = heatmap[x - 1][y]
                    heatmap[x - 1][y].right_neighbor = cell

                # link with top neighbor
                if y > 0:
                    cell.top_neighbor = heatmap[x][y - 1]
                    heatmap[x][y - 1].bottom_neighbor = cell

                heatmap[x][y] = cell

        return heatmap

    def contains(self, position: (float, float)) -> bool:
        """
        Check if a position is inside the heatmap (including borders)
        """
        if (self.p0[0] <= position[0] <= self.p1[0]
                and self.p0[1] <= position[1] <= self.p1[1]):
            return True
        return False

    def cell_containing_position(self, position: (float, float)) -> LinkedHeatmapCell | None:
        """
        Get the cell corresponding to a given position.
        """
        if not self.contains(position):
            return None

        for i in range(0, self.heatmap.shape[0]):
            for j in range(0, self.heatmap.shape[1]):
                if self.heatmap[i][j].contains(position):
                    return self.heatmap[i][j]
        return None

    def cell_coordinates(self, cell: LinkedHeatmapCell) -> (int, int):
        """
        Search the heatmap for a specific cell and return the indices
        """
        assert cell in self.heatmap
        rows, cols = np.where(self.heatmap == cell)
        return rows[0], cols[0]

    def start_degradation(self, env: Environment,
                          degradation_interval: float,
                          update_interval: float,
                          measurement_impact: float,
                          slow_count_retardation: float = 4.0,
                          upper_bound: float = 100.0):
        """
        Start counter degrading process.

        :param degradation_interval:
        :param env:
        :param update_interval:
        :param measurement_impact:
        :param slow_count_retardation:
        :param upper_bound:
        :return:
        """
        assert self.running_degradation is False
        self.env = env

        self.measurement_impact = measurement_impact
        self.measurement_upperbound = upper_bound

        for x in range(0, len(self.heatmap)):
            for y in range(0, len(self.heatmap[0])):
                self.heatmap[x][y].increment_impact = self.measurement_impact
                self.heatmap[x][y].counter_upperbound = self.measurement_upperbound
                self.heatmap[x][y].halftime_slow = update_interval * slow_count_retardation
                self.heatmap[x][y].halftime_fast = update_interval

        self.process = env.process(self.__cell_counter_degradation(degradation_interval))

    def __cell_counter_degradation(self, degradation_interval: float):
        """
        
        :param degradation_interval:
        :return:
        """
        while True:
            yield self.env.timeout(degradation_interval)

            for x in range(0, len(self.heatmap)):
                for y in range(0, len(self.heatmap[0])):
                    self.heatmap[x][y].degrade_counter(degradation_interval)

    def input_measurement(self, position: (float, float)):
        """
        Put a position measurement into the heatmap (incrementing the counter
        of the respective heatmap cell)
        """
        assert self.contains(position)

        x = math.floor(position[0] / self.cell_size)
        y = math.floor(position[1] / self.cell_size)

        assert self.heatmap[x][y].contains(position)
        self.heatmap[x][y].increment_counter(position)

    @staticmethod
    def matrix_max_below_threshold(matrix: NDArray[np.float64], threshold: float) -> (None, None | float, Iterable):
        """
        In the given matrix, search for the indices of the maximum value (below a given threshold).
        Does return multiple indices, if the same maximum value occurs multiple times.

        :param matrix:
        :param threshold: Search maximum value below this threshold.
        :return: The maximum value and tuples of indices of its occurance.
        """
        max_v = float('-inf')

        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                if max_v < matrix[i][j] < threshold:
                    max_v = matrix[i][j]

        if max_v == float('-inf'):
            return None, None

        rows, cols = np.where(matrix >= max_v)
        indices = zip(rows, cols)

        return max_v, indices

    def heatmap_matrix(self, type_counter: str) -> NDArray[np.float64]:
        """
        Get the fast or slow counter matrix.
        :param type_counter: 'fast' or 'slow'
        """
        match type_counter:
            case 'fast':
                matrix = np.zeros(self.heatmap.shape)

                for i in range(0, matrix.shape[0]):
                    for j in range(0, matrix.shape[1]):
                        matrix[i][j] = self.heatmap[i][j].fastDegradationCounter
                return matrix

            case 'slow':
                matrix = np.zeros(self.heatmap.shape)

                for i in range(0, matrix.shape[0]):
                    for j in range(0, matrix.shape[1]):
                        matrix[i][j] = self.heatmap[i][j].slowDegradationCounter
                return matrix

            case _:
                raise Exception("Not allowed matrix counter type")

    def smoothed_hotspot_matrix(self,
                                hotspot_count_maximum: int = 10,
                                highpass_filter_threshold: float = 10) -> NDArray[np.float64]:
        """
        Retrieve the hotspot matrix for the fast degrading counter and smooth it using
        a Gaussian kernel convolution.
        Cells within the reach of a hotspot (i.e., adjacent or within reach of the convolution kernel)
        are masked, to later exclude from routing.

        :param hotspot_count_maximum: Maximum number of hotspots to be served. Hotspots are sorted based on
            their intensity, thus, retrieving the first n=highpass_filter_threshold hotspots.
        :param highpass_filter_threshold: Lowest counter value to count as a hotspot.

        """
        hotspot_count: int = 0

        smoothed = GaussianKernel.convolution(self.heatmap_matrix('fast'), self.kernel)
        max_value, max_indices = Heatmap.matrix_max_below_threshold(smoothed, float('inf'))

        if max_value is None:
            raise Exception("wait!")

        for i in range(0, smoothed.shape[0]):
            for j in range(0, smoothed.shape[1]):
                if smoothed[i][j] < highpass_filter_threshold:
                    smoothed[i][j] = -1

        # mask adjacent cells from hotspot (defined by kernel size)
        omit_size = self.kernel.shape[0]
        while True:
            for index in max_indices:

                for i in range(0, omit_size):
                    x_i = index[0] - int(math.floor(omit_size / 2)) + i
                    if x_i < 0 or x_i >= smoothed.shape[0]:
                        continue

                    for j in range(0, omit_size):
                        y_i = index[1] - int(math.floor(omit_size / 2)) + j
                        if y_i < 0 or y_i >= smoothed.shape[1]:
                            continue

                        if x_i == index[0] and y_i == index[1]:
                            continue

                        smoothed[x_i][y_i] = -2

            hotspot_count += 1
            new_max, max_indices = Heatmap.matrix_max_below_threshold(smoothed, max_value)

            if new_max is None or new_max <= 0 or new_max < highpass_filter_threshold or hotspot_count >= hotspot_count_maximum:
                break

        return smoothed
