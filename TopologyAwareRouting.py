import itertools

import networkx as nx
from networkx.algorithms.approximation.traveling_salesman import christofides

from util.Heatmap import Heatmap
from util.LinkedHeatmapCell import LinkedHeatmapCell
from routing.DivideConquerDirectedRouting import DivideConquerDirectedRouting


class TopologyAwareRouting:

    @staticmethod
    def waypoints(heatmap: Heatmap, start_end_location) -> list[(float, float, bool)]:
        """
        Calculate a path of waypoints using the given heatmap and an additional start and end location for
        the TSP tour.

        :param heatmap:
        :param start_end_location:
        :return: list of waypoints --- (x,y) coordinates and a boolean indicating a hotspot center location.
        """
        smoothed_matrix = heatmap.smoothed_hotspot_matrix()

        hotspots: list[LinkedHeatmapCell] = list()
        hotspot_surroundings: list[LinkedHeatmapCell] = list()

        maximum: float = float("-inf")

        for i in range(0, smoothed_matrix.shape[0]):
            for j in range(0, smoothed_matrix.shape[1]):
                if smoothed_matrix[i][j] == -2:
                    hotspot_surroundings.append(heatmap.heatmap[i][j])
                elif smoothed_matrix[i][j] > 0:
                    hotspots.append(heatmap.heatmap[i][j])

                    if smoothed_matrix[i][i] > maximum:
                        maximum = smoothed_matrix[i][j]

        if len(hotspots) < 2:
            print("not enough hotspots found for routing")
            return

        # get closest hotspot
        starting_cell = hotspots[0]
        for i in range(1, len(hotspots)):
            if hotspots[i].distance_centroid(start_end_location) < starting_cell.distance_centroid(start_end_location):
                starting_cell = hotspots[i]

        tour = TopologyAwareRouting.christofides_tour(hotspots)
        # first and last point of the tour should be the same (i.e., starting cell)
        assert tour[0] == tour[-1]

        if tour[0] != starting_cell:
            tour.pop()  # remove last (break circle)
            while tour[0] != starting_cell:
                tour.append(tour.pop(0))  # put first element to the end
            tour.append(tour[0])  # close the circle again

        path: (float, float, bool) = list()
        source_cell = tour.pop(0)
        while len(tour) > 0:
            destination_cell = tour.pop(0)
            # cell to cell path
            path.extend(TopologyAwareRouting.cell_to_cell(heatmap, source_cell, destination_cell))

            source_cell = destination_cell

        if len(path) == 0:
            raise Exception

        path.insert(0, (start_end_location[0], start_end_location[1], False))
        path.append((start_end_location[0], start_end_location[1], False))
        return path

    @staticmethod
    def cell_to_cell(heatmap: Heatmap, source, destination) -> list[(float, float, bool)]:
        """
        Best route over most long-term populated heatmap cells from a
        source to a destination cell.
        """
        route = DivideConquerDirectedRouting.path(heatmap.heatmap_matrix('slow'),
                                                  heatmap.cell_coordinates(source),
                                                  heatmap.cell_coordinates(destination))
        path = list()
        for i in range(0, len(route)):
            (x, y) = route[i]
            path.append((*heatmap.heatmap[x][y].center(),
                         i == 0 or i == len(route) - 1))
        return path

    @staticmethod
    def christofides_tour(hotspots: list[LinkedHeatmapCell]) -> list[LinkedHeatmapCell]:
        """
        Approximate TSP tour using the Christofides algorithm from networkX.
        """
        G = nx.Graph()
        combinations = itertools.combinations(hotspots, 2)

        # populate graph
        for (a, b) in combinations:
            d = a.distance_centroid(b.center())
            G.add_edge(a, b, weight=d)

        return christofides(G)
