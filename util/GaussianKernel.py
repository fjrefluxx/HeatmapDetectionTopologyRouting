import math

import numpy as np
from numpy.typing import NDArray


class GaussianKernel:

    @staticmethod
    def create_kernel(one_sigma_distance: float, cell_size: float) -> NDArray[np.float64]:
        """
        Create a normalized, discrete gaussian kernel matrix based on the given cell size and the 1-sigma-distance.
        The kernel covers at least the 2-sigma neighborhood.
        """

        # kernel size: must cover at least the 2*sigma range
        kernel_size: int = int(math.ceil((2 * one_sigma_distance) / cell_size) * 2)

        # kernel must be odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # sigma (std. deviation) calculated based on the 1-sigma-distance and the cell size
        sigma = one_sigma_distance / cell_size

        '''
        calculation based on clemisch's answers in
        https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
        '''
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        kernel = kernel / np.sum(kernel)

        return kernel

    @staticmethod
    def convolution(matrix: NDArray[np.float64], kernel: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Matrix convolution with a given kernel. Kernel must be quadratic, its size must be odd.
        :return: convolved matrix
        """
        assert kernel.shape[0] == kernel.shape[1]  # kernel must be quadratic
        assert kernel.shape[0] % 2 != 0  # kernel edge length must be odd

        convolved_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))

        for x in range(0, matrix.shape[0]):
            for y in range(0, matrix.shape[1]):
                convolved_matrix[x][y] = GaussianKernel.convolve_entry(matrix, x, y, kernel)

        return convolved_matrix

    @staticmethod
    def convolve_entry(matrix: NDArray[np.float64], x: int, y: int, kernel: NDArray[np.float64]) -> float:
        """
        Convolve a matrix entry with the given kernel. Positions outside the matrix are evaluated as zeros.

        :param matrix: Matrix of the convolution
        :param x: Entry x-coordinate in matrix
        :param y: Entry y-coordinate in matrix
        :param kernel: Kernel for the convolution, applied on the entry.
        :return: Resulting entry value from the convolution with the kernel.
        """
        assert kernel.shape[0] == kernel.shape[1]  # kernel must be quadratic
        assert kernel.shape[0] % 2 != 0  # kernel edge length must be odd

        sum: float = 0
        offset: int = math.floor(kernel.shape[0] / 2)

        if matrix[x][y] != 0:
            print()

        for i in range(0, kernel.shape[0]):
            x_index: int = x - offset + i
            if x_index < 0 or x_index >= matrix.shape[0]:
                continue

            for j in range(0, kernel.shape[1]):
                y_index: int = y - offset + j
                if y_index < 0 or y_index >= matrix.shape[1]:
                    continue

                sum += matrix[x_index][y_index] * kernel[i][j]

        return sum
