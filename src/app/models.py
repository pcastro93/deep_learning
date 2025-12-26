import copy
import random
from typing import Callable, Generator, Self, override

from .loggers import get_logger

logger = get_logger(__name__)


class Matrix:
    def __init__(self, **kwargs):
        if kwargs.get("rows") and kwargs.get("cols"):
            self.rows: int = kwargs.get("rows")
            self.cols: int = kwargs.get("cols")
            self.matrix = [[0.0 for _ in range(self.cols)] for _ in range(self.rows)]
        if kwargs.get("matrix"):
            self.rows = len(kwargs.get("matrix"))
            self.cols = len(kwargs.get("matrix")[0])
            self.matrix: list[list[float]] = kwargs.get("matrix")

    @override
    def __str__(self):
        rows: list[str] = []
        for r in self.matrix:
            rows.append(" ".join(map(str, r)))
        return "\n".join(rows)

    def __mul__(self, other: Self | int | float) -> Self:
        if isinstance(other, self.__class__):
            # Check if can multiply
            # Self is on the left
            # It can be multiplied if self is nxm and other is mxp and the result is nxp
            if self.cols != other.rows:
                raise Exception(
                    "Can't multiply matrices: left=(rows={lr}, cols={lc}), right=(rows={rr}, cols={rc})".format(
                        lr=self.rows, lc=self.cols, rr=other.rows, rc=other.cols
                    )
                )
            res = Matrix(rows=self.rows, cols=other.cols)
            # Multiply using row view
            i = 0
            for r in self.matrix:
                for c in range(other.cols):
                    # get column c of other.matrix
                    col_c = [other_row_i[c] for other_row_i in other.matrix]
                    res.matrix[i][c] = sum(r[i] * col_c[i] for i in range(self.cols))
                i += 1
            return res
        elif isinstance(other, (int, float)):
            res = Matrix(rows=self.rows, cols=self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    res.matrix[i][j] = self.matrix[i][j] * other
            return res
        else:
            raise Exception("Can't multiply matrix by " + other.__class__.__name__)

    def __add__(self, other: Self | int | float):
        # Self is on the left
        if isinstance(other, self.__class__):
            if self.cols != other.cols:
                raise Exception("Can't add matrices with different number of cols")
            # To be valid it should be "broadcast" or both matrices of the same size
            # (the opposite is: both differ in the number of rows and none of them has 1 row)
            if self.rows != other.rows and self.rows != 1 and other.rows != 1:
                raise Exception("Can't broadcast nor add matrices")
            # Matrices addition
            if self.rows == other.rows:
                res = Matrix(rows=self.rows, cols=self.cols)
                for i in range(self.rows):
                    for j in range(self.cols):
                        res.matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
                return res
            else:
                # Vector + Matrix or Matrix + Vector
                vec = self if self.rows == 1 else other
                mat = self if self.rows != 1 else other
                res = Matrix(rows=mat.rows, cols=mat.cols)
                for i in range(mat.rows):
                    for j in range(mat.cols):
                        res.matrix[i][j] = mat.matrix[i][j] + vec.matrix[0][j]
                return res
        elif isinstance(other, (int, float)):
            res = Matrix(rows=self.rows, cols=self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    res.matrix[i][j] = self.matrix[i][j] + other
            return res
        else:
            raise Exception(
                "Can't add matrix by non-integer" + other.__class__.__name__
            )

    def transpose(self):
        res = Matrix(rows=self.cols, cols=self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                res.matrix[j][i] = self.matrix[i][j]
        return res

    def populate(self, fn: Callable):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = fn()

    def reduce(self) -> Self:
        """Reduce matrix to a single row"""
        res = Matrix(rows=1, cols=self.cols)
        for j in range(self.cols):
            res.matrix[0][j] = sum(self.matrix[i][j] for i in range(self.rows))
        return res

    def hadamard_product(self, other: Self) -> Self:
        if self.rows != other.rows or self.cols != other.cols:
            raise Exception("Matrices must have the same dimensions")
        res = Matrix(rows=self.rows, cols=self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                res.matrix[i][j] = self.matrix[i][j] * other.matrix[i][j]
        return res

    def flatten(self) -> list[float]:
        return [self.matrix[i][j] for i in range(self.rows) for j in range(self.cols)]

    def to_list(self) -> list[list]:
        return [[self.matrix[i][j] for j in range(self.cols)] for i in range(self.rows)]

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols


def get_batches(
    data_x: list[list], data_y: list[list], batch_size: int = 32
) -> Generator[tuple[list[list], list[list]]]:
    """Generate batches of the given size.

    Args:
        data_x: list of input data
        data_y: list of output data
        batch_size: size of each batch

    Returns:
        Generator of batches of the given size
    """
    indexes = list(range(len(data_x)))
    random.shuffle(indexes)
    for i in range(0, len(indexes), batch_size):
        current_indexes = indexes[i : min(i + batch_size, len(indexes))]
        current_batch_x = [data_x[j] for j in current_indexes]
        current_batch_y = [data_y[j] for j in current_indexes]
        yield current_batch_x, current_batch_y


class LinearUnit:
    def __init__(self, data_x: list[list], data_y: list[list]):
        self.data_x = copy.deepcopy(data_x)
        self.data_y = copy.deepcopy(data_y)
        self.w_current: Matrix | None = None
        self.b_current: Matrix | None = None
        self.epoch_loss_scores: list | None = None
        self.batch_loss_scores: list | None = None

    def train(self, learning_rate: float, epochs: int, batch_size: int):
        # ========================================
        # n: features (inputs)
        # m: batch size
        # k: outputs of the linear unit
        # ========================================
        # x: batch input (multiple inputs (each with n features)). matrix of mxn
        # w: weights. matrix of nxk dimensions
        # b: matrix of 1xk dimensions
        # y: matrix of mxk dimensions

        # Initialize weights and biases
        self.w_current = Matrix(rows=len(self.data_x[0]), cols=len(self.data_y[0]))
        self.w_current.populate(lambda: random.normalvariate(0, 1))
        self.b_current = Matrix(rows=1, cols=len(self.data_y[0]))
        self.b_current.populate(lambda: random.normalvariate(0, 1))

        self.epoch_loss_scores = []
        self.batch_loss_scores = []

        for epoch_i in range(epochs):
            logger.debug("Epoch: %s/%s", epoch_i + 1, epochs)
            epoch_loss = 0.0
            current_batch_size = 0
            for current_batch in get_batches(self.data_x, self.data_y, batch_size):
                current_batch_size += 1
                current_batch_x = Matrix(matrix=current_batch[0])
                current_batch_y = Matrix(matrix=current_batch[1])
                y_pred = current_batch_x * self.w_current + self.b_current
                logger.debug("y_pred=\n[%s]\n", y_pred)

                # Calculate the loss using MSE
                # f(x) = y = wx + b
                # loss(y) = (1/x.rows) * (y_pred - y_noised)**2 # MSE
                # y_pred = cw*cx + cb
                # y_noised = w * x + b
                # g(f(x)) = loss(y)
                #
                # Gradient of W (d_w):
                # d_loss/d_w = d_loss/d_y * d_y/d_w # chain rule
                # d_loss/d_y = 2 * (1/x.rows) * (y_pred - y_noised)
                # d_y/d_w = x
                # So: d_loss/d_w = (2/x.rows) * (y_pred - y_noised) * x
                # error = (y_pred - y_noised)
                # Then:
                # w_gradient = (2/x.rows) * error * x
                #
                # Gradient of B (d_b)
                # d_loss/d_b = d_loss/d_y * dy/db
                # dy/db = 1
                # d_loss/d_b = 2 * (1/x.rows) * (y_pred - y_noised)
                # Because b is added x.rows times (one per input)
                # d_b = sum(d_loss/_db)
                #               = sum_of_the_rows_of(2 * (1/x.rows) * (y_pred - y_noised))
                #               = (2/x.rows) * sum(error)
                error = y_pred + (current_batch_y * -1)
                # because d_loss/dy is in both gradients we can reuse it
                dl_dy = error * (2 / (current_batch_x.rows * current_batch_y.cols))
                w_gradient = current_batch_x.transpose() * dl_dy
                b_gradient = dl_dy.reduce()
                # logger.info("error=\n[%s]\n", error)
                # logger.info("w_gradient=\n[%s]\n", w_gradient)
                # logger.info("b_gradient=\n[%s]\n", b_gradient)
                error_sq = error.hadamard_product(error)

                batch_loss = sum(error_sq.reduce().matrix[0]) / (
                    error_sq.rows * error_sq.cols
                )
                epoch_loss += batch_loss
                self.batch_loss_scores.append(batch_loss)

                self.w_current = self.w_current + (w_gradient * learning_rate) * -1
                self.b_current = self.b_current + (b_gradient * learning_rate) * -1
            self.epoch_loss_scores.append(epoch_loss / current_batch_size)

    def infer(self, x: list[list]):
        x_mat = Matrix(matrix=x)
        return x_mat * self.w_current + self.b_current
