import numpy as np
import pytest

from app.models import Matrix


@pytest.fixture
def a():
    return Matrix(matrix=[[1, 2], [3, 4]])


@pytest.fixture
def b():
    return Matrix(matrix=[[5, 6], [7, 8]])


@pytest.fixture
def a2():
    return np.array([[1, 2], [3, 4]])


@pytest.fixture
def b2():
    return np.array([[5, 6], [7, 8]])


def test_2d_matrix_instatiation():
    m = Matrix(rows=2, cols=3)
    n = Matrix(matrix=[[1, 2], [3, 4]])
    assert m.rows == 2
    assert m.cols == 3
    assert m.matrix[0][1] == 0
    assert n.rows == 2
    assert n.cols == 2
    assert n.matrix[0][1] == 2


def test_2d_matrix_multiplication(a: Matrix, b: Matrix, a2: np.ndarray, b2: np.ndarray):
    c = a * b
    d = a * 3
    c2 = a2 @ b2
    d2 = a2 * 3
    assert np.array_equal(c.matrix, c2)
    assert np.array_equal(d.matrix, d2)


def test_2d_matrix_transpose(a: Matrix, a2: np.ndarray):
    at = a.transpose()
    at2 = a2.T
    assert np.array_equal(at.matrix, at2)


def test_2d_matrix_addition(a: Matrix, b: Matrix):
    v = Matrix(matrix=[[1, 2]])
    c = a + b
    d = a + 3
    e = a + v
    f = v + a
    a2 = np.array([[1, 2], [3, 4]])
    b2 = np.array([[5, 6], [7, 8]])
    v2 = np.array([1, 2])
    c2 = a2 + b2
    d2 = a2 + 3
    e2 = a2 + v2
    f2 = v2 + a2
    assert np.array_equal(c.matrix, c2)
    assert np.array_equal(d.matrix, d2)
    assert np.array_equal(e.matrix, e2)
    assert np.array_equal(f.matrix, f2)


def test_2d_matrix_reduction(a: Matrix, a2: np.ndarray):
    assert np.array_equal(a.reduce().matrix[0], np.sum(a2, axis=0))


def test_hadamard_product(a: Matrix, b: Matrix, a2: np.ndarray, b2: np.ndarray):
    c = a.hadamard_product(b)
    c2 = a2 * b2
    assert np.array_equal(c.matrix, c2)


def test_2d_matrix_flatten(a: Matrix, a2: np.ndarray):
    assert np.array_equal(a.flatten(), a2.flatten())
