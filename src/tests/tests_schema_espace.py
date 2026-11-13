import numpy as np
from schemas_espace import derivative_x, derivative_y, divergence, upwind1


def build_field(func, x, y):
    """
    Évalue une fonction analytique f(x, y) en chaque nœud de la grille.

    Parameters
    ----------
    func : callable
        Fonction f(x, y) renvoyant un float.
    x : numpy.ndarray, shape (nx,)
        Coordonnées des nœuds en x.
    y : numpy.ndarray, shape (ny,)
        Coordonnées des nœuds en y.

    Returns
    -------
    field : numpy.ndarray, shape (nx, ny)
        Valeurs de func aux nœuds, field[i, j] = func(x[i], y[j]).
    """
    nx = len(x)
    ny = len(y)
    field = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            field[i, j] = func(x[i], y[j])
    return field


def max_interior_error(numeric, exact):
    """
    Erreur maximale entre deux tableaux, sur les nœuds intérieurs.

    Parameters
    ----------
    numeric : numpy.ndarray, shape (nx, ny)
        Résultat calculé par le schéma.
    exact : numpy.ndarray, shape (nx, ny)
        Valeur analytique.

    Returns
    -------
    err : float
        max |numeric - exact| sur les nœuds 1 ... n-2.
    """
    nx, ny = numeric.shape
    err = 0.0
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            err = max(err, abs(numeric[i, j] - exact[i, j]))
    return err


def observed_order(errors):
    """
    Ordre de convergence observé entre deux maillages dont le pas est divisé par 2.

    Parameters
    ----------
    errors : list of float
        Erreurs pour des maillages successifs (n = 33, 65, 129, ...).

    Returns
    -------
    order : float
        log2(erreur grossière / erreur fine) entre les deux derniers maillages.
    """
    return np.log2(errors[-2] / errors[-1])


def test_derivative_x_order_2():
    """derivative_x doit être d'ordre 2 sur sin(x)·cos(y)."""
    errors = []
    for n in (33, 65, 129):
        x = np.linspace(0.0, 1.0, n)
        y = np.linspace(0.0, 1.0, n)
        h = x[1] - x[0]
        phi = build_field(lambda a, b: np.sin(a) * np.cos(b), x, y)
        exact = build_field(lambda a, b: np.cos(a) * np.cos(b), x, y)
        errors.append(max_interior_error(derivative_x(phi, h), exact))
    assert observed_order(errors) > 1.9


def test_derivative_y_order_2():
    """derivative_y doit être d'ordre 2 sur sin(x)·cos(y)."""
    errors = []
    for n in (33, 65, 129):
        x = np.linspace(0.0, 1.0, n)
        y = np.linspace(0.0, 1.0, n)
        h = x[1] - x[0]
        phi = build_field(lambda a, b: np.sin(a) * np.cos(b), x, y)
        exact = build_field(lambda a, b: -np.sin(a) * np.sin(b), x, y)
        errors.append(max_interior_error(derivative_y(phi, h), exact))
    assert observed_order(errors) > 1.9


def test_divergence_order_2():
    """divergence de (sin x cos y, cos x sin y) = 2 cos x cos y, ordre 2."""
    errors = []
    for n in (33, 65, 129):
        x = np.linspace(0.0, 1.0, n)
        y = np.linspace(0.0, 1.0, n)
        h = x[1] - x[0]
        u = build_field(lambda a, b: np.sin(a) * np.cos(b), x, y)
        v = build_field(lambda a, b: np.cos(a) * np.sin(b), x, y)
        exact = build_field(lambda a, b: 2.0 * np.cos(a) * np.cos(b), x, y)
        errors.append(max_interior_error(divergence(u, v, h, h), exact))
    assert observed_order(errors) > 1.9


def test_upwind1_order_1():
    """upwind1 doit être d'ordre 1 (vitesse constante positive)."""
    errors = []
    for n in (33, 65, 129):
        x = np.linspace(0.0, 1.0, n)
        y = np.linspace(0.0, 1.0, n)
        h = x[1] - x[0]
        phi = build_field(lambda a, b: np.sin(a) * np.cos(b), x, y)
        u = np.ones((n, n))
        v = 0.5 * np.ones((n, n))
        exact = build_field(
            lambda a, b: np.cos(a) * np.cos(b) - 0.5 * np.sin(a) * np.sin(b), x, y
        )
        errors.append(max_interior_error(upwind1(phi, u, v, h, h), exact))
    order = observed_order(errors)
    assert 0.9 < order < 1.2


def test_walls_are_zero():
    """Les nœuds de paroi doivent rester à zéro (contrat du guide)."""
    n = 17
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    h = x[1] - x[0]
    phi = build_field(lambda a, b: np.sin(a) * np.cos(b), x, y)
    result = derivative_x(phi, h)
    for k in range(n):
        assert result[0, k] == 0.0
        assert result[n - 1, k] == 0.0
        assert result[k, 0] == 0.0
        assert result[k, n - 1] == 0.0