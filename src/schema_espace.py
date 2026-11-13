import numpy as np
from numba import njit


@njit
def upwind1(phi, u, v, dx, dy):
    """
    Terme convectif udphi/dx + vdphi/dy, schema amont (upwind) d'ordre 1.

    La derivee est decentree du cote d'ou vient l'information : difference
    arriere si la vitesse locale est positive, avant sinon. Seuls les noeuds
    interieurs sont calcules.

    Parameters
    ----------
    phi : numpy.ndarray, shape (nx, ny)
        Champ transporte (u ou v).
    u : numpy.ndarray, shape (nx, ny)
        Composante x de la vitesse d'advection.
    v : numpy.ndarray, shape (nx, ny)
        Composante y de la vitesse d'advection.
    dx : float
        Pas d'espace en x.
    dy : float
        Pas d'espace en y.

    Returns
    -------
    phi_new: numpy.ndarray, shape (nx, ny)
        Valeur de u dphi/dx + v dphi/dy aux nouds interieurs, 0 sur les parois.
    """
    nx, ny = phi.shape
    phi_new = np.zeros((nx, ny))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):

            # Derivee en x : decentree selon le signe de u
            if u[i, j] > 0.0:
                dphi_dx = (phi[i, j] - phi[i - 1, j]) / dx
            else:
                dphi_dx = (phi[i + 1, j] - phi[i, j]) / dx

            # Derivee en y : decentree selon le signe de v
            if v[i, j] > 0.0:
                dphi_dy = (phi[i, j] - phi[i, j - 1]) / dy
            else:
                dphi_dy = (phi[i, j + 1] - phi[i, j]) / dy

            phi_new[i, j] = u[i, j] * dphi_dx + v[i, j] * dphi_dy

    return phi_new


@njit
def centered2(phi, u, v, dx, dy):
    """
    Differences centrees d'ordre 2.

    Instable avec Euler explicite : a utiliser avec RK2 au minimum.

    Parameters
    ----------
    phi : numpy.ndarray, shape (nx, ny)
        Champ transporte (u ou v).
    u : numpy.ndarray, shape (nx, ny)
        Composante x de la vitesse d'advection.
    v : numpy.ndarray, shape (nx, ny)
        Composante y de la vitesse d'advection.
    dx : float
        Pas d'espace en x.
    dy : float
        Pas d'espace en y.

    Returns 
    -------
    phi_new : numpy.ndarray, shape (nx, ny)
        Valeur de u·dphi/dx + v·dphi/dy aux nœuds interieurs, 0 sur les parois.
    """
    nx, ny = phi.shape
    phi_new = np.zeros((nx, ny))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            dphi_dx = (phi[i + 1, j] - phi[i - 1, j]) / (2.0 * dx)
            dphi_dy = (phi[i, j + 1] - phi[i, j - 1]) / (2.0 * dy)
            phi_new[i, j] = u[i, j] * dphi_dx + v[i, j] * dphi_dy

    return phi_new


SPACE_SCHEMES = {"upwind1": upwind1, "centered2": centered2}


# def convective_term(phi, u, v, dx, dy, scheme):
#     """
#     Calcule u·dphi/dx + v·dphi/dy avec le schema choisi par son nom.

#     Parameters
#     ----------
#     phi : numpy.ndarray, shape (nx, ny)
#         Champ transporte (u ou v).
#     u, v : numpy.ndarray, shape (nx, ny)
#         Composantes de la vitesse d'advection.
#     dx, dy : float
#         Pas d'espace en x et en y.
#     scheme : str
#         Nom du schema, cle de SPACE_SCHEMES (par["advection_scheme"]).

#     Returns
#     -------
#     numpy.ndarray, shape (nx, ny)
#         Terme convectif, nul sur les nœuds de paroi.

#     Raises
#     ------
#     ValueError
#         Si le nom du schema est inconnu.
#     """
#     if scheme not in SPACE_SCHEMES:
#         raise ValueError(f"Schema d'espace inconnu : {scheme}")
#     return SPACE_SCHEMES[scheme](phi, u, v, dx, dy)


@njit
def derivative_x(phi, dx):
    """
    Dérivée première en x, différences centrées d'ordre 2.

    dphi/dx[i, j] = (phi[i+1, j] - phi[i-1, j]) / (2 dx)

    Seuls les nœuds intérieurs sont calculés ; les nœuds de paroi restent à zéro.

    Parameters
    ----------
    phi : numpy.ndarray, shape (nx, ny)
        Champ à dériver (u, v ou p).
    dx : float
        Pas d'espace en x.

    Returns
    -------
    dphi_dx : numpy.ndarray, shape (nx, ny)
        Dérivée en x aux nœuds intérieurs, 0 sur les parois.
    """
    nx, ny = phi.shape
    dphi_dx = np.zeros((nx, ny))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            dphi_dx[i, j] = (phi[i + 1, j] - phi[i - 1, j]) / (2.0 * dx)

    return dphi_dx


@njit
def derivative_y(phi, dy):
    """
    Dérivée première en y, différences centrées d'ordre 2.

    dphi/dy[i, j] = (phi[i, j+1] - phi[i, j-1]) / (2 dy)

    Seuls les nœuds intérieurs sont calculés ; les nœuds de paroi restent à zéro.

    Parameters
    ----------
    phi : numpy.ndarray, shape (nx, ny)
        Champ à dériver (u, v ou p).
    dy : float
        Pas d'espace en y.

    Returns
    -------
    dphi_dy : numpy.ndarray, shape (nx, ny)
        Dérivée en y aux nœuds intérieurs, 0 sur les parois.
    """
    nx, ny = phi.shape
    dphi_dy = np.zeros((nx, ny))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            dphi_dy[i, j] = (phi[i, j + 1] - phi[i, j - 1]) / (2.0 * dy)

    return dphi_dy


@njit
def divergence(u, v, dx, dy):
    """
    Divergence de la vitesse : du/dx + dv/dy, centrée d'ordre 2.

    Somme de derivative_x(u, dx) et derivative_y(v, dy). Nulle sur les
    nœuds de paroi.

    Parameters
    ----------
    u : numpy.ndarray, shape (nx, ny)
        Composante x de la vitesse.
    v : numpy.ndarray, shape (nx, ny)
        Composante y de la vitesse.
    dx : float
        Pas d'espace en x.
    dy : float
        Pas d'espace en y.

    Returns
    -------
    div : numpy.ndarray, shape (nx, ny)
        Divergence aux nœuds intérieurs, 0 sur les parois.
    """
    nx, ny = u.shape
    du_dx = derivative_x(u, dx)
    dv_dy = derivative_y(v, dy)
    div = np.zeros((nx, ny))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            div[i, j] = du_dx[i, j] + dv_dy[i, j]

    return div