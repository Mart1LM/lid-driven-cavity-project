import numpy as np
from numba import njit


@njit
def euler(q, F, dt):
    """
    Euler explicite, ordre 1 : q^{n+1} = q^n + dt * F(q^n).

    Parameters
    ----------
    q : tuple of numpy.ndarray
        Inconnue au temps n,(u, v), chaque tableau de forme (nx, ny).
    F : callable
        Dérivée en temps : F(q) renvoie un tuple (du_dt, dv_dt) de même
        structure que q.
    dt : float
        Pas de temps [s].

    Returns
    -------
    q_new : tuple of numpy.ndarray
        Inconnue au temps n+1 (q n'est pas modifié).
    """
    k1 = F(q)

    q_new = []
    for c in range(len(q)):
        nx, ny = q[c].shape
        champ_new = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                champ_new[i, j] = q[c][i, j] + dt * k1[c][i, j]
        q_new.append(champ_new)

    return tuple(q_new)

@njit
def _heun_update(a, k1, k2, dt):
    """
    Calcule a + dt/2 * (k1 + k2), noeud par noeud (mise à jour finale de Heun).

    Parameters
    ----------
    a : numpy.ndarray, shape (nx, ny)
        Tableau au temps n.
    k1 : numpy.ndarray, shape (nx, ny)
        Dérivée en temps évaluée en q^n.
    k2 : numpy.ndarray, shape (nx, ny)
        Dérivée en temps évaluée à l'étage intermédiaire.
    dt : float
        Pas de temps [s].

    Returns
    -------
    out : numpy.ndarray, shape (nx, ny)
        Tableau au temps n+1.
    """
    nx, ny = a.shape
    out = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            out[i, j] = a[i, j] + 0.5 * dt * (k1[i, j] + k2[i, j])
    return out


import numpy as np


def rk2_heun(q, F, dt):
    """
    Méthode de Heun (RK2), ordre 2.

        k1 = F(q^n)
        k2 = F(q^n + dt * k1)
        q^{n+1} = q^n + dt/2 * (k1 + k2)

    Parameters
    ----------
    q : tuple of numpy.ndarray
        Inconnue au temps n, par exemple (u, v), chaque tableau de forme (nx, ny).
    F : callable
        Dérivée en temps : F(q) renvoie un tuple (du_dt, dv_dt) de même
        structure que q.
    dt : float
        Pas de temps [s].

    Returns
    -------
    q_new : tuple of numpy.ndarray
        Inconnue au temps n+1 (q n'est pas modifié).
    """
    # Étage 1 : pente au début du pas
    k1 = F(q)

    # État intermédiaire : q^n + dt * k1
    q_stage = []
    for c in range(len(q)):
        nx, ny = q[c].shape
        champ_stage = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                champ_stage[i, j] = q[c][i, j] + dt * k1[c][i, j]
        q_stage.append(champ_stage)

    # Étage 2 : pente à l'état intermédiaire
    k2 = F(tuple(q_stage))

    # Mise à jour finale : moyenne des deux pentes
    q_new = []
    for c in range(len(q)):
        nx, ny = q[c].shape
        champ_new = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                champ_new[i, j] = q[c][i, j] + 0.5 * dt * (k1[c][i, j] + k2[c][i, j])
        q_new.append(champ_new)

    return tuple(q_new)

def rk4(q, F, dt):
    """
    Méthode de Runge-Kutta classique (RK4), ordre 4.

        k1 = F(q^n)
        k2 = F(q^n + dt/2 * k1)
        k3 = F(q^n + dt/2 * k2)
        k4 = F(q^n + dt * k3)
        q^{n+1} = q^n + dt/6 * (k1 + 2 k2 + 2 k3 + k4)

    Parameters
    ----------
    q : tuple of numpy.ndarray
        Inconnue au temps n, par exemple (u, v), chaque tableau de forme (nx, ny).
    F : callable
        Dérivée en temps : F(q) renvoie un tuple (du_dt, dv_dt) de même
        structure que q.
    dt : float
        Pas de temps [s].

    Returns
    -------
    q_new : tuple of numpy.ndarray
        Inconnue au temps n+1 (q n'est pas modifié).
    """
    # Coefficients des états intermédiaires : q + coef * dt * k_précédent
    stage_coefs = (0.5, 0.5, 1.0)

    # Étage 1 : pente au début du pas
    k = [F(q)]

    # Étages 2 à 4 : pente évaluée à chaque état intermédiaire
    for s in range(3):
        q_stage = []
        for c in range(len(q)):
            nx, ny = q[c].shape
            champ_stage = np.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    champ_stage[i, j] = q[c][i, j] + stage_coefs[s] * dt * k[s][c][i, j]
            q_stage.append(champ_stage)
        k.append(F(tuple(q_stage)))

    # Mise à jour finale : moyenne pondérée des quatre pentes
    q_new = []
    for c in range(len(q)):
        nx, ny = q[c].shape
        champ_new = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                champ_new[i, j] = q[c][i, j] + dt / 6.0 * (
                    k[0][c][i, j]
                    + 2.0 * k[1][c][i, j]
                    + 2.0 * k[2][c][i, j]
                    + k[3][c][i, j]
                )
        q_new.append(champ_new)

    return tuple(q_new)
TIME_SCHEMES = {"euler": euler, "rk2_heun": rk2_heun,"rk4":rk4}


def get_time_scheme(name):
    """
    Renvoie la fonction de schéma en temps associée à son nom.

    Parameters
    ----------
    name : str
        Nom du schéma, clé de TIME_SCHEMES (par["time_scheme"]).

    Returns
    -------
    callable
        Fonction (q, F, dt) -> q_new.

    Raises
    ------
    ValueError
        Si le nom du schéma est inconnu.
    """
    if name not in TIME_SCHEMES:
        raise ValueError(f"Schéma en temps inconnu : {name}")
    return TIME_SCHEMES[name]

