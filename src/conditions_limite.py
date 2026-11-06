"""
conditions_limite.py : TOUTES les conditions aux limites (et rien d'autre).
 
Grille simple : u, v, p sont de forme (nx, ny), les nœuds des parois font partie de la grille.
    i = 0 (gauche), i = nx-1 (droite), j = 0 (bas), j = ny-1 (haut = couvercle)
 
Fonctions publiques
-------------------
lid_velocity(t, par)                -> float   vitesse U(t) du couvercle
apply_bc_velocity(u, v, t, par)     -> None    MODIFIE u, v EN PLACE
apply_bc_pressure(p)                -> None    MODIFIE p EN PLACE
"""
import numpy as np
 
 
def lid_velocity(t: float, par: dict) -> float:
    """Vitesse du couvercle à l'instant t.
 
    ramp_time = 0 : démarrage brutal, U(t) = U0 dès t = 0.
    ramp_time > 0 : montée lisse de 0 à U0 sur ramp_time, puis U0 :
                    U(t) = U0 * (1 - cos(pi t / ramp_time)) / 2.
    """
    ramp = par["ramp_time"]
    if ramp <= 0.0 or t >= ramp:
        return par["U0"]
    return par["U0"] * 0.5 * (1.0 - np.cos(np.pi * t / ramp))
 
 
def apply_bc_velocity(u: np.ndarray, v: np.ndarray, t: float, par: dict) -> None:
    """Impose la vitesse sur les nœuds de paroi (modification EN PLACE).
 
    - parois gauche, droite, bas : u = 0 et v = 0 (adhérence)
    - couvercle (j = ny-1)       : u = U(t), v = 0, coins compris
    Les nœuds intérieurs ne sont pas touchés.
    """
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    v[0, :] = 0.0
    v[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    u[:, -1] = lid_velocity(t, par)      # en dernier : les coins du haut valent U(t)
 
 
def apply_bc_pressure(p: np.ndarray) -> None:
    """Condition dp/dn = 0 sur les quatre parois, par recopie du voisin intérieur
    (modification EN PLACE). Les côtés gauche/droit sont traités avant haut/bas."""
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]
    p[:, 0] = p[:, 1]
    p[:, -1] = p[:, -2]
 