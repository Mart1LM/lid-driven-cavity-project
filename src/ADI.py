"""
adi.py : diffusion par la méthode ADI (Alternating Direction Implicit)
======================================================================
Schéma de Peaceman-Rachford : d'ordre 2 en temps ET en espace, stable quel que soit dt.

PROBLÈME RÉSOLU (étape 2 du pas fractionné)
-------------------------------------------
    du/dt = nu * (d2u/dx2 + d2u/dy2)        et la même chose pour v

Un schéma implicite en 2D demanderait de résoudre un gros système à nx*ny inconnues.
L'ADI contourne cela en traitant UNE direction à la fois, ce qui ne donne que des
systèmes TRIDIAGONAUX (peu coûteux : algorithme de Thomas).

LE SCHÉMA : deux demi-pas de dt/2
---------------------------------
Notons dxx et dyy les différences finies centrées d'ordre 2 en x et en y
(phi[i+1] - 2 phi[i] + phi[i-1]) / dx^2, et
    fo_x = nu*dt/dx^2        ry = nu*dt/dy^2

    demi-pas 1 : implicite en x, explicite en y
        (1 - Fox/2 * Dxx) phi_demi = (1 + ry/2 * Dyy) phi_n
    demi-pas 2 : implicite en y, explicite en x
        (1 - ry/2 * Dyy) phi_n+1  = (1 + fo_x/2 * Dxx) phi_demi

Dans chaque demi-pas, le membre de droite est connu, et à gauche il reste, pour chaque
ligne (ou colonne) de la grille, un système tridiagonal indépendant des autres.
Ici Dxx désigne l'opérateur « à 3 points » ci-dessus (ce n'est pas schemas_espace.laplacian,
car l'ADI a besoin des deux directions séparément).

PROPRIÉTÉS
----------
- ordre 2 en temps et en espace ;
- inconditionnellement stable : plus de contrainte dt <= fourier * dx^2 / nu ;
- mais pas de pas de temps énorme conseillé : les modes très rapides ne sont presque
  pas amortis (le schéma reste précis, mais peut osciller si dt est très au-dessus de
  la limite explicite, surtout près des coins du couvercle).

CONDITIONS AUX LIMITES
----------------------
Seuls les nœuds intérieurs sont inconnus ; les parois sont des valeurs imposées
(Dirichlet) fournies par conditions_limite.apply_bc_velocity :
- demi-pas 1 : parois à l'instant t (début du pas) ;
- demi-pas 2 : parois à l'instant t + dt (fin du pas), comme dans le reste du code.
Si le couvercle est constant (ramp_time = 0, ou rampe terminée), c'est exact.
Pendant une rampe rapide, l'instant intermédiaire est traité de façon approchée.

COMMENT L'UTILISER (3 réglages)
-------------------------------
1. diffusion.py : importer diffusion_adi et ajouter  "adi": diffusion_adi  à DIFFUSION_METHODS.
2. .yml : diffusion_method: adi
3. fonctions_diverses.compute_dt : en ADI, retirer la contrainte de Fourier (garder la
   contrainte advective cfl*dx/|u|max), voir INTERFACES section 7.
par["time_scheme"] est IGNORÉ pour la diffusion en ADI (le schéma est son propre schéma en temps).

CONVENTIONS (voir INTERFACES.md)
--------------------------------
- Tableaux (nx, ny), [i, j] <-> point (x[i], y[j]).
- Aucun argument n'est modifié : on renvoie de nouveaux tableaux.
"""
import numpy as np
from numba import njit
from conditions_limite import apply_bc_velocity


# ==========================================================================
# Résolution d'un système tridiagonal (algorithme de Thomas)
# ==========================================================================

@njit
def solve_tridiagonal(a, b, c, d):  #algo à clarifier
    """
    Résout T x = d pour une matrice tridiagonale T à coefficients CONSTANTS :
    a sur la sous-diagonale, b sur la diagonale, c sur la sur-diagonale.

    L'algorithme de Thomas est une élimination de Gauss simplifiée : une passe
    « descendante » qui élimine la sous-diagonale, puis une remontée.

    Arguments :
        a, b, c : nombres (coefficients de la matrice)
        d       : tableau (m, k). Chaque COLONNE est un second membre : on résout
                  donc k systèmes de taille m en même temps (les k lignes de la grille).

    Retourne : x, tableau (m, k), solution de chaque système.
    """
    m = d.shape[0]
    c_prime = np.zeros(m)
    d_prime = np.zeros_like(d)

    # passe descendante
    c_prime[0] = c / b
    d_prime[0] = d[0] / b
    for i in range(1, m):
        denom = b - a * c_prime[i - 1]
        c_prime[i] = c / denom
        d_prime[i] = (d[i] - a * d_prime[i - 1]) / denom

    # remontée
    x = np.zeros_like(d)
    x[-1] = d_prime[-1]
    for i in range(m - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


# ==========================================================================
# Un demi-pas ADI
# ==========================================================================

@njit
def half_step_x(phi, fo_x, fo_y):
    """
    Demi-pas « implicite en x, explicite en y » :
        (1 - /2 * Dxx) phi_new = (1 + fo_y/2 * Dyy) phi        (nœuds intérieurs)

    Arguments :
        phi    : champ (nx, ny), parois déjà à la bonne valeur
        fo_x, fo_y : nu*dt/dx^2 et nu*dt/dy^2

    Retourne : nouveau tableau (nx, ny) ; les parois sont recopiées de phi.
    """
    inner = phi[1:-1, 1:-1]
    # membre de droite : phi + (fo_y/2) * Dyy(phi)   (explicite en y)
    rhs = inner + 0.5 * fo_y * (phi[1:-1, 2:] - 2.0 * inner + phi[1:-1, :-2])
    # les parois gauche/droite sont connues : leur contribution passe à droite
    rhs[0, :] += 0.5 * fo_x * phi[0, 1:-1]
    rhs[-1, :] += 0.5 * fo_x * phi[-1, 1:-1]

    new = phi.copy()
    new[1:-1, 1:-1] = solve_tridiagonal(-0.5 * fo_x, 1.0 + fo_x, -0.5 * fo_x, rhs)
    return new


@njit
def half_step_y(phi, fo_x, fo_y):
    """
    Demi-pas « implicite en y, explicite en x » :
        (1 - fo_y/2 * Dyy) phi_new = (1 + fo_x/2 * Dxx) phi        (nœuds intérieurs)

    Astuce : c'est le demi-pas en x appliqué au tableau TRANSPOSÉ, en échangeant fo_x et fo_y.
    Arguments et retour : comme half_step_x.
    """
    return np.ascontiguousarray(half_step_x(phi.T, fo_y, fo_x).T) #fonction ascontiguous à clarifier 


# ==========================================================================
# Fonction appelée par diffusion.py
# ==========================================================================

@njit
def diffusion_adi(u_star, v_star, t, dt, dx, dy, par):
    """
    Étape 2 du pas fractionné par la méthode ADI (Peaceman-Rachford).
    Même signature que diffusion.diffusion_step.

    Arguments :
        u_star, v_star : vitesse après l'étape 1, tableaux (nx, ny)
        t              : instant au début du pas
        dt             : pas de temps
        dx, dy         : pas d'espace
        par            : dictionnaire des paramètres (utilise par["nu"] et, via
                         apply_bc_velocity, les paramètres du couvercle)

    Retourne : (u_2star, v_2star), NOUVEAUX tableaux (nx, ny).
        u_star et v_star ne sont pas modifiés.
    """
    fo_x = par["nu"] * dt / dx**2
    fo_y = par["nu"] * dt / dy**2

    u = u_star.copy()
    v = v_star.copy()

    # --- demi-pas 1 : implicite en x, parois à l'instant t --- # Pourquoi pas d'instant t+dt/2 ici ?
    apply_bc_velocity(u, v, t, par)
    u = half_step_x(u, fo_x, fo_y)
    v = half_step_x(v, fo_x, fo_y)

    # --- demi-pas 2 : implicite en y, parois à l'instant t + dt ---
    apply_bc_velocity(u, v, t + dt, par)
    u = half_step_y(u, fo_x, fo_y)
    v = half_step_y(v, fo_x, fo_y)

    apply_bc_velocity(u, v, t + dt, par)         # parois propres en fin de pas
    return u, v