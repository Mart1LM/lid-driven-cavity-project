"""
poisson.py : solveurs de l'équation de Poisson (utilisés UNIQUEMENT par l'étape 3, pression)
==============================================================================================

PROBLÈME RÉSOLU
---------------
    laplacien(p) = b        sur les nœuds intérieurs,     dp/dn = 0 sur les parois

avec b = rho/dt * div (calculé dans pression.py). On discrétise le laplacien
par le stencil à 5 points (le même que schemas_espace.laplacian) :

    cx*(p[i+1,j] - 2 p[i,j] + p[i-1,j]) + cy*(p[i,j+1] - 2 p[i,j] + p[i,j-1]) = b[i,j]
    avec cx = 1/dx^2 et cy = 1/dy^2.

On isole p[i,j] pour obtenir la « mise à jour » d'un nœud :

    p_nouveau[i,j] = ( cx*(p[i+1,j] + p[i-1,j]) + cy*(p[i,j+1] + p[i,j-1]) - b[i,j] )
                     / ( 2*(cx + cy) )

Les trois méthodes ne diffèrent que par la façon d'appliquer cette formule :

    Jacobi        : on calcule TOUS les nouveaux p à partir des ANCIENS (tableau à part).
    Gauss-Seidel  : on écrase p au fur et à mesure, donc chaque nœud profite des
                    valeurs déjà mises à jour de ses voisins (en théorie ~2x moins
                    d'itérations que Jacobi ; mesuré ici : un peu moins).
    SOR           : Gauss-Seidel + « sur-relaxation » : on dépasse un peu la valeur
                    de Gauss-Seidel, p = (1-omega)*p_ancien + omega*p_GS. Avec un bon
                    omega (entre 1 et 2) c'est beaucoup plus rapide.
                    omega = 1 redonne exactement Gauss-Seidel.

STRUCTURE DU FICHIER
--------------------
    1. noyaux de calcul (boucles sur les nœuds) : step_jacobi, step_sor, residual_max
    2. une boucle d'itération commune : _iterate (arrêt, CL, historique)
    3. les solveurs : jacobi, sor  (signatures de INTERFACES)
    4. solve_poisson : choisit le solveur selon par["poisson_solver"]

POUR AJOUTER UN SOLVEUR (gradient conjugué, red-black...) :
    écrire une fonction « sweep » (voir _sweep_jacobi), un solveur qui appelle
    _iterate avec ce sweep, et l'ajouter au dictionnaire POISSON_SOLVERS.


CONVENTIONS (voir INTERFACES.md)
--------------------------------
- Tableaux de forme (nx, ny), p[i, j] <-> point (x[i], y[j]).
  (Attention : dans un script « meshgrid » par défaut, c'est l'inverse, p[j, i].)
- Seuls les nœuds intérieurs 1..n-2 sont calculés ; les nœuds de paroi sont fixés
  par conditions_limite.apply_bc_pressure, appelée à CHAQUE itération.
- Aucun argument n'est modifié : les solveurs travaillent sur une copie de p0.
"""
import numpy as np

from conditions_limite import apply_bc_pressure

try:
    from numba import njit
except ImportError:                      # pas de numba : même code, en Python pur
    def njit(func):
        return func


# Gauss-Seidel = SOR avec omega = 1 (définition mathématique, pas un réglage).
OMEGA_GAUSS_SEIDEL = 1.0


# ==========================================================================
# 1. Noyaux de calcul : une boucle sur les nœuds intérieurs
# ==========================================================================
@njit
def step_jacobi(p, p_new, b, cx, cy):
    """
    Un balayage de Jacobi : p_new[i,j] calculé à partir des valeurs de p.

    Arguments :
        p     : pression actuelle, (nx, ny), NON modifiée
        p_new : tableau (nx, ny) où l'on écrit le résultat (nœuds intérieurs)
        b     : second membre, (nx, ny)
        cx,cy : 1/dx^2 et 1/dy^2
    """
    nx, ny = p.shape
    inv_diag = 1.0 / (2.0 * (cx + cy))
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            p_new[i, j] = ((p[i + 1, j] + p[i - 1, j]) * cx
                           + (p[i, j + 1] + p[i, j - 1]) * cy
                           - b[i, j]) * inv_diag


@njit
def step_sor(p, b, cx, cy, omega):
    """
    Un balayage de SOR, EN PLACE (omega = 1 : Gauss-Seidel).

    Arguments :
        p     : pression, (nx, ny), MODIFIÉE (nœuds intérieurs)
        b     : second membre, (nx, ny)
        cx,cy : 1/dx^2 et 1/dy^2
        omega : coefficient de sur-relaxation
    """
    nx, ny = p.shape
    inv_diag = 1.0 / (2.0 * (cx + cy))
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            p_gs = ((p[i + 1, j] + p[i - 1, j]) * cx
                    + (p[i, j + 1] + p[i, j - 1]) * cy
                    - b[i, j]) * inv_diag
            p[i, j] = (1.0 - omega) * p[i, j] + omega * p_gs


@njit
def residual_max(p, b, cx, cy):
    """
    Résidu de l'équation : max sur les nœuds intérieurs de |laplacien(p) - b|.
    C'est « de combien p rate l'équation de Poisson » (0 = solution exacte).

    Retourne : float.
    """
    nx, ny = p.shape
    res = 0.0
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            lap = (cx * (p[i + 1, j] - 2.0 * p[i, j] + p[i - 1, j])
                   + cy * (p[i, j + 1] - 2.0 * p[i, j] + p[i, j - 1]))
            r = abs(lap - b[i, j])
            if r > res:
                res = r
    return res


# ==========================================================================
# 2. Boucle d'itération commune
# ==========================================================================
def _sweep_jacobi(p, work, b, cx, cy, omega):
    """
    Un balayage de Jacobi. Écrit dans `work`, puis échange les deux tableaux :
    le nouveau p devient `work`, l'ancien p devient le tableau de travail.

    Retourne : (p, work) mis à jour.
    """
    step_jacobi(p, work, b, cx, cy)
    return work, p


def _sweep_sor(p, work, b, cx, cy, omega):
    """
    Un balayage de SOR (en place ; `work` ne sert pas).

    Retourne : (p, work).
    """
    step_sor(p, b, cx, cy, omega)
    return p, work


def _iterate(sweep, b, p0, dx, dy, tol, maxiter, omega, history):
    """
    Boucle commune : répète « balayage, CL de pression, calcul du résidu »
    jusqu'à convergence ou jusqu'à maxiter.

    Critère d'arrêt (RELATIF) :  max|laplacien(p) - b|  <=  tol * max|b|.
    On compare au « poids » du second membre pour que tol ait le même sens
    quelle que soit l'échelle de b (qui dépend de rho et de dt).

    Arguments :
        sweep   : fonction de balayage (_sweep_jacobi ou _sweep_sor)
        b       : second membre, (nx, ny)
        p0      : pression de départ, (nx, ny) (non modifiée)
        dx, dy  : pas d'espace
        tol     : tolérance relative
        maxiter : nombre maximal d'itérations
        omega   : coefficient de sur-relaxation (ignoré par Jacobi)
        history : None, ou une liste dans laquelle on ajoute le résidu à chaque
                  itération (pour tracer la courbe de convergence)

    Retourne : (p, n_iter, res_poisson)
        p           : nouveau tableau (nx, ny)
        n_iter      : nombre d'itérations faites. n_iter == maxiter signifie
                      très probablement « pas convergé »
        res_poisson : résidu absolu final, max|laplacien(p) - b|
    """
    cx = 1.0 / dx**2
    cy = 1.0 / dy**2

    p = p0.copy()                 # on ne touche jamais à p0
    apply_bc_pressure(p)          # départ cohérent avec dp/dn = 0
    work = p.copy()               # tableau de travail (utile pour Jacobi)

    # Seuil d'arrêt ; eps évite un seuil nul si b = 0 (fluide au repos).
    res_target = tol * max(np.max(np.abs(b)), np.finfo(float).eps)

    res = residual_max(p, b, cx, cy)
    if history is not None:
        history.append(res)

    n_iter = 0
    while res > res_target and n_iter < maxiter:
        p, work = sweep(p, work, b, cx, cy, omega)
        apply_bc_pressure(p)      # en place : dp/dn = 0 à chaque itération
        res = residual_max(p, b, cx, cy)
        n_iter += 1
        if history is not None:
            history.append(res)

    return p, n_iter, res


# ==========================================================================
# 3. Les solveurs (signatures de INTERFACES)
# ==========================================================================
def jacobi(b, p0, dx, dy, tol, maxiter, history=None):
    """
    Itérations de Jacobi pour laplacien(p) = b.

    Arguments :
        b       : second membre, (nx, ny)
        p0      : pression de départ, (nx, ny) (non modifiée)
        dx, dy  : pas d'espace
        tol     : tolérance relative sur le résidu (voir _iterate)
        maxiter : nombre maximal d'itérations
        history : optionnel, liste où l'on récolte le résidu de chaque itération

    Retourne : (p, n_iter, res_poisson)
    """
    return _iterate(_sweep_jacobi, b, p0, dx, dy, tol, maxiter,
                    omega=None, history=history)


def sor(b, p0, dx, dy, tol, maxiter, omega, history=None):
    """
    Gauss-Seidel avec sur-relaxation (SOR) pour laplacien(p) = b.
    omega = 1 : Gauss-Seidel simple.

    Arguments :
        b       : second membre, (nx, ny)
        p0      : pression de départ, (nx, ny) (non modifiée)
        dx, dy  : pas d'espace
        tol     : tolérance relative sur le résidu (voir _iterate)
        maxiter : nombre maximal d'itérations
        omega   : coefficient de sur-relaxation, 0 < omega < 2
                  (voir optimal_omega pour une bonne valeur)
        history : optionnel, liste où l'on récolte le résidu de chaque itération

    Retourne : (p, n_iter, res_poisson)
    """
    return _iterate(_sweep_sor, b, p0, dx, dy, tol, maxiter,
                    omega=omega, history=history)


def optimal_omega(nx, ny, dx, dy):
    """
    Estimation du omega optimal de SOR pour le stencil à 5 points.

    Formule classique : omega = 2 / (1 + sqrt(1 - rho_J^2)), où rho_J est le
    rayon spectral de Jacobi. Pour une grille carrée, cela donne
    omega = 2 / (1 + sin(pi/N)). Les conditions de Neumann ne sont pas
    exactement le cas de la formule (établie pour Dirichlet) : c'est une bonne
    valeur de départ, à affiner par essais autour de ce nombre.

    Arguments :
        nx, ny : nombre de nœuds en x et en y
        dx, dy : pas d'espace

    Retourne : float, entre 1 et 2.
    """
    cx = 1.0 / dx**2
    cy = 1.0 / dy**2
    rho_jacobi = (cx * np.cos(np.pi / (nx - 1)) + cy * np.cos(np.pi / (ny - 1))) / (cx + cy)
    return 2.0 / (1.0 + np.sqrt(1.0 - rho_jacobi**2))


# ==========================================================================
# 4. Choix du solveur à partir du .yml
# ==========================================================================
def _solve_jacobi(b, p0, dx, dy, par):
    p, _, _ = jacobi(b, p0, dx, dy, par["tol"], par["maxiter"])
    return p


def _solve_gauss_seidel(b, p0, dx, dy, par):
    p, _, _ = sor(b, p0, dx, dy, par["tol"], par["maxiter"], OMEGA_GAUSS_SEIDEL)
    return p


def _solve_sor(b, p0, dx, dy, par):
    p, _, _ = sor(b, p0, dx, dy, par["tol"], par["maxiter"], par["omega"])
    return p


POISSON_SOLVERS = {
    "jacobi": _solve_jacobi,
    "gauss_seidel": _solve_gauss_seidel,
    "sor": _solve_sor,
}


def solve_poisson(b, p0, dx, dy, par):
    """
    Résout laplacien(p) = b avec le solveur choisi par par["poisson_solver"]
    ("jacobi", "gauss_seidel" ou "sor"). Utilise aussi par["tol"], par["maxiter"]
    et, pour "sor", par["omega"].

    Arguments :
        b      : second membre, (nx, ny)
        p0     : pression de départ, (nx, ny) (non modifiée)
        dx, dy : pas d'espace
        par    : dictionnaire des paramètres

    Retourne : p, NOUVEAU tableau (nx, ny).

    Lève ValueError si le nom de solveur est inconnu.
    """
    name = par["poisson_solver"]
    if name not in POISSON_SOLVERS:
        raise ValueError(
            f"poisson_solver inconnu : '{name}'. "
            f"Choix possibles : {list(POISSON_SOLVERS)}"
        )
    return POISSON_SOLVERS[name](b, p0, dx, dy, par)