
"""
pression.py : ÉTAPE 3 du pas fractionné (correction de pression)
=================================================================

IDÉE (slides « Pressure step » et « Méthode prédicteur-correcteur »)
--------------------------------------------------------------------
Après les étapes 1 (transport) et 2 (diffusion), on dispose d'une vitesse
(u_2star, v_2star) qui n'a PAS une divergence nulle : le fluide « se comprime »
ou « se dilate » un peu, ce qui est interdit pour un fluide incompressible.

On corrige donc cette vitesse avec un gradient de pression :

    u_new = u_2star - dt/rho * dp/dx        (1)   correction
    v_new = v_2star - dt/rho * dp/dy

et on choisit p pour que la vitesse corrigée soit à divergence nulle :

    div(u_new, v_new) = 0                   (2)

En remplaçant (1) dans (2), on obtient l'équation de Poisson :

    laplacien(p) = rho/dt * div(u_2star, v_2star)           (3)

Une seule pression « p » satisfait (3) à une constante près (conditions de
Neumann partout), d'où le retrait de la moyenne à la fin.

LES 4 SOUS-ÉTAPES DE pressure_step
----------------------------------
    1. div  = divergence(u_2star, v_2star)         -> compute_divergence
    2. p    = solution de (3)                      -> solve_pressure
    3. grad = (dp/dx, dp/dy)                       -> compute_pressure_gradient
    4. u_new, v_new = formule (1) + CL de vitesse  -> correct_velocity

CHOIX DU SCHÉMA D'ESPACE (clé « pressure_scheme » du .yml)
----------------------------------------------------------
Ce fichier ne contient AUCUNE formule de différences finies : toutes sont dans
schemas_espace.py. Ici on choisit seulement QUEL schéma utiliser pour la
divergence (sous-étape 1) et pour le gradient de pression (sous-étape 3).

Le choix se fait dans le dictionnaire PRESSURE_SPACE_SCHEMES ci-dessous.
Pour ajouter une combinaison : ajouter UNE ligne dans ce dictionnaire
(et, si le schéma de dérivée est nouveau, une ligne dans schemas_espace.py).

    nom          divergence   gradient    remarque
    -----------  -----------  ----------  ------------------------------------
    centered2    centré       centré      choix de INTERFACES (par défaut).
                                          Facile, mais le Laplacien « réel »
                                          div(grad p) est un stencil large
                                          (pas 2dx) alors que Poisson utilise le
                                          stencil serré (pas dx) : la divergence
                                          finale n'est pas nulle. Mesuré sur un
                                          cas analytique : ~ dx^2 à l'intérieur,
                                          mais reste de l'ordre de 0.5 près des
                                          parois (ne diminue pas avec le
                                          maillage). Damier de pression possible.
    fwd_bwd      arrière      avant       div(grad p) = exactement le stencil
                                          serré à 5 points utilisé par Poisson.
                                          La divergence finale (mesurée avec ce
                                          même opérateur) est uniforme, égale à
                                          la moyenne de div retirée pour rendre
                                          Poisson solvable (vérifié : écart-type
                                          ~1e-9) ; elle est nulle si cette
                                          moyenne l'est.
                                          Mais chaque dérivée n'est que d'ordre 1 :
                                          l'erreur sur la vitesse est d'ordre dx
                                          et la divergence centrée, elle, n'est
                                          pas nulle.
    bwd_fwd      avant        arrière     symétrique de fwd_bwd.

Il n'y a pas de « meilleur » choix a priori : c'est justement ce que le
rapport peut comparer (divergence, énergie cinétique, profils de Ghia).

Pour l'intuition : « avant » = on regarde le voisin de droite (i+1),
« arrière » = le voisin de gauche (i-1). Prendre un sens pour le gradient et le
sens opposé pour la divergence, c'est ce que fait naturellement une grille
décalée, d'où l'intérêt de cette combinaison sur notre grille simple.

CONVENTIONS (voir INTERFACES.md)
--------------------------------
- Tableaux de forme (nx, ny), [i, j] <-> point (x[i], y[j]).
- Les fonctions « étape » ne modifient PAS leurs arguments : elles renvoient de
  nouveaux tableaux. Les seules modifications en place sont celles de
  apply_bc_velocity et apply_bc_pressure (conditions_limite.py).
- Les dérivées de schemas_espace sont nulles sur les parois : seuls les nœuds
  intérieurs 1..n-2 sont corrigés, la CL de vitesse s'occupe des parois.

CHANGEMENTS D'INTERFACE À DISCUTER AVEC LE GROUPE
-------------------------------------------------
(a) correct_velocity reçoit en plus `t` : elle appelle apply_bc_velocity(u, v, t,
    par) qui a besoin de t (la signature de INTERFACES l'oubliait).
(b) schemas_espace.derivative_x / derivative_y / divergence reçoivent un argument
    OPTIONNEL `scheme` (défaut "centered2") : voir schemas_espace_ajouts.py.
    Les appels existants ne changent pas.
(c) nouvelle clé optionnelle du .yml : pressure_scheme (défaut "centered2").
"""
import numpy as np

from schemas_espace import divergence, derivative_x, derivative_y
from conditions_limite import apply_bc_velocity, apply_bc_pressure
from poisson import solve_poisson


# ==========================================================================
# Catalogue des schémas d'espace utilisables pour l'étape de pression
# ==========================================================================
# Chaque entrée donne le nom du schéma (clé de FIRST_DERIVATIVE_SCHEMES dans
# schemas_espace.py) à utiliser pour la divergence et pour le gradient.
PRESSURE_SPACE_SCHEMES = {
    "centered2": {"div": "centered2", "grad": "centered2"},
    "fwd_bwd":   {"div": "backward1", "grad": "forward1"},
    "bwd_fwd":   {"div": "forward1",  "grad": "backward1"},
}


def get_pressure_schemes(par):
    """
    Lit par["pressure_scheme"] et renvoie les deux noms de schémas à utiliser.

    Arguments :
        par : dictionnaire des paramètres (clé optionnelle "pressure_scheme")

    Retourne : (div_scheme, grad_scheme), deux chaînes de caractères
        à transmettre à schemas_espace.

    Lève ValueError si le nom est inconnu.
    """
    name = par.get("pressure_scheme", "centered2")
    if name not in PRESSURE_SPACE_SCHEMES:
        raise ValueError(
            f"pressure_scheme inconnu : '{name}'. "
            f"Choix possibles : {list(PRESSURE_SPACE_SCHEMES)}"
        )
    choice = PRESSURE_SPACE_SCHEMES[name]
    return choice["div"], choice["grad"]


# ==========================================================================
# Sous-fonctions de l'étape 3
# ==========================================================================
def compute_divergence(u, v, dx, dy, par):
    """
    Divergence de la vitesse avec le schéma choisi pour la pression.

    Sert aussi à MESURER la qualité de la projection : pour comparer des
    schémas, il faut mesurer la divergence avec le MÊME schéma que celui qui
    a servi à projeter (voir projection_error).

    Arguments :
        u, v   : composantes de la vitesse, tableaux (nx, ny)
        dx, dy : pas d'espace
        par    : dictionnaire des paramètres

    Retourne : div, tableau (nx, ny), nul sur les nœuds de paroi.
    """
    div_scheme, _ = get_pressure_schemes(par)
    return divergence(u, v, dx, dy, scheme=div_scheme)


def compute_pressure_gradient(p, dx, dy, par):
    """
    Gradient de pression (dp/dx, dp/dy) avec le schéma choisi.

    Arguments :
        p      : pression, tableau (nx, ny) (CL de pression déjà appliquée)
        dx, dy : pas d'espace
        par    : dictionnaire des paramètres

    Retourne : (dpdx, dpdy), deux tableaux (nx, ny), nuls sur les parois.
    """
    _, grad_scheme = get_pressure_schemes(par)
    dpdx = derivative_x(p, dx, scheme=grad_scheme)
    dpdy = derivative_y(p, dy, scheme=grad_scheme)
    return dpdx, dpdy


def remove_mean(p):
    """
    Retire la moyenne de la pression (la pression n'est définie qu'à une
    constante près : si p est solution, p + C l'est aussi).

    Arguments :
        p : pression, tableau (nx, ny)

    Retourne : un NOUVEAU tableau (nx, ny) de moyenne nulle.
    """
    return p - np.mean(p)


def solve_pressure(div, p0, dt, dx, dy, par):
    """
    Résout l'équation de Poisson   laplacien(p) = rho/dt * div   (équation 3).

    Arguments :
        div    : divergence de (u_2star, v_2star), tableau (nx, ny)
        p0     : pression du pas précédent, tableau (nx, ny).
                 Sert de point de départ au solveur (« initial guess ») :
                 la pression change peu d'un pas à l'autre, donc on converge
                 plus vite qu'en partant de zéro.
        dt     : pas de temps
        dx, dy : pas d'espace
        par    : dictionnaire des paramètres (utilise par["rho"] et, via
                 poisson.solve_poisson, le choix du solveur)

    Retourne : p, NOUVEAU tableau (nx, ny), de moyenne nulle.

    Précaution de solvabilité : avec des CL de Neumann partout, le second
    membre b doit avoir une somme nulle (sinon le système n'a pas de solution
    et le solveur stagne, cf. INTERFACES section 8). On retire donc sa moyenne
    sur les nœuds intérieurs avant de résoudre. L'effet sur la solution est
    très faible, car mean(b) est petit pour un champ presque à divergence nulle.
    """
    # Second membre de Poisson : b = rho/dt * div
    b = par["rho"] / dt * div

    # Solvabilité : on enlève la moyenne de b (nœuds intérieurs seulement,
    # car les nœuds de paroi ne sont pas des inconnues).
    interior = (slice(1, -1), slice(1, -1))
    b[interior] -= np.mean(b[interior])

    # Résolution. On passe une COPIE de p0 pour ne jamais modifier l'argument.
    p = solve_poisson(b, p0.copy(), dx, dy, par)

    # Retrait de la constante arbitraire, puis CL de pression (en place sur p,
    # qui est notre propre tableau) pour que p[0,:] = p[1,:], etc.
    p = remove_mean(p)
    apply_bc_pressure(p)
    return p


def correct_velocity(u_2star, v_2star, p, t, dt, dx, dy, par):
    """
    Corrige la vitesse avec le gradient de pression (équation 1) :

        u_new = u_2star - dt/rho * dp/dx
        v_new = v_2star - dt/rho * dp/dy

    Arguments :
        u_2star, v_2star : vitesse après transport + diffusion, (nx, ny)
        p                : pression solution de Poisson, (nx, ny)
        t                : instant au DÉBUT du pas de temps.
                           La CL de vitesse est appliquée à t + dt (instant
                           auquel la vitesse corrigée est valable).
                           ATTENTION : à harmoniser avec diffusion_step.
        dt               : pas de temps
        dx, dy           : pas d'espace
        par              : dictionnaire des paramètres (utilise par["rho"])

    Retourne : (u_new, v_new), NOUVEAUX tableaux (nx, ny).

    Les dérivées sont nulles sur les parois, donc seuls les nœuds intérieurs
    sont modifiés ; on réapplique ensuite la CL de vitesse sur les parois.
    """
    dpdx, dpdy = compute_pressure_gradient(p, dx, dy, par)

    u_new = u_2star - dt / par["rho"] * dpdx
    v_new = v_2star - dt / par["rho"] * dpdy

    apply_bc_velocity(u_new, v_new, t + dt, par)   # en place, sur nos tableaux
    return u_new, v_new


# ==========================================================================
# Fonction principale de l'étape 3
# ==========================================================================
def pressure_step(u_2star, v_2star, p, t, dt, dx, dy, par):
    """
    Étape 3 du pas fractionné : projection à divergence nulle.

    Arguments :
        u_2star, v_2star : vitesse après les étapes 1 et 2, (nx, ny)
        p                : pression du pas précédent, (nx, ny)
        t                : instant au début du pas de temps
        dt               : pas de temps
        dx, dy           : pas d'espace
        par              : dictionnaire des paramètres

    Retourne : (u_new, v_new, p_new), trois NOUVEAUX tableaux (nx, ny).
        Aucun argument n'est modifié.
    """
    # 1. Ce qui « ne va pas » dans la vitesse : sa divergence
    div = compute_divergence(u_2star, v_2star, dx, dy, par)

    # 2. La pression qui compense cette divergence (Poisson)
    p_new = solve_pressure(div, p, dt, dx, dy, par)

    # 3 + 4. Gradient de pression, correction de la vitesse, CL
    u_new, v_new = correct_velocity(u_2star, v_2star, p_new, t, dt, dx, dy, par)

    return u_new, v_new, p_new


# ==========================================================================
# Outil de diagnostic (pour le rapport : comparer les schémas)
# ==========================================================================
def projection_error(u, v, dx, dy, par):
    """
    Maximum de |div| mesuré avec le schéma de divergence de la pression.

    À utiliser après pressure_step pour vérifier que la projection a marché.
    Pour comparer des schémas de manière équitable, on peut aussi mesurer avec
    fonctions_diverses.max_divergence (divergence centrée, la même pour tous).

    Retourne : float.
    """
    return float(np.max(np.abs(compute_divergence(u, v, dx, dy, par))))