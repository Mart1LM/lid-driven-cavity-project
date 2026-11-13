"""
advection.py : ÉTAPE 1 du pas fractionné (transport)
=====================================================

CE QUE FAIT CE FICHIER
----------------------
Il fait avancer de dt l'équation de transport (sans pression, sans viscosité) :

    du/dt + u du/dx + v du/dy = 0        et la même chose pour v

et renvoie (u_star, v_star). C'est un fichier d'ORCHESTRATION : il ne contient
aucune formule de différences finies ni de schéma en temps, il les assemble.

    schemas_espace.convective_term      -> « u dphi/dx + v dphi/dy »   (espace)
    schemas_temps_explicite.<schéma>    -> euler, rk2_heun...           (temps)
    conditions_limite.apply_bc_velocity -> vitesse sur les parois       (CL)

MÉTHODE DES LIGNES (ce qu'on fait ici)
--------------------------------------
On écrit l'équation sous la forme   dq/dt = F(q)   avec q = (u, v) et
F(q) = -convective_term. Le schéma d'espace est choisi par
par["advection_scheme"] (upwind1, centered2...), le schéma en temps par
par["time_scheme"] (euler, rk2_heun...). Les deux choix sont INDÉPENDANTS.

    advection_step :  q  --(schéma en temps appliqué à F_advection)-->  q*  --(CL)-->  (u_star, v_star)

SCHÉMAS QUI MÉLANGENT ESPACE ET TEMPS (Lax-Wendroff...)
-------------------------------------------------------
Ils ne se découpent pas en « F + schéma en temps » (INTERFACES, section 8).
Pour en ajouter un : écrire une fonction avec la signature de advection_step
et l'enregistrer dans COUPLED_ADVECTION_STEPS sous le nom utilisé dans le .yml
(advection_scheme: lax_wendroff). advection_step l'utilisera automatiquement,
sans toucher au reste.

CONVENTIONS (voir INTERFACES.md)
--------------------------------
- Tableaux (nx, ny), [i, j] <-> point (x[i], y[j]).
- Les dérivées de schemas_espace sont nulles sur les parois : seuls les nœuds
  intérieurs avancent. Les parois reçoivent leur valeur de la CL.
- advection_step ne modifie pas ses arguments : elle renvoie de nouveaux tableaux.
- La CL de vitesse est appliquée à l'instant t + dt (fin du pas), comme dans
  pression.py. t est l'instant au DÉBUT du pas.
"""

from schemas_temps_explicite import get_time_scheme
from conditions_limite import apply_bc_velocity
from schemas_espace import SPACE_SCHEMES

# Schémas d'advection espace+temps couplés : nom -> fonction ayant la signature
#   f(u, v, t, dt, dx, dy, par) -> (u_star, v_star)
# Vide pour l'instant (exemple futur : "lax_wendroff": advection_lax_wendroff).
COUPLED_ADVECTION_STEPS = {}


def convective_term(phi, u, v, dx, dy, scheme):
    """
    Calcule u·dphi/dx + v·dphi/dy avec le schema choisi par son nom.

    Parameters
    ----------
    phi : numpy.ndarray, shape (nx, ny)
        Champ transporte (u ou v).
    u, v : numpy.ndarray, shape (nx, ny)
        Composantes de la vitesse d'advection.
    dx, dy : float
        Pas d'espace en x et en y.
    scheme : str
        Nom du schema, cle de SPACE_SCHEMES (par["advection_scheme"]).

    Returns
    -------
    numpy.ndarray, shape (nx, ny)
        Terme convectif, nul sur les nœuds de paroi.

    Raises
    ------
    ValueError
        Si le nom du schema est inconnu.
    """
    if scheme not in SPACE_SCHEMES:
        raise ValueError(f"Schema d'espace inconnu : {scheme}")
    return SPACE_SCHEMES[scheme](phi, u, v, dx, dy)
def F_advection(q, dx, dy, par):
    """
    Second membre de l'équation de transport : dq/dt = F(q) = -(u.grad)q.

    Arguments :
        q      : tuple (u, v), deux tableaux (nx, ny)
        dx, dy : pas d'espace
        par    : dictionnaire des paramètres (utilise par["advection_scheme"])

    Retourne : (du_dt, dv_dt), deux tableaux (nx, ny), nuls sur les parois.
    """
    u, v = q
    scheme = par["advection_scheme"]
    du_dt = -convective_term(u, u, v, dx, dy, scheme)   # transport de u
    dv_dt = -convective_term(v, u, v, dx, dy, scheme)   # transport de v
    return du_dt, dv_dt


def advection_method_of_lines(u, v, t, dt, dx, dy, par):
    """
    Avance de dt par « méthode des lignes » : schéma en temps appliqué à F_advection,
    puis CL de vitesse. Mêmes arguments et retour que advection_step.
    """
    time_step = get_time_scheme(par["time_scheme"])

    def F(q):
        return F_advection(q, dx, dy, par)

    u_star, v_star = time_step((u, v), F, dt)
    apply_bc_velocity(u_star, v_star, t + dt, par)      # en place, sur nos tableaux
    return u_star, v_star


def advection_step(u, v, t, dt, dx, dy, par):
    """
    Étape 1 du pas fractionné : avance de dt l'équation
        du/dt + u du/dx + v du/dy = 0   (idem pour v).

    Arguments :
        u, v   : vitesse au temps n, tableaux (nx, ny)
        t      : instant au début du pas
        dt     : pas de temps
        dx, dy : pas d'espace
        par    : dictionnaire des paramètres (advection_scheme, time_scheme...)

    Retourne : (u_star, v_star), NOUVEAUX tableaux (nx, ny). u et v ne sont pas modifiés.

    Si par["advection_scheme"] est dans COUPLED_ADVECTION_STEPS, on utilise cette
    fonction à la place de la méthode des lignes.
    """
    scheme = par["advection_scheme"]
    if scheme in COUPLED_ADVECTION_STEPS:
        return COUPLED_ADVECTION_STEPS[scheme](u, v, t, dt, dx, dy, par)
    return advection_method_of_lines(u, v, t, dt, dx, dy, par)