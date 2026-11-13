"""
diffusion.py : ÉTAPE 2 du pas fractionné (diffusion visqueuse)
===============================================================

CE QUE FAIT CE FICHIER
----------------------
Il fait avancer de dt l'équation de diffusion de la quantité de mouvement :

    du/dt = nu * laplacien(u)        et la même chose pour v

à partir de (u_star, v_star) (sortie de l'étape 1) et renvoie (u_2star, v_2star).
Comme advection.py, c'est un fichier d'ORCHESTRATION : les formules sont dans
schemas_espace (laplacian) et schemas_temps_explicite (euler, rk2_heun...).

    diffusion_step :  choisit la méthode selon par["diffusion_method"]
        "explicit" -> diffusion_explicit  (schéma en temps explicite appliqué à F_diffusion)
        "adi"      -> (plus tard, voir INTERFACES section 7)

POURQUOI UN DICTIONNAIRE DE MÉTHODES
------------------------------------
diffusion_step garde TOUJOURS la même signature, que la méthode soit explicite
ou implicite : main, advection et pression n'ont donc jamais à changer.
Pour ajouter l'ADI : écrire diffusion_adi (dans schemas_implicites.py ou ici)
avec la signature de diffusion_step, puis ajouter UNE ligne à DIFFUSION_METHODS.

STABILITÉ
---------
En explicite, il faut dt <= fourier * dx^2 / nu (fourier <= 1/4 en 2D).
C'est compute_dt (fonctions_diverses) qui s'en occupe ; ici on ne vérifie rien.

CONVENTIONS (voir INTERFACES.md)
--------------------------------
- Tableaux (nx, ny), [i, j] <-> point (x[i], y[j]).
- Le laplacien est nul sur les parois : seuls les nœuds intérieurs avancent.
- diffusion_step ne modifie pas ses arguments : elle renvoie de nouveaux tableaux.
- La CL de vitesse est appliquée à l'instant t + dt (fin du pas), comme dans
  advection.py et pression.py. t est l'instant au DÉBUT du pas.
"""
from schemas_espace import laplacian
from schemas_temps_explicite import get_time_scheme
from conditions_limite import apply_bc_velocity


def F_diffusion(q, dx, dy, par):
    """
    Second membre de l'équation de diffusion : dq/dt = F(q) = nu * laplacien(q).

    Arguments :
        q      : tuple (u, v), deux tableaux (nx, ny)
        dx, dy : pas d'espace
        par    : dictionnaire des paramètres (utilise par["nu"])

    Retourne : (du_dt, dv_dt), deux tableaux (nx, ny), nuls sur les parois.
    """
    u, v = q
    nu = par["nu"]
    du_dt = nu * laplacian(u, dx, dy)
    dv_dt = nu * laplacian(v, dx, dy)
    return du_dt, dv_dt


def diffusion_explicit(u_star, v_star, t, dt, dx, dy, par):
    """
    Diffusion explicite : schéma en temps (par["time_scheme"]) appliqué à
    F_diffusion, puis CL de vitesse. Mêmes arguments et retour que diffusion_step.
    """
    time_step = get_time_scheme(par["time_scheme"])

    def F(q):
        return F_diffusion(q, dx, dy, par)

    u_2star, v_2star = time_step((u_star, v_star), F, dt)
    apply_bc_velocity(u_2star, v_2star, t + dt, par)    # en place, sur nos tableaux
    return u_2star, v_2star


# Correspondance « nom dans le .yml » -> fonction.
# Plus tard : "adi": diffusion_adi   (voir schemas_implicites.py)
DIFFUSION_METHODS = {
    "explicit": diffusion_explicit,
}


def diffusion_step(u_star, v_star, t, dt, dx, dy, par):
    """
    Étape 2 du pas fractionné : avance de dt l'équation
        du/dt = nu * laplacien(u)   (idem pour v).

    Arguments :
        u_star, v_star : vitesse après l'étape 1, tableaux (nx, ny)
        t              : instant au début du pas
        dt             : pas de temps
        dx, dy         : pas d'espace
        par            : dictionnaire des paramètres (diffusion_method, nu, time_scheme...)

    Retourne : (u_2star, v_2star), NOUVEAUX tableaux (nx, ny).
        u_star et v_star ne sont pas modifiés.

    Lève ValueError si par["diffusion_method"] est inconnue.
    """
    name = par["diffusion_method"]
    if name not in DIFFUSION_METHODS:
        raise ValueError(
            f"diffusion_method inconnue : '{name}'. "
            f"Choix possibles : {list(DIFFUSION_METHODS)}"
        )
    return DIFFUSION_METHODS[name](u_star, v_star, t, dt, dx, dy, par)