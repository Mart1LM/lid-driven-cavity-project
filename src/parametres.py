"""
parametres.py : lecture et validation des paramètres d'un cas (fichier .yml).

Fonctions publiques
-------------------
load_params(path)   -> par    dictionnaire de tous les paramètres (+ par["nu"] calculé)
validate_params(par) -> None  lève ValueError si une clé manque, est inconnue, a le mauvais
                              type, ou une valeur incohérente

Quand un coéquipier ajoute un schéma, il ajoute son nom dans ALLOWED_NAMES ci-dessous.
"""
import yaml

# Clé du .yml -> type attendu (voir INTERFACES_simple.md, section 2.6)
EXPECTED = {
    "Lx": float, "Ly": float, "nx": int, "ny": int,
    "rho": float, "Re": float, "U0": float, "ramp_time": float,
    "t_end": float, "cfl": float, "fourier": float, "dt_max": float,
    "time_scheme": str, "advection_scheme": str, "diffusion_method": str,
    "poisson_solver": str, "tol": float, "maxiter": int, "omega": float,
    "steady_tol": float, "every_n_steps": int, "out_dir": str,
}

# Noms de méthodes autorisés (à compléter au fil du projet)
ALLOWED_NAMES = {
    "time_scheme": ["euler", "rk2_heun"],
    "advection_scheme": ["upwind1", "centered2"],
    "diffusion_method": ["explicit"],          # plus tard : "adi"
    "poisson_solver": ["jacobi", "sor"],
}


def load_params(path: str) -> dict:
    """Lit le .yml, le valide, convertit les entiers en flottants si besoin et ajoute
    par["nu"] = U0 * Lx / Re (viscosité cinématique).

    Arguments : path (str) : chemin du fichier .yml
    Retourne  : par (dict)
    """
    with open(path, "r", encoding="utf-8") as f:
        par = yaml.safe_load(f)
    if not isinstance(par, dict):
        raise ValueError(f"{path} : le fichier doit contenir des lignes 'clé: valeur'")
    validate_params(par)
    for key, typ in EXPECTED.items():
        if typ is float:
            par[key] = float(par[key])        # 1 devient 1.0
    par["nu"] = par["U0"] * par["Lx"] / par["Re"]
    return par


def validate_params(par: dict) -> None:
    """Vérifie clés, types et valeurs. Ne modifie pas par."""
    missing = [k for k in EXPECTED if k not in par]
    if missing:
        raise ValueError(f"Paramètres manquants dans le .yml : {missing}")
    unknown = [k for k in par if k not in EXPECTED and k != "nu"]
    if unknown:
        raise ValueError(f"Paramètres inconnus (faute de frappe ?) : {unknown}")

    for key, typ in EXPECTED.items():
        val = par[key]
        is_bool = isinstance(val, bool)
        if typ is str:
            ok = isinstance(val, str)
        elif typ is int:
            ok = isinstance(val, int) and not is_bool
        else:
            ok = isinstance(val, (int, float)) and not is_bool
        if not ok:
            hint = " (YAML : écrire 1.0e-2 et non 1e-2)" if typ is float and isinstance(val, str) else ""
            raise ValueError(f"'{key}' doit être de type {typ.__name__}, reçu {val!r}{hint}")

    for key, allowed in ALLOWED_NAMES.items():
        if par[key] not in allowed:
            raise ValueError(f"'{key}' = {par[key]!r} inconnu. Choix possibles : {allowed}")

    positive = ["Lx", "Ly", "rho", "Re", "U0", "t_end", "cfl", "fourier", "dt_max",
                "tol", "maxiter", "every_n_steps"]
    for key in positive:
        if par[key] <= 0:
            raise ValueError(f"'{key}' doit être > 0")
    if par["nx"] < 4 or par["ny"] < 4:
        raise ValueError("nx et ny doivent être >= 4")
    if par["ramp_time"] < 0 or par["steady_tol"] < 0:
        raise ValueError("'ramp_time' et 'steady_tol' doivent être >= 0 (steady_tol = 0 : désactivé)")
    if par["fourier"] > 0.25:
        raise ValueError("'fourier' > 0.25 : instable pour la diffusion explicite en 2D")
    if not 0.0 < par["omega"] < 2.0:
        raise ValueError("'omega' doit être dans ]0, 2[")
