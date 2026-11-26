"""
campagne.py : campagne de tests du simulateur (plusieurs Re x maillages x pas de temps)
==========================================================================================

CE QUE FAIT CE SCRIPT
---------------------
1. GÉNÈRE un fichier de cas .yml par combinaison (Re, maillage, niveau de pas de temps),
   exactement au format lu par parametres.load_params (clés de INTERFACES section 2.6, sans pressure_scheme : voir BASE).
2. LANCE chaque cas avec main.main(config) et sauvegarde ses sorties dans
   <out-dir>/<nom du cas>/ (fields.npz, images, resultats.txt, resultats.json, log.txt).
   Un cas qui diverge ou plante n'arrête pas la campagne : il est noté dans le résumé.
3. RASSEMBLE tout dans <out-dir>/ :
       summary.csv / summary.txt  : un tableau, une ligne par cas
       profils_Re<Re>.png         : profils de u(x=Lx/2, y) de tous les cas, par Reynolds

COMMENT LE LANCER
-----------------
Depuis le même dossier que pour main.py (celui qui contient cases/ et results/),
dans l'environnement cfd :

    python src\\campagne.py --dry-run                    # liste les cas sans rien lancer
    python src\\campagne.py                              # preset "smoke" (quelques secondes)
    python src\\campagne.py --preset standard
    python src\\campagne.py --preset full --skip-existing   # reprend là où on s'est arrêté
    python src\\campagne.py --preset stress              # limites de stabilité
    python src\\campagne.py --summary-only               # refait seulement le résumé

Personnaliser (ces options REMPLACENT les listes du preset) :
    --re 100 1000   --nx 33 65 129   --levels ref div2   --t-end 20
Changer un paramètre de TOUS les cas (valeur lue comme en YAML) :
    --set advection_scheme=centered2 --out-dir results/campagne_c2
    (--set pressure_scheme=... demande que parametres.py connaisse cette clé : voir BASE)
Séparer le travail (par ex. un terminal par Reynolds) : lancer avec --re différents ;
le résumé reprend tous les cas déjà présents dans --out-dir.
Ctrl+C interrompt proprement : le résumé des cas terminés est quand même écrit.

LES TROIS AXES
--------------
- Re       : nombre de Reynolds (nu = 1/Re, car Lx = Ly = U0 = rho = 1).
- nx       : nombre de nœuds par direction (grille carrée nx x nx). Prendre nx IMPAIR
             pour que la ligne médiane x = Lx/2 tombe sur des nœuds.
- niveau de pas de temps (DT_LEVELS) : couple (cfl, fourier). Le pas de temps réel est
  fixé par fonctions_diverses.compute_dt. Ici dt_max est mis grand exprès pour que ce
  soient cfl et fourier qui décident. HYPOTHÈSE : compute_dt prend le plus petit de
  dt_max, cfl*dx/|u|max et fourier*dx^2/nu. La colonne dt_moy du résumé (= t_final / pas)
  permet de vérifier ce que compute_dt a vraiment fait.

CE QU'ON COMPARE
----------------
- u_min : minimum de u sur la ligne verticale médiane, grandeur classique de la cavité.
  Écart à Ghia et al. (1982) en % : les valeurs GHIA_U_MIN ci-dessous sont des valeurs
  de mémoire à VÉRIFIER contre l'article (ou contre fd.ghia_reference).
- ecart_ref : écart maximum de la ligne médiane de u avec le cas de RÉFÉRENCE du même
  Re (maillage le plus fin, puis pas de temps le plus petit, parmi les cas valides).
  Sert à voir la convergence en maillage et en temps sans dépendre d'une référence externe.
- temps de calcul total et temps de l'étape pression.
Statuts : ok = calcul terminé ; diverged = NaN/inf ; error = exception Python (voir log.txt).
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import yaml

import main
from poisson import optimal_omega


# ==========================================================================
# Réglages de la campagne (à modifier ici si besoin)
# ==========================================================================
# Paramètres communs à tous les cas (voir INTERFACES section 2.6).
BASE = {
    "Lx": 1.0, "Ly": 1.0, "rho": 1.0, "U0": 1.0,
    "ramp_time": 0.0,                    # démarrage brutal du couvercle
    "dt_max": 1.0,                       # grand : ce sont cfl et fourier qui fixent dt
    "time_scheme": "rk2_heun",
    "advection_scheme": "upwind1",
    "diffusion_method": "explicit",
    # "pressure_scheme" n'est PAS écrit : validate_params (parametres.py) ne la connaît pas
    # encore, et pression.py prend "centered2" par défaut. Pour comparer des schémas de
    # pression : ajouter la clé à la liste des clés connues de parametres.py, puis lancer
    # avec --set pressure_scheme=fwd_bwd.
    "poisson_solver": "sor",
    "tol": 1.0e-6, "maxiter": 5000,
    "steady_tol": 1.0e-6,
    "every_n_steps": 500,
    "snapshot_dt": 0.25,
    "snapshot_max": 1000
}

# Niveaux de pas de temps : nom -> (cfl, fourier). fourier <= 0.25 (sinon validate_params refuse).
DT_LEVELS = {
    "ref":     (0.4, 0.20),   # valeurs du .yml de INTERFACES
    "div2":    (0.2, 0.10),   # pas de temps ~2 fois plus petit
    "div4":    (0.1, 0.05),   # ~4 fois plus petit
    "plus":    (0.8, 0.25),   # plus grand : proche de la limite
    "extreme": (1.5, 0.25),   # CFL > 1 : on s'attend à une instabilité
}

# Durée simulée selon Re (assez longue pour approcher le régime stationnaire).
T_END_BY_RE = {100.0: 40.0, 400.0: 80.0, 1000.0: 150.0}
T_END_DEFAULT = 100.0

# Presets : listes d'axes ; t_end = None -> T_END_BY_RE, sinon durée fixe.
PRESETS = {
    "smoke":    {"re": [100.0],               "nx": [17, 33],      "levels": ["ref"],                    "t_end": 2.0},
    "standard": {"re": [100.0, 400.0, 1000.0], "nx": [33, 65],      "levels": ["ref", "div2"],           "t_end": None},
    "full":     {"re": [100.0, 400.0, 1000.0], "nx": [33, 65, 129], "levels": ["ref", "div2", "div4"],   "t_end": None},
    "stress":   {"re": [100.0, 1000.0, 5000.0], "nx": [33, 65],     "levels": ["ref", "plus", "extreme"], "t_end": 20.0},
}

# Minimum de u sur la ligne verticale médiane, Ghia et al. (1982), grille 129x129.
# VALEURS DE MÉMOIRE : à vérifier contre l'article avant de les citer dans le rapport.
GHIA_U_MIN = {100.0: -0.21090, 400.0: -0.32726, 1000.0: -0.38289}


# ==========================================================================
# 1. Génération des cas
# ==========================================================================
def case_name(re, nx, level):
    """Nom d'un cas, ex. 'Re1000_n65_dtref' (sert de nom de fichier et de dossier)."""
    return f"Re{re:g}_n{nx}_dt{level}"


def make_case_params(re, nx, level, t_end, out_dir, overrides):
    """
    Construit le dictionnaire de paramètres d'un cas (= contenu du .yml).

    Arguments :
        re        : nombre de Reynolds
        nx        : nombre de nœuds par direction (ny = nx)
        level     : clé de DT_LEVELS
        t_end     : instant final
        out_dir   : dossier de sortie du cas (relatif au dossier de lancement)
        overrides : dictionnaire {clé: valeur} appliqué en dernier (option --set)

    Retourne : dictionnaire contenant UNIQUEMENT des types Python natifs (YAML-safe).
    """
    cfl, fourier = DT_LEVELS[level]
    par = dict(BASE)
    dx = par["Lx"] / (nx - 1)
    par.update({
        "nx": int(nx), "ny": int(nx),
        "Re": float(re),
        "t_end": float(t_end),
        "cfl": float(cfl), "fourier": float(fourier),
        "omega": round(float(optimal_omega(nx, nx, dx, dx)), 4),   # omega adapté au maillage
        "out_dir": out_dir.replace(os.sep, "/"),
    })
    par.update(overrides)
    return par


def write_case_file(path, par):
    """Écrit le dictionnaire de paramètres dans un .yml (flottants écrits « 1.0e-06 », lisibles par YAML)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(par, f, sort_keys=False)


def build_case_list(re_list, nx_list, levels, t_end_fixed, cases_dir, out_dir, overrides):
    """
    Construit la liste des cas (un dictionnaire par cas), triée du moins cher au plus cher
    (petit maillage d'abord), et écrit les fichiers .yml.

    Retourne : liste de dict {name, Re, nx, level, cfg_path, out_dir, par}.
    """
    cases = []
    for re in re_list:
        for nx in nx_list:
            for level in levels:
                name = case_name(re, nx, level)
                t_end = t_end_fixed if t_end_fixed is not None else T_END_BY_RE.get(float(re), T_END_DEFAULT)
                case_out = os.path.join(out_dir, name)
                par = make_case_params(re, nx, level, t_end, case_out, overrides)
                cfg_path = os.path.join(cases_dir, name + ".yml")
                write_case_file(cfg_path, par)
                cases.append({"name": name, "Re": float(re), "nx": int(nx), "level": level,
                              "cfg_path": cfg_path, "out_dir": case_out, "par": par})
    cases.sort(key=lambda c: (c["nx"], c["Re"], -DT_LEVELS[c["level"]][0]))
    return cases


# ==========================================================================
# 2. Exécution d'un cas
# ==========================================================================
def centerline_profiles(u, v, nx, ny, lx, ly):
    """
    Profils sur les lignes médianes : u(x = Lx/2, y) et v(x, y = Ly/2).

    Arguments : u, v (nx, ny) avec [i, j] <-> (x[i], y[j]) ; nx, ny ; lx, ly.
    Si nx (ou ny) est pair, la ligne médiane tombe entre deux nœuds : on fait la moyenne.

    Retourne : (x, y, u_c, v_c) tableaux 1D de tailles (nx,), (ny,), (ny,), (nx,).
    """
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    if nx % 2 == 1:
        u_c = u[nx // 2, :]
    else:
        u_c = 0.5 * (u[nx // 2 - 1, :] + u[nx // 2, :])
    if ny % 2 == 1:
        v_c = v[:, ny // 2]
    else:
        v_c = 0.5 * (v[:, ny // 2 - 1] + v[:, ny // 2])
    return x, y, u_c, v_c


def run_case(case):
    """
    Lance un cas avec main.main, en capturant sa sortie console dans log.txt.

    Retourne : dictionnaire « record » (une ligne du résumé) et écrit
    <out_dir>/case_summary.json ainsi que <out_dir>/profils.npz.
    Toute exception est attrapée : status = "error" et le message est gardé.
    """
    par = case["par"]
    out_dir = case["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    record = {"name": case["name"], "Re": case["Re"], "nx": case["nx"], "level": case["level"],
              "cfl": par["cfl"], "fourier": par["fourier"], "t_end": par["t_end"],
              "out_dir": out_dir, "status": "error", "error": ""}
    log_path = os.path.join(out_dir, "log.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            u, v, p = main.main(case["cfg_path"])
        record.update(read_main_results(out_dir))
        finite = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.all(np.isfinite(p)))
        if record.get("diverged") or not finite:
            record["status"] = "diverged"
        else:
            record["status"] = "ok"
            x, y, u_c, v_c = centerline_profiles(u, v, par["nx"], par["ny"], par["Lx"], par["Ly"])
            np.savez(os.path.join(out_dir, "profils.npz"), x=x, y=y, u_c=u_c, v_c=v_c)
            record["u_min"] = float(np.min(u_c))
            record["y_u_min"] = float(y[int(np.argmin(u_c))])
    except Exception as err:                       # un cas qui plante ne doit pas arrêter la campagne
        record["status"] = "error"
        record["error"] = f"{type(err).__name__}: {err}"
        with open(log_path, "a", encoding="utf-8") as log:
            log.write("\n" + traceback.format_exc())
    with open(os.path.join(out_dir, "case_summary.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def read_main_results(out_dir):
    """
    Lit <out_dir>/resultats.json (écrit par main) et le met à plat.

    Retourne : dict avec n_steps, t_final, dt_moy, steady, diverged, res_steady, max_div,
    kinetic_energy, time_total, time_advection, time_diffusion, time_pression.
    Dictionnaire vide si le fichier n'existe pas.
    """
    path = os.path.join(out_dir, "resultats.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    timers = data.get("timers", {})
    return {
        "n_steps": data.get("n_steps"), "t_final": data.get("t_final"),
        "dt_moy": data.get("dt_mean"),
        "steady": data.get("steady"), "diverged": data.get("diverged"),
        "res_steady": data.get("res_steady"), "max_div": data.get("max_div"),
        "kinetic_energy": data.get("kinetic_energy"),
        "time_total": data.get("time_total"),
        "time_advection": timers.get("advection"), "time_diffusion": timers.get("diffusion"),
        "time_pression": timers.get("pression"),
    }


# ==========================================================================
# 3. Résumé et comparaison
# ==========================================================================
def load_all_records(out_dir):
    """Lit tous les <out_dir>/*/case_summary.json (cas de cette campagne ET des précédentes)."""
    records = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*", "case_summary.json"))):
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def add_comparisons(records):
    """
    Ajoute à chaque cas valide : err_ghia_pct (écart de u_min à Ghia, en %) et
    ecart_ref (écart max du profil médian de u avec le cas de référence du même Re).

    Référence d'un Re : maillage le plus fin, puis cfl le plus petit, parmi les cas ok.
    Les profils sont comparés sur les points du cas de référence (interpolation linéaire).
    """
    for rec in records:
        ghia = GHIA_U_MIN.get(rec["Re"])
        if rec["status"] == "ok" and ghia is not None:
            rec["err_ghia_pct"] = 100.0 * abs(rec["u_min"] - ghia) / abs(ghia)

    for re in sorted({r["Re"] for r in records}):
        valid = [r for r in records if r["Re"] == re and r["status"] == "ok"]
        if not valid:
            continue
        ref = max(valid, key=lambda r: (r["nx"], -r["cfl"]))
        ref_prof = np.load(os.path.join(ref["out_dir"], "profils.npz"))
        for rec in valid:
            prof = np.load(os.path.join(rec["out_dir"], "profils.npz"))
            u_on_ref = np.interp(ref_prof["y"], prof["y"], prof["u_c"])
            rec["ecart_ref"] = float(np.max(np.abs(u_on_ref - ref_prof["u_c"])))
            rec["is_ref"] = rec is ref


SUMMARY_COLUMNS = [
    ("name", "cas"), ("status", "statut"), ("Re", "Re"), ("nx", "nx"), ("level", "niveau_dt"),
    ("cfl", "cfl"), ("fourier", "fourier"), ("n_steps", "pas"), ("t_final", "t_final"),
    ("dt_moy", "dt_moy"), ("steady", "stationnaire"), ("res_steady", "res_steady"),
    ("max_div", "max_div"), ("u_min", "u_min"), ("err_ghia_pct", "err_ghia_%"),
    ("ecart_ref", "ecart_ref"), ("time_total", "temps_s"), ("time_pression", "temps_pression_s"),
    ("error", "erreur"),
]


def _fmt(value, key):
    """Formate une valeur pour le tableau texte."""
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, float):
        if key in ("err_ghia_%", "temps_s", "temps_pression_s"):
            return f"{value:.1f}"
        if key in ("u_min", "t_final"):
            return f"{value:.4f}"
        return f"{value:.3g}"
    return str(value)


def write_summary(records, out_dir):
    """Écrit summary.csv (toutes les colonnes) et summary.txt (tableau lisible) ; affiche le tableau."""
    records = sorted(records, key=lambda r: (r["Re"], r["nx"], -r["cfl"]))
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([label for _, label in SUMMARY_COLUMNS])
        for rec in records:
            writer.writerow([rec.get(key, "") for key, _ in SUMMARY_COLUMNS])

    shown = [(k, lab) for k, lab in SUMMARY_COLUMNS
             if lab in ("cas", "statut", "pas", "dt_moy", "stationnaire", "u_min",
                        "err_ghia_%", "ecart_ref", "temps_s")]
    rows = [[lab for _, lab in shown]]
    for rec in records:
        rows.append([_fmt(rec.get(k), lab) for k, lab in shown])
    widths = [max(len(r[i]) for r in rows) for i in range(len(shown))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows]
    errors = [f"  {r['name']} : {r['error']}" for r in records if r.get("error")]
    text = "\n".join(lines)
    if errors:
        text += "\n\nErreurs :\n" + "\n".join(errors)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nRésumé : {csv_path}")


def plot_profiles(records, out_dir):
    """Un graphique par Reynolds : u(x = Lx/2, y) de tous les cas valides superposés."""
    import matplotlib.pyplot as plt
    for re in sorted({r["Re"] for r in records}):
        valid = [r for r in records if r["Re"] == re and r["status"] == "ok"]
        if not valid:
            continue
        fig, ax = plt.subplots(figsize=(6, 7))
        for rec in sorted(valid, key=lambda r: (r["nx"], -r["cfl"])):
            prof = np.load(os.path.join(rec["out_dir"], "profils.npz"))
            ax.plot(prof["u_c"], prof["y"], label=f"n={rec['nx']}, dt {rec['level']}")
        ghia = GHIA_U_MIN.get(re)
        if ghia is not None:
            ax.axvline(ghia, color="black", ls="--", lw=0.8, label="u_min Ghia (à vérifier)")
        ax.set_xlabel("u(x = Lx/2, y)")
        ax.set_ylabel("y")
        ax.set_title(f"Re = {re:g} : profils sur la ligne médiane")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"profils_Re{re:g}.png"), dpi=120)
        plt.close(fig)


def build_summary(out_dir):
    """Rassemble tous les cas présents dans out_dir : comparaisons, tableau, graphiques."""
    records = load_all_records(out_dir)
    if not records:
        print("Aucun cas terminé dans", out_dir)
        return
    add_comparisons(records)
    write_summary(records, out_dir)
    try:
        plot_profiles(records, out_dir)
    except Exception as err:                       # les graphiques ne doivent pas bloquer le résumé
        print(f"Graphiques non produits : {type(err).__name__}: {err}")


# ==========================================================================
# 4. Ligne de commande
# ==========================================================================
def parse_overrides(items):
    """Transforme ['cle=valeur', ...] en dictionnaire ; la valeur est lue comme en YAML (1.0e-6, true, texte...)."""
    overrides = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"--set attend cle=valeur, reçu : {item}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def parse_args():
    parser = argparse.ArgumentParser(description="Campagne de tests : Re x maillages x pas de temps.")
    parser.add_argument("--preset", choices=list(PRESETS), default="smoke")
    parser.add_argument("--re", type=float, nargs="+", help="remplace la liste de Reynolds du preset")
    parser.add_argument("--nx", type=int, nargs="+", help="remplace la liste de maillages du preset (impairs conseillés)")
    parser.add_argument("--levels", nargs="+", choices=list(DT_LEVELS), help="remplace les niveaux de pas de temps")
    parser.add_argument("--t-end", type=float, help="durée simulée imposée à tous les cas")
    parser.add_argument("--set", action="append", metavar="CLE=VALEUR", help="change un paramètre de tous les cas")
    parser.add_argument("--cases-dir", default=os.path.join("cases", "campagne"), help="où écrire les .yml")
    parser.add_argument("--out-dir", default=os.path.join("results", "campagne"), help="où écrire les résultats")
    parser.add_argument("--dry-run", action="store_true", help="écrit les .yml et liste les cas, sans rien lancer")
    parser.add_argument("--skip-existing", action="store_true", help="ne relance pas les cas déjà terminés")
    parser.add_argument("--summary-only", action="store_true", help="refait seulement le résumé")
    return parser.parse_args()


def main_campaign():
    args = parse_args()
    if args.summary_only:
        build_summary(args.out_dir)
        return

    preset = PRESETS[args.preset]
    re_list = args.re or preset["re"]
    nx_list = args.nx or preset["nx"]
    levels = args.levels or preset["levels"]
    t_end = args.t_end if args.t_end is not None else preset["t_end"]
    overrides = parse_overrides(args.set)

    cases = build_case_list(re_list, nx_list, levels, t_end, args.cases_dir, args.out_dir, overrides)
    print(f"{len(cases)} cas (fichiers .yml écrits dans {args.cases_dir}) :")
    for c in cases:
        print(f"  {c['name']:<22} t_end = {c['par']['t_end']:g}  cfl = {c['par']['cfl']:g}  fourier = {c['par']['fourier']:g}")
    if args.dry_run:
        return

    t_campaign = time.perf_counter()
    try:
        for k, case in enumerate(cases, start=1):
            done_file = os.path.join(case["out_dir"], "case_summary.json")
            if args.skip_existing and os.path.exists(done_file):
                print(f"[{k}/{len(cases)}] {case['name']} : déjà fait, ignoré")
                continue
            print(f"[{k}/{len(cases)}] {case['name']} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            rec = run_case(case)
            detail = f"{rec['n_steps']} pas" if rec.get("n_steps") else rec.get("error", "")
            print(f"{rec['status']}  ({detail}, {time.perf_counter() - t0:.1f} s)", flush=True)
    except KeyboardInterrupt:
        print("\nInterrompu : résumé des cas déjà terminés.")
    finally:
        print(f"Durée de la campagne : {time.perf_counter() - t_campaign:.1f} s")
        build_summary(args.out_dir)


if __name__ == "__main__":
    main_campaign()