"""
fonctions_diverses.py : boîte à outils (rien de la physique de l'écoulement ici).
 
Grille / champs : make_grid, init_fields
Pas de temps    : compute_dt
Diagnostics     : kinetic_energy, max_divergence, steady_residual
Validation      : ghia_reference, centerline_profiles
Tracés          : plot_velocity, plot_streamlines, plot_pressure, plot_history,
                  plot_ghia_comparison
Sauvegardes     : save_fields, load_fields
 
Convention : tous les champs ont la forme (nx, ny) et [i, j] <-> point (x[i], y[j]).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
 
 
# ----------------------------------------------------------------- grille et champs
def make_grid(par: dict):
    """Maillage uniforme, nœuds sur les parois : x[i] = i*dx, dx = Lx/(nx-1).
 
    Retourne (x, y, dx, dy) avec x de forme (nx,) et y de forme (ny,).
    """
    x = np.linspace(0.0, par["Lx"], par["nx"])
    y = np.linspace(0.0, par["Ly"], par["ny"])
    return x, y, x[1] - x[0], y[1] - y[0]
 
 
def init_fields(nx: int, ny: int):
    """Fluide au repos : retourne trois tableaux (u, v, p) de zéros, de forme (nx, ny)."""
    return np.zeros((nx, ny)), np.zeros((nx, ny)), np.zeros((nx, ny))
 
 
# ----------------------------------------------------------------- pas de temps
def compute_dt(u: np.ndarray, v: np.ndarray, dx: float, dy: float, par: dict) -> float:
    """Plus grand pas de temps admissible (coefficients de sécurité cfl et fourier).
 
        dt_adv = cfl * min(dx / |u|max, dy / |v|max)        (toujours)
        dt_dif = fourier * min(dx, dy)^2 / nu               (seulement si diffusion explicite)
        dt     = min(dt_max, dt_adv, dt_dif)
 
    |u|max et |v|max sont pris au moins égaux à U0 : sinon dt serait infini à t = 0
    (fluide au repos) puis chuterait brutalement.
    """
    umax = max(np.abs(u).max(), par["U0"])
    vmax = max(np.abs(v).max(), par["U0"])
    dt = min(par["dt_max"], par["cfl"] * min(dx / umax, dy / vmax))
    if par["diffusion_method"] == "explicit":
        dt = min(dt, par["fourier"] * min(dx, dy) ** 2 / par["nu"])
    return float(dt)
 
 
# ----------------------------------------------------------------- diagnostics
def kinetic_energy(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> float:
    """Énergie cinétique par unité de masse : 1/2 * somme(u^2 + v^2) * dx * dy."""
    return float(0.5 * np.sum(u ** 2 + v ** 2) * dx * dy)
 
 
def max_divergence(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> float:
    """Max de |div u| sur les nœuds intérieurs : mesure la qualité de la projection.
    Utilise schemas_espace.divergence (fichier d'un coéquipier)."""
    from schemas_espace import divergence
    d = divergence(u, v, dx, dy)
    return float(np.abs(d[1:-1, 1:-1]).max())
 
 
def steady_residual(u: np.ndarray, u_new: np.ndarray, dt: float) -> float:
    """res_steady = max|u_new - u| / dt : tend vers 0 quand l'écoulement devient stationnaire."""
    return float(np.abs(u_new - u).max() / dt)
 
 
# ----------------------------------------------------------------- validation (Ghia)
def ghia_reference(Re: float, data_dir: str = "data") -> dict:
    """Lit les points de référence de Ghia, Ghia & Shin (1982).
 
    Fichiers attendus (à créer en recopiant les tableaux de l'article, 1 ligne d'en-tête) :
        data/ghia_u_Re100.csv   colonnes : y,u   (u sur la verticale x = Lx/2)
        data/ghia_v_Re100.csv   colonnes : x,v   (v sur l'horizontale y = Ly/2)
    Retourne {'y': ..., 'u': ..., 'x': ..., 'v': ...}. Lève FileNotFoundError si absents.
    """
    tag = f"Re{int(Re)}"
    fu = os.path.join(data_dir, f"ghia_u_{tag}.csv")
    fv = os.path.join(data_dir, f"ghia_v_{tag}.csv")
    for f in (fu, fv):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Données de Ghia absentes : {f} (voir la docstring)")
    y, u = np.loadtxt(fu, delimiter=",", skiprows=1, unpack=True)
    x, v = np.loadtxt(fv, delimiter=",", skiprows=1, unpack=True)
    return {"y": y, "u": u, "x": x, "v": v}
 
 
def centerline_profiles(u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Profils sur les lignes médianes, par interpolation linéaire entre nœuds.
 
    Retourne (y, u_c, x, v_c) avec u_c = u(x = Lx/2, y) et v_c = v(x, y = Ly/2).
    """
    def mid(field, axis):
        n = field.shape[axis]
        pos = 0.5 * (n - 1)
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, n - 1)
        w = pos - i0
        return (1 - w) * np.take(field, i0, axis=axis) + w * np.take(field, i1, axis=axis)
    return y, mid(u, 0), x, mid(v, 1)
 
 
# ----------------------------------------------------------------- tracés
def plot_velocity(u, v, x, y, path: str) -> None:
    """Carte de la norme de la vitesse + flèches (sous-échantillonnées)."""
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    speed = np.sqrt(u ** 2 + v ** 2)
    im = ax.imshow(speed.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="viridis")
    s = max(1, len(x) // 20)
    X, Y = np.meshgrid(x, y, indexing="ij")
    ax.quiver(X[::s, ::s], Y[::s, ::s], u[::s, ::s], v[::s, ::s], color="white", scale_units="xy")
    fig.colorbar(im, ax=ax, label="|vitesse| (m/s)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def plot_streamlines(u, v, x, y, path: str) -> None:
    """Lignes de courant (streamplot attend des tableaux [y, x] : d'où les transposées)."""
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.streamplot(x, y, u.T, v.T, color="tab:blue", density=1.5, linewidth=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_aspect("equal")
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def plot_pressure(p, x, y, path: str) -> None:
    """Carte de pression."""
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    cs = ax.contourf(x, y, p.T, levels=30, cmap="coolwarm")
    fig.colorbar(cs, ax=ax, label="p (Pa)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def plot_history(history: dict, path: str) -> None:
    """Évolution de res_steady, max|div| et de l'énergie cinétique.
 
    history : dict de listes avec les clés 't', 'res_steady', 'div_max', 'ke'.
    """
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, constrained_layout=True)
    t = np.asarray(history["t"])
    floor = 1e-300                                    # évite log(0)
    axes[0].semilogy(t, np.maximum(history["res_steady"], floor))
    axes[0].set_ylabel("res_steady")
    axes[1].semilogy(t, np.maximum(history["div_max"], floor))
    axes[1].set_ylabel("max |div u|")
    axes[2].plot(t, history["ke"])
    axes[2].set_ylabel("énergie cinétique")
    axes[2].set_xlabel("t (s)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def plot_ghia_comparison(u, v, x, y, Re: float, path: str) -> None:
    """Profils médians calculés contre points de Ghia. Lève FileNotFoundError si pas de données."""
    ref = ghia_reference(Re)
    yc, uc, xc, vc = centerline_profiles(u, v, x, y)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ax1.plot(uc, yc, label="calcul")
    ax1.plot(ref["u"], ref["y"], "ro", label="Ghia et al. (1982)")
    ax1.set_xlabel("u (x = Lx/2)")
    ax1.set_ylabel("y")
    ax2.plot(xc, vc, label="calcul")
    ax2.plot(ref["x"], ref["v"], "ro", label="Ghia et al. (1982)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("v (y = Ly/2)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Re = {Re:g}")
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
# ----------------------------------------------------------------- sauvegardes
def save_fields(path: str, u, v, p, t: float, n: int) -> None:
    """Sauvegarde les champs et l'instant dans un fichier .npz."""
    np.savez(path, u=u, v=v, p=p, t=t, n=n)
 
 
def load_fields(path: str):
    """Relit un fichier écrit par save_fields. Retourne (u, v, p, t, n)."""
    d = np.load(path)
    return d["u"], d["v"], d["p"], float(d["t"]), int(d["n"])
 