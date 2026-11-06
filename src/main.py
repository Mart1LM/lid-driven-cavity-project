"""
main.py : chef d'orchestre. Appelle les étapes dans l'ordre, rien d'autre.
 
Usage :   python main.py cases/cavity_Re100.yml
 
Un pas de temps (pas fractionné) :
    (u, v) --advection_step--> (u_star, v_star)        étape 1 : transport
           --diffusion_step--> (u_2star, v_2star)      étape 2 : diffusion
           --pressure_step---> (u_new, v_new, p)       étape 3 : pression
"""
import os
import sys
 
import parametres
import conditions_limite as cl
import fonctions_diverses as fd
import advection
import diffusion
import pression
 
 
def advance_one_step(u, v, p, t, dt, dx, dy, par):
    """Avance l'écoulement de t à t + dt. N'altère pas u, v, p.
 
    Retourne (u_new, v_new, p_new).
    """
    u_star, v_star = advection.advection_step(u, v, t, dt, dx, dy, par)
    u_2star, v_2star = diffusion.diffusion_step(u_star, v_star, t, dt, dx, dy, par)
    return pression.pressure_step(u_2star, v_2star, p, t, dt, dx, dy, par)
 
 
def main(config_path: str):
    """Charge le cas, boucle en temps jusqu'à t_end (ou régime stationnaire), écrit les sorties."""
    par = parametres.load_params(config_path)
    x, y, dx, dy = fd.make_grid(par)
    u, v, p = fd.init_fields(par["nx"], par["ny"])
    t, n = 0.0, 0
    cl.apply_bc_velocity(u, v, t, par)           # le couvercle est en place dès t = 0
    os.makedirs(par["out_dir"], exist_ok=True)
 
    history = {"t": [], "res_steady": [], "div_max": [], "ke": []}
    print(f"Cas : Re = {par['Re']:g}, grille {par['nx']}x{par['ny']}, "
          f"dx = {dx:.3e}, nu = {par['nu']:.3e}")
 
    while t < par["t_end"] - 1e-12:
        dt = min(fd.compute_dt(u, v, dx, dy, par), par["t_end"] - t)
        u_new, v_new, p_new = advance_one_step(u, v, p, t, dt, dx, dy, par)
 
        res_steady = fd.steady_residual(u, u_new, dt)
        u, v, p = u_new, v_new, p_new
        t += dt
        n += 1
 
        if n % par["every_n_steps"] == 0:
            history["t"].append(t)
            history["res_steady"].append(res_steady)
            history["div_max"].append(fd.max_divergence(u, v, dx, dy))
            history["ke"].append(fd.kinetic_energy(u, v, dx, dy))
            print(f"n = {n:6d}  t = {t:8.3f}  dt = {dt:.2e}  res_steady = {res_steady:.2e}  "
                  f"max|div| = {history['div_max'][-1]:.2e}")
        if res_steady < par["steady_tol"]:        # steady_tol = 0 : critère désactivé
            print("Régime stationnaire atteint.")
            break
 
    out = par["out_dir"]
    fd.save_fields(os.path.join(out, "fields.npz"), u, v, p, t, n)
    fd.plot_velocity(u, v, x, y, os.path.join(out, "velocity.png"))
    fd.plot_streamlines(u, v, x, y, os.path.join(out, "streamlines.png"))
    fd.plot_pressure(p, x, y, os.path.join(out, "pressure.png"))
    if history["t"]:
        fd.plot_history(history, os.path.join(out, "history.png"))
    try:
        fd.plot_ghia_comparison(u, v, x, y, par["Re"], os.path.join(out, "ghia.png"))
    except FileNotFoundError as err:
        print(f"Comparaison Ghia ignorée : {err}")
    return u, v, p
 
 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage : python main.py cases/<cas>.yml")
    main(sys.argv[1])
 