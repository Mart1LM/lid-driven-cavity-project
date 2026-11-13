"""
Tests de parametres, conditions_limite et fonctions_diverses.
Lancer :   python tests/test_initialisation.py      (ou : pytest tests)
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "stubs"))   # seulement pour schemas_espace manquant

import parametres
import conditions_limite as cl
import fonctions_diverses as fd

CASE = os.path.join(ROOT, "cases", "cavity_Re100.yaml")


def _par(**changes):
    par = parametres.load_params(CASE)
    par.update(changes)
    return par


def test_load_params():
    par = parametres.load_params(CASE)
    assert abs(par["nu"] - 0.01) < 1e-15
    assert isinstance(par["nx"], int) and isinstance(par["Re"], float)


def test_validation_errors():
    for bad in ({"Re": "1e2"}, {"fourier": 0.5}, {"time_scheme": "inconnu"}, {"nx": 2},
                {"cle_inconnue": 1}):
        par = _par(**bad)
        try:
            parametres.validate_params(par)
        except ValueError:
            continue
        raise AssertionError(f"aurait dû échouer : {bad}")
    par = _par()
    del par["Lx"]
    try:
        parametres.validate_params(par)
        raise AssertionError("clé manquante non détectée")
    except ValueError:
        pass


def test_grid_and_fields():
    par = _par(nx=11, ny=21)
    x, y, dx, dy = fd.make_grid(par)
    assert x.shape == (11,) and y.shape == (21,)
    assert x[0] == 0 and abs(x[-1] - par["Lx"]) < 1e-14 and abs(dx - 0.1) < 1e-14
    u, v, p = fd.init_fields(11, 21)
    assert u.shape == v.shape == p.shape == (11, 21) and not u.any()


def test_lid_velocity():
    assert cl.lid_velocity(0.0, _par(ramp_time=0.0)) == 1.0
    par = _par(ramp_time=2.0)
    assert cl.lid_velocity(0.0, par) == 0.0
    assert abs(cl.lid_velocity(1.0, par) - 0.5) < 1e-14
    assert cl.lid_velocity(5.0, par) == 1.0


def test_bc_velocity():
    par = _par()
    u, v = np.ones((9, 9)), np.ones((9, 9))
    cl.apply_bc_velocity(u, v, 0.0, par)
    assert (u[0, :-1] == 0).all() and (u[-1, :-1] == 0).all() and (u[:, 0] == 0).all()   # hors coins du haut
    assert (u[:, -1] == par["U0"]).all()                       # coins du haut compris
    assert (v[0, :] == 0).all() and (v[-1, :] == 0).all() and (v[:, 0] == 0).all() and (v[:, -1] == 0).all()
    assert (u[1:-1, 1:-1] == 1).all()                          # intérieur inchangé


def test_bc_pressure():
    p = np.arange(36, dtype=float).reshape(6, 6)
    cl.apply_bc_pressure(p)
    assert (p[0, 1:-1] == p[1, 1:-1]).all() and (p[-1, 1:-1] == p[-2, 1:-1]).all()
    assert (p[1:-1, 0] == p[1:-1, 1]).all() and (p[1:-1, -1] == p[1:-1, -2]).all()


def test_compute_dt():
    par = _par()
    x, y, dx, dy = fd.make_grid(par)
    u, v, p = fd.init_fields(par["nx"], par["ny"])
    dt = fd.compute_dt(u, v, dx, dy, par)
    assert 0 < dt <= par["dt_max"]
    par_big_nu = _par(Re=1.0, nu=1.0)                                 # grande viscosité : diffusion limitante
    assert fd.compute_dt(u, v, dx, dy, par_big_nu) < dt
    par_adi = dict(par_big_nu, diffusion_method="adi")         # implicite : plus de contrainte de Fourier
    assert fd.compute_dt(u, v, dx, dy, par_adi) >= fd.compute_dt(u, v, dx, dy, par_big_nu)


def test_diagnostics():
    u = np.full((5, 5), 2.0)
    v = np.zeros((5, 5))
    assert abs(fd.kinetic_energy(u, v, 0.25, 0.25) - 0.5 * 25 * 4 * 0.0625) < 1e-12
    assert abs(fd.steady_residual(u, u + 0.1, 0.5) - 0.2) < 1e-12
    x = np.linspace(0, 1, 5)
    u = np.tile(x[:, None], (1, 5)) * 1.0                      # u = x -> divergence = 1
    assert abs(fd.max_divergence(u, np.zeros((5, 5)), 0.25, 0.25) - 1.0) < 1e-12


def test_centerline():
    for n in (9, 10):                                          # nombre impair et pair
        x = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, x, indexing="ij")
        yc, uc, xc, vc = fd.centerline_profiles(X, Y, x, x)    # u = x, v = y
        assert np.allclose(uc, 0.5) and np.allclose(vc, 0.5)


def test_save_load():
    u, v, p = np.random.rand(4, 4), np.random.rand(4, 4), np.random.rand(4, 4)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "f.npz")
        fd.save_fields(path, u, v, p, 1.5, 7)
        u2, v2, p2, t, n = fd.load_fields(path)
    assert np.array_equal(u, u2) and np.array_equal(p, p2) and t == 1.5 and n == 7


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print(f"OK  {f.__name__}")
    print(f"{len(tests)} tests réussis.")
