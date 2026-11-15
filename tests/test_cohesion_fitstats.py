"""Tests: cohesion (Sanchez & Scheeres 2014), effective parameters, spin-state naming.

    PYTHONPATH=. python tests/test_cohesion_fitstats.py
"""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from lcinv import cohesion as co, fitstats  # noqa: E402
from lcinv.hybrid import naming  # noqa: E402


def test_cohesion():
    G, rho = co.G, 2000.0
    # sphere: omega_alpha^2 = 4/3 pi G rho, the classic spin limit P = sqrt(3 pi / (G rho))
    wa2 = co.omega_alpha2(500, 500, 500, rho)
    assert np.isclose(wa2, 4 / 3 * np.pi * G * rho)
    Pc = np.sqrt(3 * np.pi / (G * rho)) / 3600
    assert co.required_cohesion(500, 500, 500, 1.01 * Pc, rho) == 0.0          # below the limit
    s = co.required_cohesion(500, 500, 500, 0.5 * Pc, rho)
    w2 = (2 * np.pi / (0.5 * Pc * 3600)) ** 2
    assert np.isclose(s, rho * 500 ** 2 * (w2 - wa2))                        # eq. 24 (phi = 90 deg)
    # friction angle: eq. 23 factor (3 + sin phi) / (2 (3 - sin phi)), 1/2 at phi = 0
    assert np.isclose(co.required_cohesion(500, 500, 500, 0.5 * Pc, rho, phi_deg=0.0), 0.5 * s)
    # strength-dominated regime: sigma ~ rho alpha^2 omega^2 for a 10 m body spinning in 1 minute
    a = co.required_cohesion(5.0, 4.0, 3.0, 1 / 60, rho)
    assert np.isclose(a, rho * 25 * (2 * np.pi / 60) ** 2, rtol=1e-3)
    # JAX version agrees, incl. Carlson R_D
    from scipy.special import elliprd
    assert np.isclose(float(co._rd_jax(0.3, 1.7, 2.2)), elliprd(0.3, 1.7, 2.2), rtol=1e-12)
    assert np.isclose(float(co.required_cohesion_jax(800., 500., 400., 0.7, rho)),
                      co.required_cohesion(800., 500., 400., 0.7, rho, clip=False), rtol=1e-10)
    # axis ratios and moments -> semi-axes with the right volume
    A = co.semi_axes_from_ratios(1.0, 1.5, 1.2)
    assert np.isclose(np.prod(A), 500.0 ** 3) and np.isclose(A[0] / A[1], 1.5) and np.isclose(A[1] / A[2], 1.2)
    a_, b_, c_ = 3.0, 2.0, 1.5
    I = [(b_ ** 2 + c_ ** 2) / 5, (a_ ** 2 + c_ ** 2) / 5, (a_ ** 2 + b_ ** 2) / 5]
    ax = co.semi_axes_from_moments(I, 1.0)
    assert np.allclose(np.array(ax) / ax[0], np.array([a_, b_, c_]) / a_)
    # penalty: zero below the limit, grows above
    assert co.penalty(20.0, 25.0) == 0.0 and co.penalty(50.0, 25.0) > 10
    print("cohesion ok")


def test_effective_parameters():
    # ridge regression: p_eff = sum d_i / (d_i + lambda) in the eigenbasis
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 6)) * np.array([10, 5, 2, 1, 0.3, 0.1])
    Hl = X.T @ X
    lam = 4.0
    d = np.linalg.eigvalsh(Hl)
    p = fitstats.effective_block(Hl, Hl + lam * np.eye(6))
    assert np.isclose(p, np.sum(d / (d + lam)), rtol=1e-8), p
    q = fitstats.quality(np.full(10, 0.02), np.full(10, 0.01), 0.0, 2.0, 3, 1, 1)
    assert np.isclose(q["chi2_red_nominal"], 40 / 8)
    print("fitstats ok")


def test_naming():
    assert naming.name(2) == "spin state 2" and naming.tag(3) == "spin_state_3"
    assert naming.index_of("mode 4") == 4 and naming.index_of("posterior_mode12") == 12
    assert naming.tag_of("spin state 1") == "spin_state_1" and naming.tag_of("mode 2") == "spin_state_2"
    s = {"modes": [{"name": "mode 1", "w": 1}, {"name": "mode 2", "w": 2}]}
    assert naming.state_info(s, 2)["w"] == 2
    print("naming ok")


def test_size():
    """CNEOS size estimator and the SBDB lookup (against a local fake server)."""
    import json as _json
    import os
    import tempfile
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from lcinv import size
    assert abs(size.diameter_km(17.75, 0.14) - 1.0) < 0.002          # the classic 1 km at H = 17.75
    e = size.size_estimate(18.0, 0.15)
    assert np.isclose(e["diameter_km"], 1329 / np.sqrt(0.15) * 10 ** (-3.6), rtol=1e-3)
    assert e["diameter_min_km"] < e["diameter_km"] < e["diameter_max_km"]

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = _json.dumps({"object": {"fullname": "566130 (2017 OY62)", "des": "566130"},
                                "phys_par": [{"name": "G", "value": "0.15"}, {"name": "H", "value": "18.4"}]})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    size.SBDB_URL = f"http://127.0.0.1:{srv.server_port}/sbdb.api"
    with tempfile.TemporaryDirectory() as d:
        Hv, name = size.query_H("566130", cache_dir=d)
        assert Hv == 18.4 and "OY62" in name and os.path.exists(os.path.join(d, "566130.json"))
        srv.shutdown()
        assert size.query_H("566130", cache_dir=d)[0] == 18.4                 # from the cache
        import pandas as pd
        e = size.estimate_for("x", row=pd.Series({"H_V": 15.0}), cache_dir=d)
        assert e["H_source"] == "table column H_V" and np.isclose(e["diameter_km"], size.diameter_km(15.0))
    print("size ok")


if __name__ == "__main__":
    test_cohesion()
    test_effective_parameters()
    test_naming()
    test_size()
