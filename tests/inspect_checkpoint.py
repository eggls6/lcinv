"""Summarise the live points of a dynesty checkpoint."""
import sys
import numpy as np
import dynesty

s = dynesty.NestedSampler.restore(sys.argv[1])
v, L = s.live_v, s.live_logl
print("nlive", len(L), "logl range %.2f .. %.2f" % (L.min(), L.max()))
for i, n in enumerate(["lambda", "beta", "P", "ln_jit"]):
    print(n, np.percentile(v[:, i], [0, 16, 50, 84, 100]).round(5))
print("beta>0 fraction", np.mean(v[:, 1] > 0))
print(np.round(v[np.argsort(-L)[:15]], 4))
