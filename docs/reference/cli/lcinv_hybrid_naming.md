# `lcinv.hybrid.naming`

*Rename legacy hybrid results ('mode' to 'spin state').*

This command reads its arguments directly (no option parser).

```text
Names of the hybrid spin states (formerly "modes") and backward compatibility.

A hybrid run explores each distinct pole solution of the global layer separately; these are
called spin states: "spin state 1", "spin state 2", ... with file tag spin_state_1, ... :

    posterior_spin_state_1.npz, viewer_spin_state_1.html, shape_viewer_spin_state_1.html,
    obj/spin_state_1/, summary_hybrid.json["spin_states"]

Results written before the rename (posterior_mode1.npz, summary["modes"], "mode 1") are still
read by every script. To rename an old results folder in place:

    python -m lcinv.hybrid.naming results/566130_X05_hybrid
```
