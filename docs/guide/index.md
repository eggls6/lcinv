# User guide

The guide follows a target from photometry to a published shape model. Each page explains one
step: what it does, which options control it, and how to read its output. For every option of
every script see the [command-line reference](../reference/cli/index.md).

## A typical session

```bash
# 1. global layer: spin states, period, evidence, convex shapes
python run_inversion.py data/K03Q21N.lc --designation K03Q21N --npool 8 --outdir results/K03Q21N
python check_convergence.py results/K03Q21N

# 2. local layer: joint spin and shape within each spin state
python run_hybrid.py data/K03Q21N.lc --designation K03Q21N --global-dir results/K03Q21N \
       --chains 4 --outdir results/K03Q21N_hybrid
python check_convergence.py results/K03Q21N_hybrid

# 3. alternatives and the comparison
python complexity_scan.py results/K03Q21N --lmax 2 4 6 --tau0 0.6 1.2
python run_binary.py --global-dir results/K03Q21N --outdir results/K03Q21N_binary
python compare_models.py results/K03Q21N results/K03Q21N_binary results/K03Q21N_hybrid

# 4. products
python export_obj.py results/K03Q21N_hybrid --diameter-km 1.2
python make_viewer.py results/K03Q21N
```

## Pages

| Step | Page |
|---|---|
| Prepare the photometry and geometry | [Input data](input-data.md) |
| Find the period and set priors | [Period search](period-search.md) |
| Map the spin states | [Global convex inversion](global-inversion.md) |
| Sample shape and spin jointly | [Hybrid model](hybrid.md) |
| Test the two-lobe hypothesis | [Contact-binary model](binary.md) |
| Fast rotators and long exposures | [Exposure smearing](smearing.md) |
| Lommel-Seeliger versus Lambert | [Scattering law](scattering.md) |
| How much shape detail, which model | [Shape complexity and model comparison](model-comparison.md) |
| Physical plausibility | [Cohesion and size](physics.md) |
| Is the result trustworthy | [Convergence](convergence.md) |
| Figures, 3D viewers, meshes | [Outputs, viewers and OBJ export](outputs.md) |
| Many targets | [Batch runs and MPC download](batch.md) |
| Speed and devices | [GPU and performance](performance.md) |
