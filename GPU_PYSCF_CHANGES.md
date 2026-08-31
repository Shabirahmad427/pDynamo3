# GPU4PySCF support in pDynamo3

## Scope and comparison baseline

This report describes the local changes that add optional GPU4PySCF support
to pDynamo3's existing PySCF QC model and improve PySCF QC/MM gradients.

The comparison baseline is commit `6b28dd8` on the `main` branch of
`Shabirahmad427/pDynamo3`. The preserved local pre-change backups were checked
and are byte-identical to the corresponding files in that commit.

The GPU code is optional. The existing CPU PySCF path remains selected when
`useGPU=False`, which is the default.

## Files changed

### `addOns/pySCF/QCModelPySCF.py`

- Added the `useGPU` option, defaulting to `False`.
- Added a non-importing availability check using
  `importlib.util.find_spec("gpu4pyscf")`.
- Added lazy import of GPU4PySCF inside `CreateMole()`. This avoids creating a
  CUDA context in the parent EasyHybrid process before it forks a calculation
  worker.
- Added direct construction of GPU4PySCF `RHF`, `UHF`, `RKS`, and `UKS`
  mean-field objects.
- Added GPU4PySCF QM/MM point-charge embedding through
  `gpu4pyscf.qmmm.add_mm_charges()`.
- Selected the cuTENSOR contraction engine for the GPU path.
- Added `gpu4pyscf`, `pyscf`, `mf`, and `mole` to `_unpicklable`. These are
  module or live calculation objects and cannot be serialized safely.
- Reinitializes PySCF state after unpickling before an energy calculation.

The supported GPU methods are limited to `RHF`, `UHF`, `RKS`, and `UKS`,
matching the mean-field methods supported by the pDynamo PySCF interface.

### `addOns/pySCF/QCMMElectrostaticModelPySCF.py`

- Added conversion of GPU-resident CuPy arrays to NumPy arrays before data is
  passed to pDynamo's CPU-side gradient storage.
- Replaced the Python loop that evaluated `int1e_iprinv` once for every MM
  center with PySCF's blocked implementations:
  `grad_hcore_mm(dm)` and `grad_nuc_mm()`.

The blocked calculation is used by both CPU PySCF and GPU4PySCF.

## How PySCF was accelerated

The acceleration has two parts:

1. **GPU4PySCF backend.** When `useGPU=True`, pDynamo creates the RHF, UHF,
   RKS, or UKS mean-field object with GPU4PySCF instead of CPU PySCF. SCF/DFT
   matrix operations, integral processing, and supported analytical-gradient
   operations are then executed on the NVIDIA GPU. QM/MM point charges are
   added through `gpu4pyscf.qmmm`.

2. **Blocked MM-center gradients.** The original pDynamo interface called
   `int1e_iprinv` separately for every MM center. The production system has
   approximately 41,854 MM centers, so this Python loop dominated the full
   gradient time even while the electronic calculation used the GPU. The new
   implementation calls `grad_hcore_mm(dm)` and `grad_nuc_mm()` once. PySCF or
   GPU4PySCF processes the MM centers in optimized blocks and returns the
   complete MM gradient array.

The measured **15.29x** improvement below comes from replacing the scalar
MM-center loop with the blocked API while GPU4PySCF was enabled in both runs.
It demonstrates acceleration of the GPU PySCF QC/MM workflow, but it must not
be presented as a GPU-versus-CPU speedup. A controlled CPU-versus-GPU benchmark
has not yet been recorded.

## Environment and activation

CPU and GPU calculations use the same stable pDynamo environment:

- Conda environment: `/home/shabir/software/envs/pdynamo3`
- Python: 3.10.20
- PySCF: 2.14.0
- GPU4PySCF: 1.8.1
- CuPy: 13.6.0
- NumPy: 1.26.4
- Activation script: `/home/shabir/software/activate_pdynamo3.sh`

There is no separate GPU activation, environment, or source tree. Set
`useGPU=True` on `QCModelPySCF` to select the GPU backend.

## Production-sized benchmark: blocked PySCF MM-center gradients

The saved benchmark uses the exact `new_script.pkl` system from the relaxed
PES calculation. Both cases use GPU4PySCF; the comparison changes only the
MM-center gradient evaluation from the original scalar integral loop to the
blocked PySCF/GPU4PySCF API.

Configuration:

- Date: 2026-08-21
- Total atoms: 41,925
- Free atoms: 10,423
- QC atoms: 71
- Approximate MM centers: 41,854
- Method: RKS/PBE/3-21G
- Charge: -1
- Multiplicity: 1
- PySCF: 2.14.0
- GPU4PySCF: 1.8.1
- PySCF threads: 1
- Calculation: fixed-coordinate energy and full analytical gradient

| Implementation | Calls (s) | Median (s) |
|---|---:|---:|
| Original scalar MM-center loop | 624.534, 632.768 | 628.651 |
| Blocked `grad_hcore_mm` + `grad_nuc_mm` | 51.294, 41.122, 39.863 | 41.122 |

Median speedup: **15.29x**.

The energy remained approximately `-5318701.42165647 kJ/mol`. Gradient RMS
agreed to approximately `2.4e-13 kJ mol-1 A-1`, and repeated blocked calls
differed by at most `1.7e-9 kJ mol-1 A-1` per gradient component. After the
first blocked call, three calls increased RSS by approximately 11 MiB and GPU
memory remained stable.

Saved inputs and results are in the calculation benchmark directory:

- `benchmark_pyscf_pes_memory.py`
- `results/pyscf_pes_gpu_gradient2.json`
- `results/pyscf_pes_gpu_batched_gradient3.json`
- `results/pyscf_pes_gpu_energy3.json`

This is an algorithmic speedup of the GPU PySCF QC/MM gradient path. It is not
a CPU-versus-GPU comparison.

## Small-system validation benchmark

The benchmark isolates the code changed in
`QCMMElectrostaticModelPySCF.py`. It compares the original per-center integral
loop with the new blocked PySCF implementation.

Configuration:

- Date: 2026-08-22
- PySCF: 2.14.0
- CPU threads: 1
- QC system: water, RHF/3-21G
- MM centers: deterministic point charges located 5-25 bohr from the QC
  system
- Timing: median of three calls after SCF convergence
- Speedup: original time divided by blocked time

| MM centers | Original loop (s) | Blocked implementation (s) | Speedup | Maximum gradient difference |
|---:|---:|---:|---:|---:|
| 100 | 0.007023 | 0.003220 | 2.181x | 6.939e-17 |
| 1,000 | 0.115224 | 0.015882 | 7.255x | 9.714e-17 |

These small-system results measure only MM-center gradient construction. They are not
end-to-end energy, optimization, or PES timings. Total acceleration depends
on the QC method, basis, number of QC atoms, number of MM centers, SCF cost,
and number of gradient evaluations.

## GPU verification and timing status

A GPU-enabled EasyHybrid calculation was previously verified with an alanine
dipeptide system (22 atoms), RKS/B3LYP/3-21G, QM/MM embedding, and
`useGPU=True`. The calculation completed through EasyHybrid's worker process.

No controlled CPU-versus-GPU timing series is currently available. A new GPU
comparison could not be run on 2026-08-22 because `nvidia-smi` could not
communicate with the NVIDIA driver and CuPy returned
`cudaErrorNoDevice`. Therefore, this report does not claim a numerical GPU
speedup.

Before publishing GPU performance numbers, run the same system, coordinates,
method, basis, convergence settings, and gradient setting with
`useGPU=False` and `useGPU=True`, include at least one warm-up call, and report
the median of repeated measurements.

## Known limitations

- GPU4PySCF is required only when `useGPU=True`.
- GPU execution supports `RHF`, `UHF`, `RKS`, and `UKS` in this interface.
- GPU arrays are copied to host memory where pDynamo requires CPU-side data.
- GPU acceleration is expected to depend strongly on calculation size; no
  general speedup factor is claimed here.

## Final unified-installation validation (2026-08-31)

CPU and GPU calculations now use the single stable environment
`/home/shabir/software/envs/pdynamo3`, activated with `activate_pdynamo3`.
The environment contains NumPy 1.26.4, SciPy 1.15.3, PySCF 2.14.0,
CuPy 13.6.0, GPU4PySCF 1.8.1, and cuTENSOR 2.6.0. `pip check` reports no
broken requirements, and CuPy detects the NVIDIA GeForce RTX 4060 Ti.

Numerical validation used pDynamo's `QCModelPySCF` on H2 with RHF/STO-3G:

- CPU (`useGPU=False`): `-2932.051561066158 kJ mol-1`.
- GPU (`useGPU=True`): `-2932.051561066207 kJ mol-1`.
- Absolute difference: `4.957e-11 kJ mol-1`.

The official `examples/book/Example1.py` CPU test completed successfully.

The useful change from the former `pDynamo3_pairlist_fix` experiment was
merged into the main tree at
`pMolecule/NBModel/__extensions__/pyrex/pMolecule.NBModel.UpdateChecker.pyx`.
Before deleting an updatable pair-list scratch node, `pNode.Clear()` now
releases its owned native pair-list wrappers. `pMolecule.NBModel` was
regenerated with Cython 3.2.5 and all nine extensions rebuilt successfully.
The official `NBModelCutOffCentering.py` regression then completed with 11
calls, 11 updates, and a maximum energy deviation of `0.00000`.

The obsolete GPU-test, standalone GPU4PySCF, batched-pDynamo, and
pairlist-fix environments/source trees were removed only after the unified
tree passed these checks.
