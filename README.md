# Steering BioEmu toward specific protein conformations

This repo contains sparse autoencoders trained on BioEmu's hidden activations, and MLP adapters that
steer BioEmu's denoising toward a named conformational state. It ships the trained
models and the code to run them on the three conformational-change benchmarks in the paper:
cryptic pockets, domain motions, and the kinase DFG flip.

## Install

```bash
uv sync --extra dev --extra bioemu-cuda    # everything below needs a GPU
```

## Model Use

### 1. Encode BioEmu activations with the SAE

```bash
uv run confsteer sae-encode \
  --fasta my_proteins.fasta \
  --sequences 0 \
  --out-dir outputs/sae
```

Generates BioEmu activations at the trained layer, encodes them, and writes the latents to
`outputs/sae/sae_latents.npz`. They are stored sparsely, and every row carries its `sequence_id`,
`residue_index` and `timestep`, so activations can be grouped by protein and residue afterwards. It
also reports reconstruction error on real activations. `--sequences 0` encodes every protein in the FASTA, as above; `--sequences N` takes the first N.

### 2. Steer and score proteins with an adapter

```bash
uv run confsteer steer-apply \
  --adapter kinase_uncond_p1 \
  --fasta my_proteins.fasta --sequences 0 \
  --draws 10 \
  --reference my_target_conformers/ \
  --out-dir outputs/my_proteins
```

`--adapter` is one of `rg_adapter_cold_v2`, `cryptic_uncond_{m1,p1}`,
`domain_motion_uncond_{m1,p1}`, `kinase_uncond_{m1,p1}`. For each protein it generates one steered
and one unsteered structure from a shared noise draw, so the difference is attributable to the
intervention rather than to sampling. `--draws N` samples N structures per protein; pass `--seed` to make a run reproducible. Add `--no-control` to generate only the steered structure.
`--sequences 0` steers every protein in the FASTA, as above; `--sequences N` takes the first N
(the default is 1) and `--sequence-id ID` names one.

Drop `--reference` to steer without scoring. Keep it to report each arm's N/CA/C/O RMSD to the
target conformer and the change between them; negative means the intervention moved the draw
closer.

**The reference layout.** All the proteins stay in the one FASTA, but target structures are put in a direction. Put **one file per protein**,
each named so it **starts with that protein's FASTA id**, and point `--reference` at the directory:

```
my_proteins.fasta          >abl1_kd ... >egfr_kd ...
my_target_conformers/
    abl1_kd_dfg_out.pdb    matched to abl1_kd
    egfr_kd_dfg_out.pdb    matched to egfr_kd
```

`.cif` works too, and one file is fine for one protein. `export_system.py` writes BOTH conformers of
a pair, so copy across only the one you are steering toward. To get a reference for a benchmark system, export its sequence and conformers:

```bash
uv run python scripts/export_system.py \
  --benchmark kinase --system abl1_kd --out-dir examples/abl1_kd
```

`--benchmark` is `cryptic`, `domain_motion`, `kinase`, `cryptic_test` or `domain_motion_test`. 

### 3. Reproduce a benchmark

```bash
uv run python scripts/run_benchmark.py \
  --adapter kinase_uncond_p1 \
  --systems 0 --draws 100 \
  --out-dir outputs/kinase_dfg_out
```

`--adapter` takes the same six names as above. Add `--split test` for the held-out split
(cryptic and domain motion), which is scored as a paired shift against the trained metric region
rather than as success@B -- success@B is defined on the benchmark panel and its rate is not
comparable across populations. Choose what runs with `--systems 3` (the first 3), `--system
egfr_kd` (named, repeatable), or `--systems 0 --draws 100` for the full panel at the reported
depth. Each draw has its own seed, so `--draws N` is N independent draws, and rerunning resumes
rather than repeating.

## Shipped models

| file | what it steers | trained toward |
| --- | --- | --- |
| `sae_k32_time_layer_m1.pt` | — | TopK sparse autoencoder, k=32, time-conditioned |
| `rg_adapter_cold_v2.pt` | radius of gyration | one sigma per unit level (k=1) |
| `cryptic_uncond_m1.pt` / `_p1.pt` | cryptic pockets | apo / holo |
| `domain_motion_uncond_m1.pt` / `_p1.pt` | domain motions | nearer / further conformer |
| `kinase_uncond_m1.pt` / `_p1.pt` | kinase DFG flip | DFG-in / DFG-out |

Each has a `.json` sidecar recording its architecture, injection sites, digest and caveats.
