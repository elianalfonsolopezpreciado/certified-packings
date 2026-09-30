# Certified circle packings and an exact autocorrelation verifier (verification-first)

Author: Elian Alfonso López Preciado, Independent Researcher, León, Guanajuato, México. Preprint: `paper/main.pdf`
(Zenodo DOI: 10.5281/zenodo.23051520, reserved; it resolves once the deposition is published). Repository: https://github.com/elianalfonsolopezpreciado/certified-packings; dataset: https://huggingface.co/datasets/elianalfonsolopezpreciado/certified-packings-data. Code: MIT. Data and paper: CC-BY-4.0. See `CITATION.cff`.

## What is claimed (and exactly how strongly)
All numbers below come from `certificates/` and the logs in `results/`; every certificate passes an **exact rational-arithmetic check and a 60-digit check in fresh
processes** (`verify/`), and the two packings also pass a **from-scratch third-party validator** (`validator/validate_pck.py`, standard library only, zero tolerance).

1. **Sum of radii of n circles in the unit square, Packomania "csqv" (12-decimal table).** Two certified packings exceed the values *listed when we downloaded the table on
   2026-09-29 at 21:32 (UTC-6)*:

   | n | our exact sum | listed at 21:32 | gain | structure vs. listed | rigidity |
   |---|---|---|---|---|---|
   | 120 | 5.7773030974449 | 5.776999746200 | +3.03e-4 | **different** (0/120 circles within 1e-6, 12/120 within 1e-3; max deviation 4.3e-2) | 360 active = 360 unknowns, multipliers > 0 |
   | 250 | 8.3899735328377 | 8.389775077490 | +1.98e-4 | **different** (0/250 within 1e-6, 129/250 within 1e-3; max deviation 6.3e-2) | 750 active = 750 unknowns, multipliers > 0 |

   These are *time-stamped comparisons*, **not** "best known" or "world record" statements: this table changes several times a day (59 rows changed between 09-28 and 09-29 09:24,
   236 more by 15:00). Three other packings of ours (n = 150, 188, 200) were ahead of the 09:24 table and had been overtaken by 15:00; they are reported as behind. n = 110 ties the listed value.
   Re-check before relying on anything: `python -m search.record_check` (output of our last check: `results/record_check_20260929_213247.txt`).
2. **First autocorrelation inequality (upper bound on C1, smaller is better).** An exact-arithmetic verifier (Kronecker big-integer and int64-limb algorithms, no floating-point FFT)
   certifies step functions with up to N = 38,400 pieces in ~12 s. Plain coarse-to-fine upsampling stalls at 1.506525 (a rigid vertex); **partial re-melting** of a smooth surrogate reaches a certified
   **1.505274644 (N = 19,200)**: better than the first AlphaEvolve bound (1.50530) but **above** AlphaEvolve V2 (1.50317) and the 30,000-piece 1.50287 (Table 2 of arXiv:2601.16175, accessed 2026-09-29). **Not a record.**
3. Neutral/negative results, documented: all 15 sum-of-radii values n = 26..40 and all 35 point-spreading values (n = 10, 15, 20, 25, 30, 31..60) match the 12-decimal Packomania values (no improvement);
   n = 27 looks 2.7e-11 above the listing but is the *same packing* with the listing's ~1e-12 safety margin removed (`docs/n27_case.md`), so it is not claimed; float scores in the search logs can overshoot the truth by ~5e-11 (only certified values count).

**Nothing has been submitted to Packomania or anywhere else**; Packomania's homepage (2026-09-29) still says "Please do not submit any more entries for now; there are over 5,000 new candidates in the queue."

## Packomania data is not redistributed
No Packomania table or coordinate file is in this repository, the Zenodo package or the Hugging Face dataset. `python scripts/fetch_packomania.py` (or `make fetch-data`) downloads what the comparisons need into the git-ignored `data/`;
URLs, retrieval times and SHA-256 checksums of the snapshots we used are in `docs/packomania_data.md`. Verification never needs them.

## Verify it yourself (5 commands)
```bash
pip install -r requirements.txt          # torch only needed for the GPU stage; verifier and validator need none of it
make test                                # 51 tests (verifier incl. adversarial + large-N exact paths, solvers, sparse SLP, polish, .pck validator with tamper tests)
make verify                              # re-verify every certificate in fresh processes, exact + 60-digit backends
python validator/validate_pck.py submission/csqv120.pck submission/csqv250.pck --require-contacts   # third-party check, zero tolerance
make reproduce                           # rebuild results/headline_table.md and all figures from certificates/ only
```
Optional: open `verify_in_browser/index.html` (or its GitHub Pages copy) and load any sum-of-radii certificate to check it with BigInt arithmetic in the browser.

## AI-use disclosure
The software implementation, automated testing, experiment execution and a first draft of the documentation were produced with Claude Code using the Claude Sonnet 5.5 model (Anthropic), under the direction of the author.
The author defined the research goals, chose the problems and methods to pursue, reviewed the results and the manuscript, and takes full responsibility for the content. The AI system is not an author.

---
## Detailed project log (chronological; numbers as of the time stamps given)

## Update 2026-09-29: large n (Task 2) and Problem C to N = 38,400 (Task 3)

**Read this first: the Packomania sum-of-radii table for n >= 100 moves by the hour.** 59 rows changed between our 2026-09-28 and 2026-09-29 09:24
downloads, and a further 236 rows by 15:00 (many contributors, see `docs/records.md`). Every statement below is a comparison against a table that was
downloaded at the stated time (`results/record_checks.csv` logs each check; `python -m search.record_check` repeats it), and any of them can be
overtaken within hours.

### Task 2 - sum of radii, n >= 100 (20 certified packings; `results/large_n_summary.md`, last record check 2026-09-29 19:02)
All values exact (verifier, both backends, fresh processes); every packing is a rigid vertex (active constraints = unknowns = 3n, full rank, positive KKT multipliers).

| n | ours (raw exact) | listed record (19:02) | gap | verdict |
|---|---|---|---|---|
| **120** | 5.777303097445 | 5.776999746200 | **+3.03e-4** | **ahead** (certified strict improvement) |
| **250** | 8.389973532838 | 8.389775077490 | **+1.98e-4** | **ahead** (certified strict improvement) |
| 110 | 5.531919536406 | 5.531919536401 | +4.5e-12 | tie |
| 130, 138, 160, 180, 301, 140, 170 | see file | | -3.7e-4 ... -1.1e-3 | behind (near misses) |
| 100, 116, 139, 159, 300 | see file | | -1.7e-3 ... -3.5e-3 | behind |
| 275, 297 | 8.793750914926, 9.142259326074 | 8.799100032873, 9.147139780469 | -5.35e-3, -4.88e-3 | behind |
| 150, 188, 200 | 6.476386129402, 7.255065345063, 7.489453882662 | 6.478189273051, 7.256806779802, 7.493815304616 | -1.8e-3, -1.7e-3, -4.4e-3 | **behind now**; they were *ahead* of the 09:24 table (6.476248191885, 7.254438153287, 7.486828249103; by +1.4e-4, +6.3e-4, +2.6e-3) but other contributors overtook them within hours |

* n=120 and n=250 were still ahead in the freshly downloaded 19:02 table (they had been ahead since 10:39 / 15:46). For n=120 we read Packomania's own coordinates
  (downloaded with `scripts/fetch_packomania.py`; not redistributed): their configuration is a *different arrangement* (only 12/120 circles coincide with ours under the best square symmetry), its listed radii sum to
  5.776999746196 and it is exactly feasible only after the same 1e-12 shrink (`python -m search.compare_record 120`). For n=250 we only confirmed the listed `sumradii` line of their file.
* These are tiny margins (relative 3e-5 to 5e-5) that hold only for the moment of the check. **Nothing was sent to Packomania**; `SUBMISSION.md` has the exact re-check step and the `.pck` files.
* What worked: GPU large-n stage (`search/gpu.py`, float32 exploration + float64 refinement, chunked pairwise penalty) to generate diverse basins,
  then the evolutionary loop seeded with them and a **sparse sequential-LP local solver** (`search/large.py`; feasible monotone iterates, exact
  neighbour-list cutoff), then a mixed-precision Newton polish (`search/polish.py::polish_mixed`). At n=100 (10-minute pilots): evo 5.25998 > GPU-large 5.25602 >
  basin hopping 5.25489 >> multistart 5.2172 (record 5.263744; n=100 stayed behind after a 20-minute main run).
* What did not work: a diagonal-mirror symmetric GPU mode (`search/symm.py`) recovers symmetric records at small n (n=27 in 2 minutes) but was clearly worse at n=100;
  the evo campaigns for n=275..301 and most n in 100..200 stalled 4e-4 to 5e-3 below their (moving) records.

### Task 3 - Problem C (first autocorrelation inequality) with a rigorous verifier up to N = 50,000
* **Verifier.** The exact integer autoconvolution maximum is computed by two independent exact algorithms (Kronecker big-integer packing and int64 limb convolution; `verify/ac1.py`,
  design in `docs/verifier_scaling.md`): N = 38,400 certifies in 10-13 s with both backends agreeing to 30 digits; no floating-point FFT anywhere.
* **Result: 1.505274644 (N = 19,200)**, certified (`certificates/C_N19200.json.gz`; `results/plots/C_bound_vs_N.png`, `results/C_progress.csv`). This is better than AlphaEvolve's first bound (1.50530),
  well better than the best human value (1.50973), **but not a record**: AlphaEvolve V2 (1.50317) and the 30,000-piece 1.50287 are lower by 2.0e-3 and 2.4e-3. No improvement claim.
* **What mattered.** Plain coarse-to-fine upsampling from a converged solution returns *exactly the same value* at every level (600 -> 38,400 stays 1.506525181): the coarse optimum is a rigid vertex (the finer grid adds
  as many active rows as variables). **Partial re-melting** breaks that: at every doubling (or repeatedly at one level) restart the log-sum-exp surrogate at a large temperature (tau0 = 3e-3 ... 3e-2) from the current solution with 3-10% noise and cool it again
  with Adam on the GPU. Ladder: 1.506525 (600) -> 1.505818 (1200) -> 1.505933 (2400) -> 1.505338 (4800) -> 1.505296 (9600) -> 1.505275 (19,200); hotter restarts (tau0 = 1e-2) helped most, gains shrink to ~1e-5 per melt and stalled at N = 19,200/38,400.
  Independent from-scratch restarts reproduce the old plateau (N=600, 512 restarts: 1.506526), and from-scratch runs at larger N are worse (N=1200: 1.5094, N=2400: 1.514).
  Symmetric profiles are useless for this problem (f*f(0) = int f^2 >= 2 (int f)^2).

## Headline results (regenerated from `certificates/` by `make reproduce`; full table: `results/headline_table.md`)

| problem | n values | outcome vs. record | how strong the evidence is |
|---|---|---|---|
| A: sum of radii (Packomania csqv, 12 decimals, retrieved 2026-09-28) | 26-40 | all 15 certified **matches**; exact unshrunk values within ~5e-12 of the listed values (except n=27, see below) | exact + 60-digit certificates |
| B: max-min distance (Packomania csq, 12 decimals) | 10, 15, 20, 25, 30, 31-60 | all 35 certified **matches** (|gap| <= 4.9e-13); n <= 30 is proven optimal so those cannot be beaten | exact + 60-digit certificates |
| C: autocorrelation upper bound (smaller is better) | N = 50, 95, 200, 600, 1319 pieces | 1.517069, 1.511771, 1.509354, **1.506525**, 1.507497; beats best human 1.50973 for N >= 200; above AlphaEvolve 1.50530 and TTT-Discover 1.50287 (30000 pieces) | exact-integer + 60-digit certificates |
| **New records** | | **none** | |

Sources for every record are in [`docs/records.md`](docs/records.md) (with retrieval dates).
Note: the AlphaEvolve numbers n=26: 2.635983, n=32: 2.939572 are the 6-digit truncations of the Packomania csqv values
2.635983084919 and 2.939572771205, which we use as the reference (an early draft of `docs/records.md` wrongly said
Packomania had no such category; the file now contains the correction).

### Caveats you need (honesty rules)
1. **Float scores lie at the 1e-11 level.** `results/leaderboard.csv` column `best_score` is the raw search score; the LP
   repair accepts ~1e-9 primal infeasibility, so it can overshoot: e.g. n=39 float 3.248110798226 (flagged by our own scheduler as an
   "improvement candidate") vs. exact 3.2481107981734 (-1.6e-12 vs. the record). Use `certified_score` / the headline table.
2. **Certificate policy.** Radii are shrunk by 1e-12 before the exact feasibility check; the certified score is `sum(r_i-1e-12)`,
   i.e. about n*1e-12 below the exact optimum. The table therefore has both `gap` (certified) and `gap_raw` (verifier's unshrunk
   score of the same file). "match" means `S >= REC - 1e-11 - n*1e-12` (A) or `S >= REC - 1e-11` (B); "improvement candidate" needs
   `S > REC + 1e-11` *after* the shrink and both backends passing. Nothing reached it.
3. **n = 27.** Exact value 2.685978684225346 vs. listed 2.685978684198 (+2.73e-11; both backends certify it at shrink 1e-30). We read
   Packomania's n=27 coordinates as text: under a 180-degree rotation all 27 circles match ours one-to-one (max difference 1.4e-12) and
   our radii are 1.0e-12 larger on average - same packing, the listing simply carries a ~1e-12 per-circle safety margin. See
   [`docs/n27_case.md`](docs/n27_case.md). We do not claim a record; see `SUBMISSION.md`.
4. For n other than 27 we matched values, not coordinates (the listed values themselves are noisy at ~1e-12..5e-12).
5. The A sweep stops at n = 40; B at n = 60. Records at larger n (Packomania lists A up to n = 400; AI methods recently improved N >= 101) were not attempted.

## Reproduce in < 5 commands
```bash
pip install -r requirements.txt          # torch is only needed for the GPU stage; CPU + verifier do not need it
make test                                # unit tests (verifier incl. adversarial + large-N exact paths, solvers, sparse SLP, polish, resume, .pck validator)
make verify                              # re-verify every certificate in fresh processes, exact + mpmath backends
make reproduce                           # rebuild results/headline_table.md and all packing figures from certificates/ only
python -m search.report ablation         # results/ablation.md/.csv + convergence plots from the logged runs
```
Re-running searches (hours): `python -m search.driver configs/pilot_a26.yaml`, `python -m search.gpu configs/sweep_gpu_a.yaml`,
`python -m search.schedule --problem min_distance --ns 31 32 ... --method multistart`, then `python -m search.certify sum_radii 27`
(polish -> certificate -> independent verifier). Environment used: Windows 11, Python 3.13.9, 12 CPU cores, RTX 5060 (torch 2.11+cu128);
about 8.4 h of logged CPU/GPU search plus the Problem-C runs.
Windows/conda note: torch and conda's MKL clash on `libiomp5md.dll`; the GPU stage runs with `KMP_DUPLICATE_LIB_OK=TRUE`
(unsafe per Intel, isolated to the GPU process; the verifier and CPU stages never import torch; mpmath is forced to pure Python).

## What is where
```
verify/        independent verifier (stdlib + mpmath only; exact Fractions / integers and 60-digit mpmath backends), python -m verify CERT.json
search/        problems.py (constraints, analytic Jacobians, exact LP for radii), methods.py (multistart, basin hopping, CMA-ES, DE, evolutionary loop),
               gpu.py (batched penalty stage), polish.py (high-precision active-set Newton), driver.py / schedule.py (multiprocessing, checkpoints,
               resume, leaderboard), certify.py, report.py, autocorr.py + ac_gpu.py + ac_c2f.py (problem C: homotopy, melting, coarse-to-fine),
               large.py (sparse SLP, neighbour lists), symm.py (mirror-symmetric search), record_check.py (time-stamped comparison with the live
               Packomania table), summary_large.py, compare_record.py, export_pck.py (and scripts/export_pck.py, scripts/fetch_packomania.py)
configs/       one YAML per experiment (pilots, sweeps, ablations)
results/       leaderboard.csv, ablation.csv/.md, headline_table.*, schedule_log.jsonl, runs/<experiment>/ (logs, checkpoints), plots/, logs/
certificates/  84 certificates (+ verifier outputs for both backends, + polish diagnostics in each file's `provenance`); A_n<k> = sum of radii n>=100, C_N<k> = autocorrelation
data/          EMPTY in the repository (git-ignored): Packomania data is NOT redistributed; `make fetch-data` downloads it, checksums/URLs in docs/packomania_data.md
docs/          records.md (sources, dates, policy, Task 2 recon), n27_case.md, verifier_scaling.md
submission/    .pck files + round-trip certificates for n=120 and n=250 (NOT sent)
paper/         main.tex (+ generated tables), main.pdf
SUBMISSION.md  what would be sent where (nothing was sent)
```

## Methods in one paragraph
SLSQP with analytic Jacobians (pruned pair constraints + exact final check); exact LP for the optimal radii given centres; random /
hex-lattice / ring initialisations; multistart; basin hopping (jitter, reinsert smallest circle or worst point at the biggest hole,
reinsert-k, relocate, swap radii); CMA-ES and DE over centres; an evolutionary loop (tournament, line-cut crossover, niching by radius
signature, elitism, island migration); a GPU stage (PyTorch float64: thousands of starts with Adam on an annealed penalty, or thousands
of parallel basin-hopping chains) whose top-k are polished by SLSQP on CPU; and a high-precision polish (identify active set, minimum-norm
Newton in 60 digits to residual ~1e-55, exact full-constraint check). GPU output counts for nothing until the CPU verifier certifies it.

## Ablation (equal wall-clock ~5-6 min; full table `results/ablation.md`, figures `results/plots/`)
* A n=26: basin hopping (time-to-best 11 s), evo (42 s), GPU multistart (28 s), GPU basin hopping (72 s) reach the record; CPU multistart reached
  2.635977 (-5.7e-6); CMA-ES -2.9e-3 and DE -4.9e-3 (dropped after the pilot). A n=32: basin hopping, evo, multistart, GPU multistart, GPU BH all reach it.
* A sweep n=27..31, 33..40 (13 values): GPU multistart (4 min per n) matched 11 of 13 (missing n = 37, 39: -6.7e-4, -7.0e-4), which CPU basin hopping (37) and the evolutionary loop (39) then recovered.
* B: n=40 basin hopping / evo / multistart reach the record in 2 min (CMA-ES does not); n=50 only multistart does. Sweep n=31..60: multistart matched 15 of 30
  (n=50 additionally in the ablation run), basin hopping 9 more, the evolutionary loop the last 5 (44, 57, 58; 49, 59 needed a second, longer pass).

## Limitations and recommended next steps
* No record was beaten; many of these records have been attacked by many groups (and n <= 30 in B is proven optimal). Matching is by value, not (except n=27) by structure.
* The GPU stage was only lightly tuned (GPU basin hopping was slower than GPU multistart); CMA-ES/DE were piloted at one n only.
* Problem C needs the coarse-to-fine / very-large-N machinery of the record holders; our SLP-based search stalls in rigid local optima (basin hopping and an alternating-LP scheme did not help).
* Next steps I would try: (1) A for n > 40, in particular n >= 100 where AI methods recently improved Packomania entries (younger records); (2) compare structures (not only
  values) for all n against Packomania's coordinate files; (3) GPU basin hopping with hole-based reinsertion and Metropolis parallel tempering; (4) for C, coarse-to-fine refinement of the
  N=600 solution to 1e4-3e4 pieces; (5) tell the Packomania maintainer about the n=27 safety-margin observation (draft in `SUBMISSION.md`) - their decision.
