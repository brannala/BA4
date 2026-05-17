# BA4 Implementation Plan

BA4 (BayesAss Edition 4) is a C++ program for Bayesian inference of recent
migration rates between populations from phased linked-SNP data. It extends
BA3 (Wilson & Rannala 2003) by replacing the per-locus allele-frequency
likelihood with a pedigree-derived latent ancestry mosaic model and a
Dirichlet-Multinomial haplotype likelihood ported from mongrail2
(Chakraborty & Rannala 2023, 2025).

This document is the implementation plan. The theoretical framework is in
`~/repos/BA4-manuscript/ba4_theory.tex`.

---

## 1. Scope and locked design decisions (v1)

| Decision | Choice |
|---|---|
| Language | C++17 |
| Build system | CMake |
| Dependencies | GSL (RNG, special functions) + htslib (VCF I/O) |
| Input format | Phased biallelic-SNP VCF only |
| Output | Posterior summaries (migration rates, per-individual class posteriors, Savage-Dickey BFs) |
| Window scheme | One window per whole chromosome |
| SNPs per chromosome | At most `L_max = 20`, uniformly spaced in cM, selected at load time |
| Centromere handling | None — whole chromosome is one unit |
| LD pruning | Never. Within-window LD is the signal. |
| Class update | Collapsed Gibbs over `G_i` (replaces BA3's M-H + Hastings) |
| Recombinant U(x) | Sparse ancestry-vector enumeration, `k_max = 4` default |
| Other class | Computed by subtraction via Bayes' theorem (manuscript §4) |
| Parallelism | OpenMP over chromosomes |

The set of identifiable genealogical classes is:

- `pure_k` for each population `k`
- `F1(j→k)` for each ordered pair `j ≠ k`
- `backcross(j→k)` for each ordered pair `j ≠ k`
- `other` (nuisance, no closed-form likelihood; posterior by subtraction)

For `K` populations the cardinality is `|G| = K + 2K(K−1) + 1`.

---

## 2. Module layout

```
BA4/
├── CMakeLists.txt
├── PLAN.md                     # this file
├── README.md
├── LICENSE
├── src/
│   ├── main.cpp                # ~200 lines: argparse, load, run MCMC, write output
│   ├── io/
│   │   ├── vcf_reader.{h,cpp}      # port mongrail2 vcf_reader.c; drop MAXLOCI=12 cap
│   │   ├── metadata.{h,cpp}        # port BA3 individual→population mapping
│   │   ├── snp_selector.{h,cpp}    # NEW: uniform-cM thinning to L_max per chrom
│   │   └── output.{h,cpp}          # port BA3 BA3out.txt writer; extend for mosaic posteriors
│   ├── model/
│   │   ├── genealogy.{h,cpp}       # GenealogicalClass enum, prior P(G|m), class iteration
│   │   ├── haplotype_cache.{h,cpp} # logP_k full-hap cache, anchor-set version counter
│   │   ├── subhap.{h,cpp}          # port mongrail2 log_prob_subhap / get_distinct_subhaps
│   │   ├── ancestry_vector.{h,cpp} # NEW: sparse enumerator of z with ≤k_max transitions
│   │   ├── q_haldane.{h,cpp}       # port mongrail2 prQ()
│   │   ├── u_recombinant.{h,cpp}   # port mongrail2 compute_U, switched to sparse enum
│   │   ├── likelihood.{h,cpp}      # per-class diplotype likelihoods
│   │   └── forward_backward.{h,cpp}# NEW: Baum-Welch on bivariate ancestry chain
│   ├── mcmc/
│   │   ├── state.{h,cpp}           # parameter + latent variable container
│   │   ├── propose_m.{h,cpp}       # port BA3 random-walk M-H on migration rates
│   │   ├── gibbs_class.{h,cpp}     # NEW: collapsed Gibbs over G_i
│   │   ├── sample_mosaic.{h,cpp}   # NEW: FB sample of Z_i for backcrosses
│   │   ├── other_posterior.{h,cpp} # NEW: P(G=other | Z) via Bayes (manuscript eq 4.16)
│   │   ├── autotune.{h,cpp}        # port BA3 autotuning
│   │   └── bayes_factor.{h,cpp}    # port BA3 Savage-Dickey BFs
│   └── util/
│       ├── rng.{h,cpp}             # GSL Taus wrapper (port BA3 init)
│       ├── logspace.{h,cpp}        # log-sum-exp, Kahan summation
│       ├── bits.{h,cpp}            # haplotype bitstring ops; uint64_t-based, no L=16 cap
│       └── cli.{h,cpp}             # argparse
├── tests/
│   ├── unit/                       # one binary per module; doctest or Catch2
│   └── golden/                     # end-to-end MCMC runs vs frozen outputs
└── examples/
    └── simulated/                  # synthetic K=3 datasets with known truth
```

---

## 3. Port-vs-write map

### Port from mongrail2 (`~/repos/mongrail2/`, pure C)
| mongrail2 source | BA4 destination | Adaptation needed |
|---|---|---|
| `prQ` (inference.c:842) | `model/q_haldane.cpp` | Cache key adds chromosome index |
| `log_prob_subhap` (459) | `model/subhap.cpp` | None — math is identical |
| `log_prob_single_hap` (435) | `model/haplotype_cache.cpp` | None |
| `get_distinct_subhaps` (407) | `model/subhap.cpp` | None |
| `cache_init` / `cache_free` (16–70) | `model/haplotype_cache.cpp` | Restructure for mutable counts; expose `bump_version()` for invalidation |
| `compute_U` / `compute_U_cached` (526 / 121) | `model/u_recombinant.cpp` | Replace `for (z = 0; z < 2^L; ++z)` with sparse iterator |
| VCF reader (`vcf_reader.c`) | `io/vcf_reader.cpp` | Drop MAXLOCI=12 cap; emit per-chromosome haplotype arrays |
| Kahan summation (test_kahan) | `util/logspace.cpp` | |
| Bit ops (`algorithms.c`) | `util/bits.cpp` | Promote `unsigned int` haplotype keys to `uint64_t` |

### Port from BA3 (`~/repos/BA3/`, C++11)
| BA3 source (main.cpp lines) | BA4 destination | Adaptation needed |
|---|---|---|
| GSL RNG init (658–659) | `util/rng.cpp` | None |
| Migration-rate M-H proposal (1188–1244) | `mcmc/propose_m.cpp` | Keep reflection + Hastings logic; swap likelihood call |
| `fillMigrantCounts` / `migCountLogProb` | `model/genealogy.cpp` | Generalize for K-population class enumeration |
| Autotuning (145–151, 1239–1242) | `mcmc/autotune.cpp` | None |
| Savage-Dickey BF (2866–3046) | `mcmc/bayes_factor.cpp` | None |
| CLI flag layout (30–77) | `util/cli.cpp` | Keep flag letters where they still apply |
| VCF/metadata glue (304–408) | `io/vcf_reader.cpp` / `io/metadata.cpp` | Combine with mongrail2 reader |
| Output formatter (1743–1859) | `io/output.cpp` | Extend with per-individual class posteriors |

### Discard
- BA3's per-locus allele-frequency likelihood (`logLik`, `oneLocusLogLik`). Replaced by BA4 haplotype likelihoods.
- BA3's class M-H + Hastings correction. Replaced by Gibbs.
- mongrail2's F2 likelihood (`lik_f`). F2 is absorbed into `other`.
- mongrail2's six-model maximum-likelihood classifier. Replaced by BA4's K-population MCMC.
- BA3's microsatellite handling.
- BA3's 5-column text input format.

### Write fresh
1. **`model/ancestry_vector`** — Sparse enumerator of ancestry vectors `z ∈ {0,1}^L` with `popcount(z[i] ^ z[i+1])` transitions ≤ k_max. Exposes a forward iterator that consumers fold into `compute_U`. Vector count: `2·Σ_{k=0..k_max} C(L−1,k)`. For L=20, k_max=4: ~10,000.
2. **`model/u_recombinant`** — Backcross U(x) with sparse enumeration:
   ```
   log U(x) = LSE_{z ∈ sparse} [ log Q(z) + log P(x | z, f) ]
   ```
   using running-max log-sum-exp (manuscript Appendix B).
3. **`model/forward_backward`** — Baum-Welch on the bivariate Markov chain over `{AA, AB, BA, BB}` (or `{AA, AB}` collapsed when one homolog is the pure ancestor). Used for backcross-class individuals to (a) compute the exact marginal `P(x | backcross, f)` for the Gibbs update and (b) sample `Z_i` for the `other` posterior.
4. **`mcmc/gibbs_class`** — Categorical Gibbs:
   ```
   P(G_i = G | data, m, f) ∝ P(G_i = G | m) · Π_c P(x_{ic} | G, f^c)
   ```
   Normalize over `G ∈ \calG`, draw. After drawing, if G changed in/out of `pure_k` for any k, bump the haplotype-cache version for k.
5. **`model/haplotype_cache::update_anchor_set`** — Incremental update of `n_k^c` when an individual's class assignment changes the non-migrant anchor set. Maintains a per-population version counter consumed by `u_recombinant` and `subhap` caches.
6. **`mcmc/other_posterior`** — Compute `P(G_i = other | Z_i)` by subtraction (manuscript eq 4.16). Pure book-keeping over forward-backward output.
7. **`io/snp_selector`** — Read all SNPs per chromosome from the VCF, compute cumulative cM positions (Haldane from bp distance × user-supplied rate, default 10⁻⁸ cM/bp), select L_max uniformly spaced in cM. If a chromosome has < L_max SNPs, use all of them.

---

## 4. Data model

### Per-chromosome state (loaded once)
- `positions_bp[c][l]` — base-pair positions of the L_c ≤ L_max selected SNPs.
- `genetic_dist_M[c][l]` — inter-SNP genetic distance in Morgans.
- `haplotypes[c][i][h]` — for each individual i and homolog h ∈ {0,1}, a `uint64_t` bitstring over L_c loci.

### Per-iteration mutable state
- `m[K][K]` — current migration rate matrix (`m[i][j]` = fraction of pop i that are F1 from j).
- `G[N]` — current genealogical class assignment per individual.
- `Z[N][C]` — current bivariate ancestry mosaic per individual per chromosome (only meaningful for backcross; stored as run-length-encoded segments).
- `n[K][C][hap]` — haplotype counts under the current anchor set.
- `version[K]` — anchor-set version counter per population, incremented on every membership change.

### Caches (rebuilt lazily, invalidated by version mismatch)
- `logP_full[K][C][hap]` — full-haplotype log probabilities.
- `logQ[C][z_idx]` — Q(z) for each sparse-enumerated ancestry vector.
- `U_memo[N][C][k_pair]` — per-individual U(x) per chromosome per (pure-pop, mixed-pop) pair. Invalidated when either pop's version counter advances.

### Memory budget (K=5, N=200, C=30, L=20, k_max=4)
- `logP_full`: 5·30·400·8 ≈ 0.5 MB
- `logQ`: 30·10⁴·8 ≈ 2.4 MB
- `U_memo`: 200·30·20·8 ≈ 1 MB
- Working set: < 50 MB total.

---

## 5. MCMC kernels

Per iteration (manuscript Algorithm 1):

1. **Migration rates** — random-walk M-H on each `m[i][j]` with reflection at boundary (BA3 port). Acceptance uses full data likelihood under proposed `m'`.
2. **Genealogical classes** — collapsed Gibbs sweep over individuals. For each i:
   - Compute `log P(G | m) + Σ_c log P(x_{ic} | G, f^c)` for every `G ∈ \calG`.
   - Normalize, draw categorically.
   - If new G differs in non-migrant status, update anchor-set membership and bump pop version.
3. **Ancestry mosaics** — for each backcross-class individual, forward-backward sample `Z_{ic}` on each chromosome.
4. **Other-class posterior** — Bayes-subtraction update on `P(G_i = other | Z_i)` recorded as a running posterior summary.
5. **Autotune** (during burn-in) — adjust M-H proposal widths to target acceptance rates.

Output every `--sample-interval` iterations: posterior means/SDs for `m`, posterior class probabilities per individual, Savage-Dickey BFs for `m_{ij} = 0`.

---

## 6. Parallelism

- **OpenMP `parallel for` over chromosomes** inside the per-individual class-likelihood evaluation. Each chromosome's U(x) and FB are independent given the cache version.
- **No parallelism over individuals** in the Gibbs sweep (anchor-set updates would race). Serial sweep is fine: per-individual cost is dominated by the chromosome loop, which is the parallelized inner stage.
- **No MPI in v1.** Single-node only.

---

## 7. Validation matrix

| Test | Source of truth | What it catches |
|---|---|---|
| `prQ` on hand-built 5-locus chromosome | manual Haldane calc + mongrail2 | port correctness |
| `log_prob_subhap` on toy counts | mongrail2 | Dirichlet-Multinomial port |
| Sparse vs. full enumeration of U(x) at L=10, k_max=10 | exhaustive 2^L sum | sparse enumerator |
| Forward-backward on simulated backcross | simulator that draws Z then x\|Z | mosaic inference |
| BA3-equivalent mode | BA3 on `~/repos/BA3/examples/2pop.txt` | regression: with L=1 per chromosome and degenerate haplotype model, posteriors should match BA3 within MCMC noise |
| End-to-end on simulated K=3, N=200 | known truth migration rates | full pipeline |
| Other-class subtraction on F2 simulations | simulated F2 individuals | other-class identification |

---

## 8. Milestones

| # | Milestone | Est. duration |
|---|---|---|
| M1 | CMake skeleton; GSL/htslib link; VCF read produces per-chromosome haplotype arrays; smoke test on mongrail2 puma example | 1 wk |
| M2 | Port `prQ`, `log_prob_subhap`, `log_prob_single_hap`; full-enumeration `compute_U`. Unit tests pass against mongrail2 numerics | 2 wks |
| M3 | Replace full enumeration with sparse `ancestry_vector` iterator; verify agreement at L=10 | 1 wk |
| M4 | MCMC scaffolding: BA3 migration-rate M-H + autotune; collapsed Gibbs over G with FIXED `logP_k` (no anchor updates yet) | 2 wks |
| M5 | Online haplotype-count updates with version counter and lazy cache rebuild. Highest-risk milestone — benchmark hot loop. | 1 wk |
| M6 | Forward-backward + "other" class subtraction | 2 wks |
| M7 | Output writer extensions; Savage-Dickey BFs; per-individual class posteriors | 1 wk |
| M8 | Validation matrix complete; reconcile any discrepancy with BA3 in degenerate case | 1 wk |
| M9 | Docs, manual page, INSTALL, packaging, GitHub release | 1 wk |

Total: ~12 weeks of focused development.

---

## 9. Risks and open questions

### Known risks
- **Gibbs vs M-H behavioral change.** BA3 uses M-H + Hastings on class assignment; BA4 uses Gibbs. Validate on simulated data that Gibbs gives the expected posterior mass concentration.
- **Cache invalidation throughput.** If anchor-set membership oscillates rapidly during burn-in, `U_memo` thrashing could dominate runtime. Track cache hit/miss rates as a diagnostic.
- **Sparse-enumeration tail mass.** For chromosomes with R > 1.5 Morgans, k_max=4 leaves > 1% Poisson tail mass. Implement per-chromosome k_max selection: smallest k_max keeping `P(n > k_max | R_c) < tol` (default 0.5%). Log per-chromosome choices.
- **Numerical underflow on long chromosomes.** Log-sum-exp with running max (manuscript Appendix B) handles this; verify on synthetic L=20 high-divergence examples.

### Open questions (not blocking M1–M3)
- **Output schema for per-individual ancestry-mosaic posteriors.** BA3's output is migration rates + per-individual class. BA4 has additional latent state (Z) — write a separate file with posterior segment boundaries? TSV per individual?
- **MCMC convergence diagnostics.** BA3 has no built-in; should BA4 emit per-chain trace files for downstream diagnostic tools (e.g. CODA in R)?
- **Multi-chain support.** Currently single-chain. Worth adding `--chains N` with OpenMP across chains in v1, or leave for v2?

---

## 10. References to source material

- Theory: `~/repos/BA4-manuscript/ba4_theory.tex` (842 lines), `ba4_theory.pdf`, `ba4.bib`
- BA3: `~/repos/BA3/src/main.cpp` (3052 lines), `~/repos/BA3/include/BA3.h` (140 lines), `~/repos/BA3/Makefile`
- mongrail2: `~/repos/mongrail2/inference.c` (876 lines), `mongrail.{c,h}`, `data.c` (633), `vcf_reader.c` (411), `algorithms.c` (46), `error.c` (150)
