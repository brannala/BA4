# BA4 Implementation Plan — Code Review

Reviewer: Claude Sonnet 4.6  
Date: 2026-05-16  
Document reviewed: `PLAN.md` (BA4 Implementation Plan, v1)

---

## Overall Assessment

The plan is sound in structure and the port/write decomposition is sensible.
The highest-quality sections are the module layout, the port-vs-write map, and
the validation matrix. There are several correctness issues in the theory
mapping, one significant performance opportunity missed, and a few gaps that
will cause problems if not resolved before the relevant milestones.

---

## Section 1: Scope and Locked Decisions

### Per-chromosome `k_max` should not be global

`L_max = 20`, `k_max = 4`, one window per chromosome is correct and
well-motivated. The per-chromosome Poisson tail mass calculation in Section 9
(< 0.5% for R ≤ 1 Morgan at `k_max = 4`) should be made a formal
per-chromosome check rather than a global default, as chromosomes vary
considerably in length. This is flagged in Section 9 risks but not reflected in
the data model. Fix before M3.

### F1 class likelihood is symmetric

`F1(j→k)` and `F1(k→j)` produce the same diplotype distribution up to the
relative likelihoods of each haplotype under each population's frequency
distribution. The two assignments are distinguishable only through the relative
likelihoods of each haplotype under each population's frequency distribution.
Ensure `likelihood.cpp` sums over both assignments (both orderings of
homolog-to-population mapping) for all individuals. VCF phasing does not
indicate parental population of origin, so this applies even in phased-input
mode.

---

## Section 3: Port-vs-Write Map

### [HIGH] Prior probability of backcross is wrong

Section 1 states `P(backcross) ≈ m_{kj}²`. This is incorrect and will cause
the Gibbs update to suppress the backcross class to negligible weight for
realistic migration rates.

In the BA3 model, the offspring-of-immigrant class requires that the
individual's *parent* was a migrant and then mated locally — one migration
event, not two. The correct prior is approximately:

```
P(backcross from j in pop k) ≈ m_{kj} · (1 - Σ_{l≠k} m_{kl})
```

This is the same order of magnitude as `P(F1 from j)`. The `m_{kj}²` formula
gives prior weight of 0.0025 vs 0.05 for m ~ 0.05, incorrectly making
backcross negligible. Fix in `model/genealogy.cpp` before M4.

### Forward-backward state space is 2-state, not 4-state, for backcross

Section 3 says the forward-backward algorithm runs on `{AA, AB, BA, BB}`. For
a backcross individual, one homolog is pure by definition — the state space
collapses to `{A, B}` on the recombinant homolog alone, with the pure homolog's
state fixed. Running the full 4-state chain is wasteful and could introduce
numerical issues for the degenerate transitions involving the pure homolog.
Make this explicit in `model/forward_backward.h`. This halves the state space
and simplifies the transition matrix. Fix before M6.

### Forward-backward scope must cover all individuals for "other" posterior

Section 5 runs the forward-backward only for backcross-class individuals. But
equation (4.16) of the manuscript requires `P(Z_i | G)` evaluated under all
genealogical classes G, including pure and F1, for each individual. For
pure/F1-assigned individuals, `Z_i` is degenerate (no transitions), and
`P(Z_degenerate | backcross) = e^{-R}` — small but nonzero. The "other"
subtraction in `mcmc/other_posterior.cpp` must therefore handle all individuals.
For pure/F1-assigned individuals the forward-backward is trivial (no transitions
to sample) but the `P(Z | G)` terms must still be evaluated to form the Bayes
normalization. Add this case before M6.

---

## Section 5: MCMC Kernels

### [HIGH] Migration-rate M-H acceptance ratio recomputes the full likelihood unnecessarily

The plan states: *"Acceptance uses full data likelihood under proposed m'."*
This is unnecessarily expensive. When only `m` is updated (`G_i` and `f` are
fixed), the diplotype likelihoods `P(x_i | G_i, f^c)` are identical under `m`
and `m'` and cancel in the ratio. The acceptance ratio reduces to:

```
α = P(m') · Π_i P(G_i | m') / [P(m) · Π_i P(G_i | m)]
```

This is O(N·K²) not O(N·C·L²·k_max). For large N and C this is a 1–2 order of
magnitude difference in per-proposal cost. This is the same optimization BA3
uses. Fix in `mcmc/propose_m.cpp` and explicitly document the factorization
before M4.

---

## Section 4: Data Model

### `U_memo` keyed by individual; haplotype-keying would be better

`U_memo[N][C][k_pair]` is keyed by individual, chromosome, and population pair.
But `U(x)` depends on the specific haplotype `x`, not the individual carrying
it. Two individuals sharing the same haplotype on chromosome `c` under the same
population pair have the same `U(x)`. A haplotype-keyed cache (as in
mongrail2's `U_cache`) gives higher hit rates, is smaller, and has better
invalidation locality. For N = 200 the difference is modest, but worth
implementing correctly from the start. Flag as a design decision in
`model/u_recombinant.h`.

### Per-chromosome `k_max_c` missing from data model

The data model has a global `k_max = 4`. Add `k_max_c[C]` to the
per-chromosome state, computed at load time in `io/snp_selector.cpp` from:

```
R_c = Σ_l genetic_dist_M[c][l]
k_max_c = smallest k s.t. P(Poisson(R_c) > k) < tol   (default tol = 0.005)
```

This also affects the `logQ[C][z_idx]` cache size, which should be allocated
per chromosome as `2 · Σ_{k=0}^{k_max_c} C(L_c − 1, k)` entries.

---

## Section 7: Validation Matrix

### Missing: K=2 agreement with mongrail2

The matrix has no test checking that BA4 with K = 2, phased data, and
pure/F1/backcross classes produces the same per-individual class posteriors as
mongrail2. This is the most direct correctness check for the ported likelihood
code and should be the first golden test added. Add to M2.

### Missing: sum-to-one identity for other-class subtraction

There is no test that `Σ_G P(G_i | data) = 1` for all individuals. This is a
book-keeping identity that should hold automatically but floating-point errors
in the subtraction can cause small violations that propagate. Add as an explicit
unit test in `mcmc/other_posterior`. Add to M6.

### Missing: Gibbs vs M-H behavioral equivalence test

Section 9 flags the Gibbs-vs-M-H change as a risk but there is no test for it.
Add a comparison: run BA3-equivalent mode (validation row 5) with both the BA3
M-H class update and the BA4 Gibbs update on the same data; verify that
posterior means agree within MCMC error. This validates the Gibbs
implementation before the full haplotype model is active. Add to M4.

---

## Section 8: Milestones

### M4 with fixed `logP_k` produces incorrect posteriors — document explicitly

M4 proposes "MCMC with FIXED `logP_k` (no anchor updates yet)" and M5 adds the
online anchor-set updates. Collapsed Gibbs over `G_i` requires class posteriors
computed under the *current* anchor set. Running M4 with fixed `logP_k` means
class assignments are drawn from an incorrect conditional posterior. This is
acceptable as a debugging and timing scaffold, but M4 outputs must not be used
for any correctness validation. Make this explicit in the M4 milestone
description to avoid confusion when M4 posteriors disagree with expectations.

---

## Section 9: Open Questions

### Mosaic output schema deferred too long

The mosaic output schema is marked "not blocking M1–M3" but it will block M7
and the validation matrix if left unresolved. The forward-backward sampler must
accumulate sufficient statistics in a format consistent with the final output.
Deciding the schema after M6 is already coded forces a rewrite.

Suggested schema: a BED-like TSV file per individual, one row per posterior
ancestry segment:

```
chrom  start_bp  end_bp  P(A|segment)  P(B|segment)
```

averaged over posterior samples of `Z`. Decide and document before M6.

### Multi-chain convergence

Deferring multi-chain support to v2 is reasonable, but without at least two
independent chains there is no reliable convergence diagnostic. The
Savage-Dickey BFs in M7 are meaningful only if the chain has converged. Add
at minimum a split-chain R-hat computation (single chain split at midpoint,
cheap to implement) before the M7 milestone, or flag prominently in the manual
that users must assess convergence externally using trace files.

---

## Summary Table

| Section | Issue | Severity | Milestone |
|---|---|---|---|
| §1/§3 | `P(backcross) ≈ m²` wrong; should be `≈ m·(1−Σm)` | **High** | M4 |
| §5 | M-H acceptance recomputes full likelihood; diplotype terms cancel | **High** | M4 |
| §3 | FB state space is 2-state not 4-state for backcross | Medium | M6 |
| §3/§5 | FB scope: all individuals need `P(Z\|G)` for "other" subtraction | Medium | M6 |
| §4 | Per-chromosome `k_max_c` missing from data model | Medium | M3 |
| §7 | No K=2 vs mongrail2 golden test | Medium | M2 |
| §1 | F1(j→k) likelihood must sum over both homolog-to-population assignments | Low | M4 |
| §4 | `U_memo` keyed by individual; haplotype-keying would be better | Low | M5 |
| §8 | M4 with fixed `logP_k` gives incorrect posteriors; document explicitly | Low | M4 |
| §9 | Mosaic output schema deferred too long; decide before M6 | Low | M6 |
| §7 | No sum-to-one test for other-class subtraction | Low | M6 |
| §9 | No convergence diagnostic without multi-chain; add split R-hat | Low | M7 |
