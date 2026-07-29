# Exact Toy Models of Superposition — Explained

This is a walkthrough of the notebook **"Exploring Exact Toy Models"**, which builds
tiny, *exactly analyzable* neural networks that reproduce the "phase diagram" of
superposition from Anthropic's blog post
[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html).

Instead of training a network and reading its weights, this notebook writes down four
candidate networks by hand, computes their expected loss with pen-and-paper calculus, and
then sweeps over two parameters (data sparsity and feature importance) to see which
network wins where — reproducing the qualitative shape of the learned phase diagram
without any training at all.

---

## 1. The task

We want an autoencoder for vectors `x` in `R^n` (the notebook uses `n = dim = 3`):

```
y = ReLU(WᵀW x + b)
```

`W` maps `R^n → R^(n-1)` — the "bottleneck" is **one dimension smaller** than the input.
Since `WᵀW` is forced to have rank `n-1`, the network *cannot* represent every input
exactly. Something has to be thrown away or blended together. The question the notebook
asks is: **which coordinate(s) should be sacrificed, and how?**

**The data distribution.** Each coordinate of `x` is independently:
- `0`, with probability `1 - s`
- `Uniform(0, 1)`, with probability `s`

`s` is called the **sparsity** parameter (really, the *activation* probability — small `s`
means the vector is mostly zeros, i.e. very sparse).

**The loss.** A weighted squared error, where every coordinate has weight `1` **except
the last coordinate**, which has weight `r`:

```
loss = (x₁-y₁)² + (x₂-y₂)² + ... + (x_{n-1}-y_{n-1})² + r·(x_n-y_n)²
```

`r` is called the **relative weight** (or *importance*) of that last, special
coordinate. This mirrors the original paper's setup, where some features matter more to
the loss than others.

---

## 2. Four hand-designed candidate networks

The notebook doesn't train anything — it picks four specific, symmetric choices of `W`
and works out their loss exactly.

Two building blocks:
- **Discard-and-guess.** Zero out a coordinate, but add back its *unconditional mean*
  (`s/2`, since it's `0` with probability `1-s` and averages `0.5` when active) as a
  constant bias. This is the best *constant* guess for a coordinate you've thrown away.
- **Superposition.** Apply the 2×2 matrix
  ```
  P₂ = [ 1  -1 ]
       [-1   1 ]
  ```
  to a *pair* of coordinates, then ReLU. Geometrically this is a projection orthogonal to
  `(1,1)`. If only one of the two coordinates is active, it comes back out perfectly
  (e.g. `(x, 0) → (x, -x) → ReLU → (x, 0)`). If **both** are active at once, they
  interfere — only the *difference* survives — and that mismatch is the cost of
  superposition.

The four models:

| Name        | What it does                                                    | Bottleneck direction thrown away    |
| ----------- | --------------------------------------------------------------- | ----------------------------------- |
| **f**       | discard the **first** coordinate (weight 1), add back its mean  | 1st coordinate                      |
| **g**       | discard the **last** coordinate (weight `r`), add back its mean | last coordinate                     |
| **h_first** | **superpose** the first two coordinates                         | interference between coords 1 & 2   |
| **h_last**  | **superpose** the last two coordinates (one has weight `r`)     | interference between coords n-1 & n |

`f` and `g` are mirror images of each other, as are `h_first` and `h_last` — the only
difference is *which* coordinate carries the special weight `r`.

---

## 3. Worked example

Let's fix `dim = 3`, sparsity `s = 0.3`, relative weight `r = 2` (i.e. the last
coordinate matters twice as much as the others), and run four different sample vectors
through all four models. In 3 dimensions, `x = (x₁, x₂, x₃)`, where `x₃` is the specially
weighted coordinate.

*(`x₂`, the middle coordinate, is never discarded/superposed by `f` or `g`, but it does
get superposed with `x₁` in `h_first` or with `x₃` in `h_last`.)*

### Case A — only the (unweighted) middle coordinate is active: `x = (0, 0.55, 0)`

| model                   | output y          | loss       |
| ----------------------- | ----------------- | ---------- |
| f (discard first)       | `(0.15, 0.55, 0)` | 0.0225     |
| g (discard last)        | `(0, 0.55, 0.15)` | 0.0450     |
| h_first (superpose 1,2) | `(0, 0.55, 0)`    | **0.0000** |
| h_last (superpose 2,3)  | `(0, 0.55, 0)`    | **0.0000** |

Nothing collides here, so both superposition models reconstruct perfectly, while `f`
and `g` still pay a small penalty for guessing at a coordinate that happens to be zero
(they don't know that in advance).

### Case B — only the first coordinate is active: `x = (0.72, 0.10, 0)`

| model                   | output y             | loss       |
| ----------------------- | -------------------- | ---------- |
| f (discard first)       | `(0.15, 0.10, 0)`    | 0.3249     |
| g (discard last)        | `(0.72, 0.10, 0.15)` | 0.0450     |
| h_first (superpose 1,2) | `(0.62, 0, 0)`       | 0.0200     |
| h_last (superpose 2,3)  | `(0.72, 0.10, 0)`    | **0.0000** |

`f` pays heavily here — it threw away exactly the coordinate that was active.
`h_first` collides `x₁` with the (small) `x₂`, but the damage is small since `x₂` is
small. `h_last` is untouched since it only entangles coordinates 2 and 3.

### Case C — only the last (weighted) coordinate is active: `x = (0, 0.10, 0.85)`

| model                   | output y             | loss       |
| ----------------------- | -------------------- | ---------- |
| f (discard first)       | `(0.15, 0.10, 0.85)` | 0.0225     |
| g (discard last)        | `(0, 0.10, 0.15)`    | **0.9800** |
| h_first (superpose 1,2) | `(0, 0.10, 0.85)`    | **0.0000** |
| h_last (superpose 2,3)  | `(0, 0, 0.75)`       | 0.0300     |

`g` is disastrous — it discarded the coordinate that matters twice as much, *and* it was
active. This is the key asymmetry that makes `r` matter.

### Case D — both special coordinates active (a "collision"): `x = (0.72, 0.10, 0.85)`

| model                   | output y             | loss   |
| ----------------------- | -------------------- | ------ |
| f (discard first)       | `(0.15, 0.10, 0.85)` | 0.3249 |
| g (discard last)        | `(0.72, 0.10, 0.15)` | 0.9800 |
| h_first (superpose 1,2) | `(0.62, 0, 0.85)`    | 0.0200 |
| h_last (superpose 2,3)  | `(0.72, 0, 0.75)`    | 0.0300 |

Here `h_first` is cheapest: it's better to let the two *unweighted* coordinates
interfere a little than to mishandle the weighted one at all.

**Takeaway from the worked example:** at `s = 0.3, r = 2`, `h_first` wins in 2 of these
4 illustrative cases and is competitive in the others — and, as the phase diagram below
confirms, `h_first` is in fact the globally best model *on average* at this exact
`(s, r)` point, even though `r = 2` favors protecting the last coordinate. Superposition
of the two *cheap* coordinates can beat protecting the *one expensive* coordinate, if
sparsity is only moderate. This is a nice, slightly counter-intuitive result that the
phase diagram makes precise.

---

## 4. Exact expected loss (closed form)

Instead of Monte-Carlo estimating the loss, the notebook derives it exactly by
integrating over the sparsity distribution.

For `f` and `g`, the loss only depends on the single discarded coordinate: with
probability `1-s` it's zero (loss = bias²), with probability `s` it's `Uniform(0,1)`
(an integral of `(x - s/2)²`):

```
loss(f_s) = (1-s)(s/2)² + s ∫₀¹ (x - s/2)² dx  =  s/3 - s²/4
loss(g_s) = r · loss(f_s)                       (same shape, weighted by r)
```

For `h_first`/`h_last`, the reconstruction is only imperfect when **both** superposed
coordinates are active at once (probability `s²`), and then the error is
`2·min(x,y)²` (or `(1+r)·min(x,y)²` if one of the two carries weight `r`):

```
loss(h_first) = 2s² ∫₀¹∫₀ˣ y² dy dx        = s²/3
loss(h_last)  = 2s² ∫₀¹∫₀ˣ (1+r) y² dy dx  = (1+r)·s²/6
```

These four formulas are the heart of the notebook — everything else (the phase diagram,
the 3D surface, the small-multiple plots) is just evaluating and comparing these four
functions of `(r, s)` over a grid.

**Sanity check.** Running 1,000,000 randomly sampled vectors through each model at
`s = 0.3, r = 2` and averaging the *actual* squared error matches the formulas almost
exactly:

| model | closed-form | Monte Carlo (N=1,000,000) |
|---|---|---|
| f (discard first) | 0.077500 | 0.077383 |
| g (discard last) | 0.155000 | 0.154968 |
| h_first (superpose first two) | 0.030000 | 0.029881 |
| h_last (superpose last two) | 0.045000 | 0.045064 |

(Small differences are just sampling noise.) This confirms `h_first` really is the best
of the four at this point, as seen in the worked example above.

---

## 5. The phase diagram

For a grid of `(relative weight r, sparsity s)` pairs, we evaluate all four closed-form
losses and record **which one is smallest**. Plotting the winning model as a color at
each grid point produces the "phase diagram": a map of which reconstruction strategy is
optimal, as a function of how important a coordinate is (`r`) and how often it's
non-zero (`s`).

The sweep (matching the notebook): `s` ranges over `10⁰` down to `10⁻³` (dense → very
sparse), `r` ranges over `10⁻²` up to `10²` (the special coordinate can be 100x less or
100x more important than the rest).

**Qualitative regions** (as described in the notebook, and visible in the dashboard
below):

- **`r > 1` (special coordinate matters more):** the diagram splits into two shades —
  `g`/discard-last is preferred at low sparsity (few collisions to worry about, so just
  protect the important coordinate perfectly by keeping it in its own dimension via
  `h_last`... in practice `h_last` dominates once superposition becomes cheap enough),
  while `h_last` (protecting the coordinate via superposition rather than outright
  discard) wins as sparsity grows, since discard-and-guess (`g`) gets worse relative to
  superposition as data gets denser.
- **`r < 1` (special coordinate matters less):** a similar split between `f`
  (discard-first) and `h_first` (superpose-first), driven mostly by sparsity — more
  collisions (`s²` term) make discarding relatively more attractive, less sparsity favors
  superposition.
- **Near `r ≈ 1` and moderate `s`:** the four regions meet, and — as the worked example
  shows — the winner can be somewhat counter-intuitive (e.g. `h_first` beating `h_last`
  even when `r > 1`), because the `s²` interference cost of superposing the *unweighted*
  pair can be cheaper than any option that touches the *weighted* coordinate.

---

## 6. What's in the interactive dashboard

Running the accompanying script produces **`phase_diagram_dashboard.html`**, a
self-contained interactive page (Plotly, no internet connection needed beyond the one
CDN script tag) with three linked views of the same grid:

1. **Phase diagram (2D heatmap).** Which of the four models wins at each
   `(r, s)` — hover any pixel to see the *exact* loss of all four models there, not just
   the winner. A marker shows the `(s=0.3, r=2)` worked example from this write-up.
2. **Least-loss surface (3D).** The `log` of the winning loss, as a surface over
   `log₁₀(s)` and `log₁₀(r)` — this is the "least loss among explicit models" plot from
   the notebook, but interactive (rotate/zoom) instead of a static matplotlib view.
3. **Per-model log-loss (small multiples).** The same grid, but showing each of the four
   models' loss on its own — this makes clear that `f`'s loss is basically controlled by
   `r` alone, `h`'s loss is controlled by `s` alone (`r` barely matters, mathematically),
   and `g` is the more complex mix of the two.

---

## 7. Files produced

| file | contents |
|---|---|
| `toy_models_explained.md` | this write-up |
| `toy_model_phase_diagram.py` | standalone script: model definitions, worked example, Monte-Carlo check, phase-diagram sweep, and Plotly dashboard generation |
| `phase_diagram_dashboard.html` | the interactive Plotly dashboard described in §6 |

To regenerate everything from scratch:

```bash
pip install numpy plotly
python toy_model_phase_diagram.py
```

This prints the worked example and Monte-Carlo check to the console, and writes a fresh
copy of `phase_diagram_dashboard.html`.

---

## 8. Relationship to the original blog post

The original [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
post *trains* small ReLU networks on synthetic sparse data and observes, empirically,
that the learned weights fall into different qualitative regimes (dedicating a
dimension to one feature, sharing a dimension between correlated/anti-correlated
features, ignoring unimportant features entirely) depending on sparsity and feature
importance.

This notebook's contribution is to show that a **much smaller, hand-picked, and exactly
solvable** family of networks already reproduces the qualitative shape of that phase
diagram — without any training or gradient descent, just calculus. That's a nice
existence proof that the *phenomenon* (which representational strategy wins depends on
sparsity and importance) isn't a training artifact — it falls directly out of the
geometry of the loss function itself.




# Superposition as a Phase Change

A companion write-up to `toy_models_explained.md`, focused specifically on the claim
that superposition behaves like a **phase transition**: as sparsity and relative
feature importance vary, the model's preferred encoding strategy doesn't deform
smoothly — it *snaps* between qualitatively different configurations, with a sharp
boundary between them.

---

## 1. The observation

When a network is forced to compress more features than it has dimensions, a given
feature seems to end up in one of three states:

1. **Dropped.** The feature isn't represented at all.
2. **Superposed.** The feature shares a dimension with one or more other features,
   at the cost of some interference/collision error when they're active together.
3. **Dedicated.** The feature gets its own private dimension, represented cleanly.

What's striking is that as you smoothly vary the training conditions (how often a
feature is active, how much it matters to the loss), the *boundaries* between these
three regimes look sharp rather than gradual — suggestive of a phase transition, the
same way water doesn't gradually "soften" into steam but flips abruptly at 100°C.

To test that idea directly, rather than trying to read it off of a large trained
model (where many features shift at once and effects tangle together), it helps to
strip the problem down to the smallest case where it can be studied in isolation:
just one or two features fighting over one spare dimension. Small enough that we can
solve for the optimal encoding **exactly**, with calculus, rather than just observing
what gradient descent happens to find.

---

## 2. Case 1 — two features sharing one dimension

**Setup.** Two features `x₁, x₂`, each independently either `0` (with probability
`1-s`) or `Uniform(0,1)` (with probability `s`). Feature 1 has importance `1`;
feature 2 (the "extra" feature) has importance `r`, swept anywhere from `0.1` to `10`.
The model is `y = ReLU(WᵀW x - b)` with `W` a `1×2` matrix — a single hidden unit
standing in for two input features.

*(Note on terminology: following the same convention used throughout, `s` is the
probability a feature is **active** — so despite being called the "sparsity"
parameter, a **smaller** `s` means **sparser** data. The sweep goes from `s=1`,
dense, down to `s=0.01`, quite sparse.)*

There are exactly three geometrically natural choices of `W`:

| `W` | strategy | what happens |
|---|---|---|
| `[1, 0]` | **ignore-extra** | keep feature 1 perfectly, throw feature 2 away |
| `[0, 1]` | **dedicate** | throw feature 1 away, keep feature 2 (the extra, importance `r`) perfectly |
| `[1, -1]` | **antipodal** | store both features in the *same* dimension, in opposite directions — recovers either one perfectly on its own, but the two interfere if both are active simultaneously |

"Antipodal" is the superposition solution: geometrically, the two basis directions
`(1,0)` and `(0,1)` get mapped to opposite points, so the network can perfectly
recover whichever one is active — as long as it's only one at a time.

### Deriving the three losses exactly

For **ignore-extra**, feature 1 comes back exactly as `x₁`. Feature 2 is replaced by
a constant guess `b = s/2` (its unconditional mean — zero with probability `1-s`,
averaging `0.5` when active). The expected squared error on feature 2, weighted by
its importance `r`, works out to a simple integral:

```
loss(ignore-extra) = r · [ (1-s)(s/2)² + s∫₀¹(x - s/2)² dx ]
                    = r · (s/3 - s²/4)
```

**Dedicate** is the mirror image — now it's feature 1 (importance 1, not `r`) that
gets thrown away:

```
loss(dedicate) = 1 · (s/3 - s²/4) = s/3 - s²/4          (doesn't depend on r at all!)
```

**Antipodal** reconstructs perfectly unless *both* features are active at once
(probability `s²`), in which case the interference error is `(1+r)·min(x₁,x₂)²`:

```
loss(antipodal) = 2s² ∫₀¹∫₀ˣ (1+r) y² dy dx = (1+r)·s²/6
```

All three losses are just **straight lines in `r`** (for fixed `s`) — that fact turns
out to matter a lot in a moment.

### Where do the regimes cross over?

**Ignore-extra vs. dedicate.** Setting `r·(s/3 - s²/4) = (s/3 - s²/4)` and cancelling
the (positive) common factor gives simply:

```
r = 1
```

...for *every* value of `s`. That's a clean, sparsity-independent rule: whichever of
the two features is intrinsically less important is the one you should sacrifice, if
you're choosing between "keep one, drop the other."

**Ignore-extra vs. antipodal** (relevant when `r < 1`, i.e. antipodal is competing
against the "drop the extra feature" option): solving
`r·(s/3 - s²/4) = (1+r)·s²/6` for `r` gives

```
r_c1(s) = 2s / (4 - 5s)
```

**Dedicate vs. antipodal** (relevant when `r > 1`): solving
`s/3 - s²/4 = (1+r)·s²/6` for `r` gives

```
r_c2(s) = 2/s - 2.5
```

### A triple point

Something nice happens if you ask: at what sparsity does the "antipodal beats
ignore-extra" boundary hit exactly `r=1`? Set `r_c1(s) = 1`:

```
2s/(4-5s) = 1   ⟹   s = 4/7 ≈ 0.571
```

Solving `r_c2(s) = 1` gives the *identical* answer, `s = 4/7`. At this single point
— `r = 1`, `s = 4/7` — **all three strategies have exactly the same loss**
(`16/147 ≈ 0.1088`, checked directly by substitution). This is a genuine **triple
point**, the same kind of feature you see on a physical phase diagram where solid,
liquid, and gas boundaries all meet at one exact temperature and pressure.

It also gives a hard threshold: for `s > 4/7` (i.e. the feature is active more than
~57% of the time), superposition is **never** the optimal strategy, no matter how you
tune `r`. Below that density, there's always some window of importance ratios where
antipodal wins. This is a sharper, quantitative version of "sparsity is a
prerequisite for superposition."

### Why this really is a phase transition, not just a crossing

Physically, a system's free energy is the *minimum* over the free energies of every
candidate phase — whichever phase is cheapest is the one the system actually adopts.
Where two candidate phases' free energies cross, the derivative of the *overall*
minimum has a kink: a **first-order transition**, the same mathematical structure
behind the latent heat of melting ice.

The toy model here has exactly that structure. The optimal loss, as a function of `r`
at fixed `s`, is:

```
loss*(r) = min( loss(ignore-extra), loss(antipodal), loss(dedicate) )
```

— the lower envelope of (up to) three straight lines. Two straight lines with
different slopes crossing always produces a "V" — a kink, a discontinuous first
derivative — right at the crossover. And critically, the *configuration* itself
(which `W` is optimal) doesn't interpolate smoothly between `[1,0]`, `[1,-1]`, and
`[0,1]` as `r` or `s` cross a boundary — it **jumps** discretely from one exact matrix
to another. That discrete jump in the "order parameter" ( the weight configuration
itself ) alongside a kinked loss curve is precisely the signature of a first-order
phase transition.

---

## 3. Case 2 — three features sharing two dimensions

The same experiment generalizes one step further: three features (importance `1, 1,
r` — two "ordinary" features and one "extra" one) sharing only **two** hidden
dimensions. Now the network has one direction it can afford to throw away, and
there are four (not three) natural candidates, described by *which* direction gets
discarded:

| discarded direction | strategy | meaning |
|---|---|---|
| extra feature's own axis | drop the extra feature entirely | the extra feature is unrepresented |
| one ordinary feature's axis | drop that ordinary feature | the extra feature gets a clean, dedicated axis |
| the *sum* of the extra feature and one ordinary feature | antipodal-pair the extra feature with an ordinary one | the remaining ordinary feature gets its own axis |
| the *sum* of the two ordinary features | antipodal-pair the two ordinary features together | the extra feature gets a clean, dedicated axis |

(A fifth option — superposing all three features jointly, rather than just a pair —
isn't considered here; with only one spare direction to discard, a fully joint
three-way interference pattern doesn't fit the same simple analysis.)

**This is exactly the model worked through in `toy_models_explained.md`.** The
mapping is direct:

| here | earlier notebook name | closed-form loss |
|---|---|---|
| drop the extra feature | `g` (discard last) | `r·(s/3 - s²/4)` |
| drop an ordinary feature | `f` (discard first) | `s/3 - s²/4` |
| pair extra with one ordinary feature | `h_last` (superpose last two) | `(1+r)·s²/6` |
| pair the two ordinary features | `h_first` (superpose first two) | `s²/3` |

Exactly the same kind of pairwise crossover analysis applies — each pair of these
four loss functions crosses along some curve in `(s, r)` space, and the true "phase
diagram" is the map of which of the four functions is smallest at each point. That
map is precisely what the interactive dashboard (`phase_diagram_dashboard.html`,
produced alongside this write-up) plots directly, including the exact losses at any
point on hover.

---

## 4. Key insights

- **Sparsity is necessary but not sufficient.** Whether superposition wins depends
  jointly on sparsity *and* relative importance — there's a genuine two-parameter
  boundary, not a fixed sparsity cutoff. (Though in the two-feature toy model above,
  there is a hard sparsity ceiling — `s > 4/7` — past which superposition can never
  win regardless of importance.)
- **The transition is first-order.** Because each candidate strategy's loss is a
  simple (locally linear-ish) function of the parameters, the *overall* optimal loss
  is a lower envelope with real kinks at the crossovers — not a smooth minimum. The
  optimal weight configuration jumps discretely between a small number of exact
  matrices rather than deforming continuously.
- **Cheaper features get sacrificed first.** In the two-feature model, the
  ignore-vs-dedicate boundary sits at exactly `r = 1`, for *any* sparsity — if you
  must fully drop one of two features, always drop the less important one. That
  boundary is sparsity-independent; only the *width of the superposition region
  around it* depends on sparsity.
- **A triple point exists.** At `r = 1, s = 4/7` in the two-feature model, all three
  strategies are exactly tied — mathematically the same kind of coincidence as the
  triple point on a material's phase diagram, where three phase boundaries meet.
- **The pattern scales up cleanly.** Moving from two features/one dimension to three
  features/two dimensions doesn't change the logic — it just adds more candidate
  discrete configurations (four instead of three) and more pairwise crossover curves,
  building the richer phase diagram already explored in the companion notebook
  write-up.
- **This is still a simplified picture.** The full space of trained-model behavior
  has more structure than this minimal, hand-picked set of candidate solutions
  captures — things like correlations between features, or richer interference
  patterns among larger groups of features, aren't represented by just three or four
  discrete configurations. This toy analysis is best read as an existence proof that
  the *phase-change character* of superposition falls directly out of the geometry
  of the loss function, not as the complete story.

---

## 5. Loss formulas, side by side

| model | strategy | closed-form loss |
|---|---|---|
| 2 features, 1 dim: ignore-extra | drop feature 2 (weight `r`) | `r(s/3 - s²/4)` |
| 2 features, 1 dim: dedicate | drop feature 1 (weight `1`) | `s/3 - s²/4` |
| 2 features, 1 dim: antipodal | superpose both | `(1+r)s²/6` |
| 3 features, 2 dims: `g` | drop the extra feature | `r(s/3 - s²/4)` |
| 3 features, 2 dims: `f` | drop an ordinary feature | `s/3 - s²/4` |
| 3 features, 2 dims: `h_last` | superpose extra + one ordinary | `(1+r)s²/6` |
| 3 features, 2 dims: `h_first` | superpose the two ordinary features | `s²/3` |

Notice the 2-feature and 3-feature cases share three identical formulas — adding a
third feature (with its own dedicated axis in the `h_first`/"pair the two ordinary
features" solution) simply adds one new option to the competition, without changing
the other three.




Here is the full explanation, restated clearly from the top.

---

### 1. Uniform Superposition
- **Definition**: Every feature has the same importance ($I_i=1$) and the same sparsity $S$ (probability of being active).
- **Why it is studied first**: It removes asymmetry, leaving only sparsity and geometry to vary. Reasoning is much simpler.
- **Scale parameters**: $n=400$ features, $m=30$ hidden dimensions.
- **Scaling laws**:
  - $n \gg m$ means the absolute $n$ does not matter.
  - $m$ only sets scale: doubling $m$ doubles the number of features learned. The interesting quantity is the **ratio** of learned features to $m$.

---

### 2. Counting Learned Features with $\|W\|_F^2$
The columns $W_i$ are the feature embeddings.

- **Represented feature**: $\|W_i\|^2 \approx 1$
- **Not represented**: $\|W_i\|^2 \approx 0$

So:
$$\|W\|_F^2 \approx \text{number of features the model actually learned}$$

This is **basis-independent**, so it is valid even in the dense regime ($S=0$) where no standard feature basis is privileged.

---

### 3. Aggregate Metric: $D^* = m / \|W\|_F^2$
This measures **"hidden dimensions consumed per feature"** (dimensions per feature).

- $D^* \approx 1$ $\Rightarrow$ one feature per dimension.
- $D^* \approx 1/2$ $\Rightarrow$ two features share one dimension.

**Observed plot behavior**: The curve is "sticky" (plateaus) at exactly **1** and **1/2**, resembling the step-like plateaus of the Fractional Quantum Hall Effect ($\sigma_{xy}$ vs. $\nu$).

---

### 4. The $1/2$ Plateau: Antipodal Pairs
This is a precise geometry:

- Features come in pairs: $W_k = -W_j$ (exact negatives).
- Both live in **the same hidden dimension**.
- Therefore two features consume one dimension $\Rightarrow$ average $1/2$ dimension per feature.

These antipodal pairs are so efficient that the model uses them preferentially over a wide sparsity range, causing the curve to "stick" at $1/2$.

---

### 5. The Iceberg Underneath
Antipodal pairs are only the visible case. Beneath the smooth $D^*$ curve are **many other discrete geometric packings** (triplets, specific angles, etc.). The model does not distribute capacity continuously; it packs features into discrete, optimal arrangements.

---

### 6. Per-Feature Dimensionality $D_i$
To see the hidden packings individually, define for each feature $i$:

$$D_i = \frac{\|W_i\|^2}{\sum_j (\hat{W}_i \cdot W_j)^2}$$

Where:
- $W_i$ = weight column for feature $i$.
- $\hat{W}_i = W_i / \|W_i\|$ = unit direction.

**What it means**:
- **Numerator**: How strongly feature $i$ is represented.
- **Denominator**: How many features share that direction. It sums squared projections of all $W_j$ onto $\hat{W}_i$.

**Antipodal example**:
- Projects onto itself: $1$
- Projects onto partner ($-W_i$): $(-1)^2 = 1$
- Denominator = $1 + 1 = 2$
- Numerator $\approx 1$
- So $D_i = 1/2$

**Unlearned feature**: $\|W_i\| \approx 0 \Rightarrow D_i \approx 0$

**Conservation**: For efficient packing, $\sum_i D_i \approx m$.

---

### 7. Breaking the Curve into Individual Points
Instead of just plotting the average $D^*$, you plot every feature's $D_i$ at each sparsity level.

- The points **cluster at rational fractions**: $1, 1/2, 1/3, \dots$
- Each cluster corresponds to one specific weight geometry (antipodal pair, triplet sharing a dimension, etc.).
- Lines are drawn at these fractions to show the quantized structure.

---

### 8. Weight Geometry Graph
To visualize the actual embedding layout:

- **Nodes**: individual features.
- **Edges**: present if the feature vectors are not orthogonal ($\hat{W}_i \cdot W_j \neq 0$).
- **Edge weight**: $|\hat{W}_i \cdot W_j|$ (absolute cosine similarity).

This reveals clusters:
- **Antipodal pairs**: two nodes with strong edge weight $\approx 1$ but opposite signs.
- **Orthogonal features**: no edge.
- **Other packings**: distinct clique/star patterns depending on how many features share a dimension.

---

### Bottom Line
In the uniform case, the model does not smoothly interpolate between representations. It **quantizes** hidden space: features are packed into discrete, highly efficient geometric configurations (antipodal pairs being the most prominent). This causes both the aggregate curve $D^*$ and the individual dimensionalities $D_i$ to stick at simple fractions like $1$ and $1/2$.


This is a beautifully rich set of observations, and you’re right to want a clear, step-by-step unpacking. Let’s take each concept in turn, building from the basic idea of uniform superposition up to the idea of multiple “phases” within superposition itself.

---

### 1. The Setting: Uniform Superposition

We’re in a simplified model where:

*   There are `n` features (e.g., `n=400`), but only `m` hidden dimensions (`m=30`) to represent them. The bottleneck is severe: `n ≫ m`.
*   All features are **equally important** (importance `I_i = 1`) and have the same sparsity `S`. This is the “uniform” assumption.
*   The model learns a weight matrix **W** of shape `(m, n)`. Each column `W_i` (an `m`-dimensional vector) is the embedding for feature `i`. Roughly, a feature is “learned” if `||W_i|| ≈ 1`, and ignored if `||W_i|| ≈ 0`.

Because `n` is much larger than `m`, the model cannot give every feature its own orthogonal direction. It must **superpose** multiple features into the same low‑dimensional space, relying on sparsity (the fact that not all features are active at once) to avoid interference.

---

### 2. The Global Metric: Dimensions per Feature `D*`

A convenient summary statistic is  
`D* = m / ||W||_F^2`,  
where `||W||_F^2` is the sum of squared entries of **W** (the squared Frobenius norm).

*   If every feature gets its own dedicated orthogonal dimension, we’d have roughly `m` learned features, each with `||W_i||² ≈ 1`, so `||W||_F^2 ≈ m` and `D* ≈ 1`.
*   If the model learns `k` features (each with norm ≈ 1) and ignores the rest, `||W||_F^2 ≈ k`, so `D* ≈ m/k`. Thus `D*` can be seen as the **average number of hidden dimensions devoted to one learned feature**. The smaller `D*`, the more features are packed into the `m` dimensions.

The sticky points at **1** and **½** in the `D*` curve mean that, over a wide range of sparsity, the model prefers to operate with exactly one feature per dimension (`D* = 1`, no superposition) or with exactly two features sharing one dimension (`D* = ½`). The latter is a particularly efficient packing.



# Dimensions per Feature: Sample `W` Matrices

## The metric

A convenient summary statistic for how a model is packing features into hidden dimensions is

```
D* = m / ||W||_F²
```

where `m` is the number of hidden dimensions and `||W||_F² = Σ W_ij²` is the squared Frobenius norm of the weight matrix `W ∈ R^(n×m)` (`n` features, `m` hidden dimensions; each **row** `W_i ∈ R^m` is one feature's embedding).

- If every feature gets its own dedicated orthogonal dimension, there are roughly `m` learned features, each with `||W_i||² ≈ 1`, so `||W||_F² ≈ m` and `D* ≈ 1`.
- If the model learns `k` features (each with norm ≈ 1) and ignores the rest, `||W||_F² ≈ k`, so `D* ≈ m/k`.

Hence `D*` is the **average number of hidden dimensions devoted to one learned feature**: the smaller `D*`, the more features are packed into the `m` available dimensions.

---

## Case 1: `D* = 1` — one dedicated dimension per feature

With `n = 3` features and `m = 3` hidden dimensions, each feature mapped to its own orthonormal direction:

```
                hidden1  hidden2  hidden3
   feature 1  [    1        0        0   ]
   feature 2  [    0        1        0   ]
   feature 3  [    0        0        1   ]
```

```
||W_dedicated||_F² = 1² × 3 = 3
D* = m / ||W||_F² = 3 / 3 = 1
```

---

Case 2: `D* = 1/2` — antipodal pairs

With `n = 6` features sharing only `m = 3` hidden dimensions, two features per dimension pointing in opposite directions (`+1` / `-1`):

```
                hidden1  hidden2  hidden3
   feature 1  [    1        0        0   ]
   feature 2  [   -1        0        0   ]
   feature 3  [    0        1        0   ]
   feature 4  [    0       -1        0   ]
   feature 5  [    0        0        1   ]
   feature 6  [    0        0       -1   ]
```

```
||W_antipodal||_F² = 1² × 6 = 6
D* = m / ||W||_F² = 3 / 6 = 1/2
```

Each hidden dimension now reconstructs whichever of its two features is active perfectly, and only pays an interference cost when both are active simultaneously — twice the features packed into the same space, at the cost of occasional collisions.

---

## Remark

A real trained `W` will not be this clean (row norms won't be exactly `1`, off-diagonal entries won't be exactly `0`), but `D*` still lands close to these values because the network converges on the same qualitative arrangement: whole dimensions dedicated to single features, or split into antipodal (or higher-order, e.g. triangular or tetrahedral) groups.
---

### 3. The ½ Sticky Point: Antipodal Pairs

The `D* = ½` regime corresponds to a very specific geometry: **antipodal pairs**. Two features are represented by opposite vectors in the same hidden dimension, e.g. `W_a = e` and `W_b = –e`, where `e` is a unit vector.

Why is this so effective?

*   In many architectures (those with a ReLU nonlinearity), negative activations are zeroed out. So a positive input for feature `a` activates `e`, while a positive input for feature `b` activates `–e`, which ReLU sets to zero in the positive direction of `e`. Thus the two features never interfere when they are both active – as long as the model’s activation function is ReLU, the representation cleanly separates them by sign.
*   Geometrically, two features pack perfectly into a 1‑dimensional subspace without increasing the norm budget. The total squared norm contributed by the pair is `1² + 1² = 2`, so two features consume only one dimension’s worth of “norm capacity”. That’s exactly what `D* = ½` captures.

The model discovers this trick by itself and clings to it across a wide range of sparsities – hence the sticky plateau.

---

### 4. Per‑Feature Dimensionality `D_i`

The global `D*` only gives an average. To see what each individual feature is doing, we define the per‑feature dimensionality:

```
D_i = ||W_i||²  /  Σ_j (Ŵ_i · W_j)²
```

where `Ŵ_i` is the unit vector in the direction of `W_i`.

Let’s unpack the intuition:

*   The numerator `||W_i||²` is just the squared norm – it measures how strongly feature `i` is represented.
*   The denominator sums the **squared projections** of all feature vectors onto the direction of `Ŵ_i`. In other words, it asks: “If I look along the line defined by feature `i`, how many features (including `i` itself) are ‘using’ this direction?”
    *   If `W_i` is orthogonal to all other features, the sum is just `(Ŵ_i·W_i)² = ||W_i||²`, so `D_i = 1`. The feature has a full dimension to itself.
    *   If `W_i` shares its direction exactly with another feature (the antipodal case), the denominator gets two equal contributions: `||W_i||²` from itself and `||W_i||²` from the antipodal partner. Then `D_i = 1/2`.
    *   If a feature is not learned (`||W_i|| ≈ 0`), both numerator and denominator vanish, but the limit gives `D_i = 0`.

In essence, `D_i` is the **effective fraction of a dimension** that feature `i` occupies, taking into account how many other features are packed along the same line or in the same subspace.

---

### 5. Clustering at Specific Fractions: The “Crystal” Structures

When you compute `D_i` for every feature across many models with different sparsities, the individual values don’t just take a continuous spectrum. They **cluster strongly at certain rational numbers**: 1, 1/2, 2/3, 3/4, 1/3, etc. Each such fraction signals a regular, symmetric geometric arrangement of feature vectors.

Here’s why:

*   **Antipodal pair (1/2):** As above, two features exactly opposite. Sum of `(Ŵ_i·W_j)²` includes 1 (self) + 1 (antipode) = 2, so `D_i = 1/2`.

*   **Three coplanar features at 120° (2/3):** Imagine three vectors in a 2D plane, equally spaced. They all have the same norm. For unit vectors, the dot product between any pair is `cos 120° = –1/2`. Then for any one feature, the denominator includes:
    *   1² = 1 (from itself),
    *   2 × (1/2)² = 2 × 1/4 = 1/2 (from the other two).
    Sum = 1.5. So `D_i = 1 / 1.5 = 2/3`. The three features together consume exactly 2 dimensions (`3 × 2/3 = 2`), which matches the subspace they lie in.

*   **Regular tetrahedron (3/4):** Four vectors in 3D pointing to the vertices of a regular tetrahedron. For unit vectors, the dot product between any pair is `–1/3`. The denominator for one feature is `1 + 3 × (1/3)² = 1 + 3/9 = 4/3`. Thus `D_i = 3/4`. Four features pack into 3 dimensions.

*   **Other fractions** correspond to other regular polytopes or highly symmetric configurations (e.g., a 2D square with features at 90° yields `D_i = 1` because they’re orthogonal – that’s not superposition; but a 2D regular pentagon would yield yet another fraction).

What the scatter plots and the “feature geometry graph” reveal is that the model spontaneously discovers these **specific discrete packings** and jumps from one to another as sparsity changes. The edges in the geometry graph connect features whose embedding vectors are not orthogonal – and the graph visualizations often show beautiful regular structures (pairs, triangles, tetrahedra, etc.).

---

### 6. The Phase Change Analogy: Many Phases of Superposition

Earlier in the research, superposition itself was framed as a **phase change**: below a critical sparsity, the model abruptly goes from representing features in dedicated dimensions (no superposition) to packing them together (superposition). That’s analogous to a substance changing from liquid to solid.

The new insight is that **superposition is not one monolithic phase**. It contains many distinct sub‑phases, each characterized by a different discrete geometric arrangement of feature embeddings. These sub‑phases correspond to the fractional cluster points:

| Dimensionality Fraction | Geometric Packing | Analogy |
|-------------------------|-------------------|---------|
| 1 | Orthogonal, no superposition | Liquid (disordered) |
| 1/2 | Antipodal pairs | Ice‑Ih (hexagonal) |
| 2/3 | Equilateral triangle in 2D | Ice‑II (rhombohedral) |
| 3/4 | Tetrahedron in 3D | Ice‑III (tetragonal) |
| … | other regular configurations | … |

Just as water can freeze into many different crystal structures (ice‑Ih, ice‑II, ice‑III, …) depending on temperature and pressure, the model’s representation of features “crystallizes” into different discrete geometries depending on the sparsity `S`. The sticky plateaus in the average `D*` curve arise because certain packings (like antipodal pairs) are especially robust and cover a wide range of sparsity before the model switches to a different configuration.

---

### Summing It All Up

- **Uniform superposition** simplifies the problem, letting us focus on how models pack equally important, equally sparse features.
- The **average dimension per feature** `D*` reveals global packing efficiency, with striking sticky points at 1 and 1/2.
- **Per‑feature dimensionality** `D_i` measures the effective fraction of a dimension each feature uses, given the competition along its own direction.
- The clustering of `D_i` at specific fractions (1/2, 2/3, 3/4, etc.) reflects the model’s discovery of highly symmetric, **discrete geometric configurations** – the “crystal structures” of superposition.
- This turns superposition from a single phase into a **rich phase diagram** with multiple ordered phases, each corresponding to a different regular arrangement of feature vectors.

In short, the model isn’t just randomly squeezing features together; it’s finding elegant, mathematically optimal packings that depend on the degree of sparsity, much like nature finds optimal atomic arrangements under different conditions.





The core mathematical phenomenon here is the equivalence between **point configurations** (polytopes) and **Gram matrices** of the form \(W^T W\), where the columns of \(W\) are the coordinates of the points. This is a two-way bridge:

- **Geometry → Matrix:** Any set of \(n\) points in \(\mathbb{R}^m\) gives a rank-\(m\), \(n\times n\) positive semidefinite matrix whose entries are the dot products between points.
- **Matrix → Geometry:** Any rank-\(r\) positive semidefinite matrix of size \(n\times n\) can be factorised as \(W^T W\) with \(W\) an \(r\times n\) matrix; the columns of \(W\) are an \(n\)-point configuration in \(\mathbb{R}^r\) whose Gram matrix is exactly that matrix.

In the context of superposition (representing \(n\) features in an \(m\)-dimensional model, with \(m < n\)), the columns of \(W\) are the embedding vectors of the features, and the matrix \(M = W^T W\) captures **how much the features interfere**:
- Diagonal entries \(M_{ii} = \|w_i\|^2\) measure a feature’s individual strength.
- Off-diagonal entries \(M_{ij} = \langle w_i, w_j\rangle\) are the interference between feature \(i\) and feature \(j\). Non-zero off-diagonals mean the features are not orthogonal; they “share” representational space.

So **every strategy for embedding \(n\) features in \(m\) dimensions is precisely a choice of \(n\) points in \(\mathbb{R}^m\)**, and the polytope (e.g. a triangle for 3 points in 2D) is just the convex hull of those points. The geometry of the polytope completely determines the interference pattern. Symmetric, well-spaced configurations (like an equilateral triangle) correspond to uniform off-diagonal entries, meaning all pairs of features interfere equally – often the optimal trade-off when features are equally important and sparse.

---

### The reverse direction: building the geometry from a “forbidden direction”

The second part of the excerpt looks at the same correspondence from the **nullspace** of \(M\). Suppose \(M = W^T W\) has size \(n\times n\) and rank \(r = n-i\). Then the nullspace of \(M\) has dimension \(i\). For any vector \(v\) in that nullspace, \(Mv = 0\) implies \(Wv = 0\). In the point configuration, this means the linear combination \(\sum_j v_j w_j = 0\). So the nullspace tells you which combinations of feature embeddings “cancel out”.

A particularly clean way to see the geometry is to start with an \(n\)-dimensional orthogonal basis (one axis per feature), and then *project* those basis vectors onto the subspace orthogonal to a chosen set of nullspace vectors. The Gram matrix of the projected vectors gives exactly a rank-\((n-i)\) matrix whose nullspace is spanned by those vectors.

- **Example:** Start in \(\mathbb{R}^3\) with the standard basis \(e_1, e_2, e_3\). Project onto the plane orthogonal to the all-ones vector \(v = (1,1,1)\). The projected vectors are \(p_i = e_i - \frac{1}{3}(1,1,1)\). Their pairwise dot products are \(\langle p_i, p_j \rangle = \delta_{ij} - \frac{1}{3}\). The resulting Gram matrix has \(2/3\) on the diagonal and \(-1/3\) off-diagonal, rank 2, nullspace spanned by \((1,1,1)\). The three points lie in a 2D plane, forming the vertices of an **equilateral triangle** (a regular 2-simplex).
- **General simplex:** Projecting the \(n\) standard basis vectors onto the hyperplane orthogonal to \((1,1,\ldots,1)\) gives the vertices of a **regular \((n-1)\)-simplex** embedded in \(\mathbb{R}^{n-1}\). This is the configuration where all points have the same norm and all pairwise dot products are equal (necessarily negative, since the sum of all vectors is zero). Its Gram matrix is proportional to \(I - \frac{1}{n}\mathbf{1}\mathbf{1}^T\).

---

### Why this matters for superposition

When a neural network represents \(n\) features in an \(m\)-dimensional hidden layer, it implicitly chooses a Gram matrix \(M\) (often close to the identity as much as possible, to keep features distinguishable). The rank constraint \(m < n\) forces \(M\) to have a nullspace, meaning some feature combinations are “not represented” (they cancel in the embedding space). The ideal symmetrical solution – when all features are equally important and equally sparse – is to make the nullspace as “uniform” as possible, i.e., to project out the direction \((1,1,\ldots,1)\). This yields the regular simplex, which guarantees **uniform, minimal interference** between all feature pairs. 

Thus, the polytope–matrix correspondence turns the design of superposition strategies into a problem of arranging points on a sphere in low dimensions, and the regular simplex is the mathematically natural starting point.



Here’s a step-by-step algorithm to construct the embedding matrix \(W\) that maps \(n\) features into \(m = n-1\) dimensions as a **regular simplex** — the optimal symmetric strategy when all features are equally important and sparse. The same algorithmic skeleton generalises to any chosen nullspace.

---

### Goal
Given \(n\) features and an embedding dimension \(m = n-1\), produce an \(m \times n\) matrix \(W\) whose columns \(w_i \in \mathbb{R}^m\) are the vertices of a regular \((n-1)\)-simplex, i.e.  
- All \(\|w_i\|\) equal,  
- All pairwise dot products \(\langle w_i, w_j \rangle\) equal (and negative),  
- The nullspace of the Gram matrix \(W^T W\) is spanned by \((1,1,\ldots,1)\).

---

### Step‑by‑step construction

1. **Form the projection matrix that nulls the all‑ones vector**  
   Define the \(n \times n\) matrix  
   \[
   P = I_n - \frac{1}{n} \mathbf{1}\mathbf{1}^T,
   \]
   where \(\mathbf{1}\) is the all‑ones vector.  
   \(P\) projects orthogonally onto the subspace orthogonal to \(\mathbf{1}\). Its rank is \(n-1\).

2. **Extract an orthonormal basis for that subspace**  
   Compute the eigenvalue decomposition \(P = U \Lambda U^T\) (or use SVD of \(P\)).  
   Since \(P\) is symmetric, the eigenvectors are orthogonal. Exactly \(n-1\) eigenvalues equal 1, and one eigenvalue equals 0 (with eigenvector \(\mathbf{1}/\sqrt{n}\)).  
   Let \(U_{n-1}\) be the \(n \times (n-1)\) matrix whose columns are the eigenvectors corresponding to the eigenvalue 1.

3. **Map the original standard basis vectors into the low‑dimensional space**  
   Treat the columns of the \(n \times n\) identity matrix as the “uncorrelated” feature directions in \(\mathbb{R}^n\).  
   Project each standard basis vector \(e_i\) onto the subspace via \(P\), then express the result in the orthonormal coordinates given by \(U_{n-1}\).  
   That coordinate vector is exactly  
   \[
   w_i = U_{n-1}^T e_i \quad \text{(a vector of length } n-1 \text{)}.
   \]
   Stacking these columns gives the **embedding matrix**
   \[
   W = U_{n-1}^T \quad \text{(size } (n-1) \times n \text{)}.
   \]

4. **Verify the Gram matrix**  
   By construction, \(W^T W = P\). So  
   \[
   (W^T W)_{ii} = 1 - \frac{1}{n}, \qquad
   (W^T W)_{ij} = -\frac{1}{n} \quad (i \neq j).
   \]
   All features have the same squared norm \(\frac{n-1}{n}\), and all cross‑interference terms equal \(-\frac{1}{n}\).

5. **(Optional) Rescale to unit norm**  
   If you prefer unit‑length feature vectors, multiply \(W\) by \(\sqrt{\frac{n}{n-1}}\). Then the Gram matrix becomes  
   \[
   \frac{n}{n-1}\left(I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^T\right)
   = \frac{n}{n-1} I_n - \frac{1}{n-1} \mathbf{1}\mathbf{1}^T.
   \]
   The off‑diagonal interference is now \(-\frac{1}{n-1}\), and the diagonal is 1.

---

### Generalisation to an arbitrary nullspace

Suppose you want a rank \(r = n-k\) matrix whose nullspace is spanned by a given set of \(k\) orthonormal vectors \(v_1, \ldots, v_k\) (in the example above, \(k=1\) and \(v_1 = \frac{1}{\sqrt{n}}\mathbf{1}\)). The same recipe applies:

1. Form \(N = [v_1 \cdots v_k]\) (size \(n \times k\)).  
2. Build the projection \(P = I_n - N N^T\).  
3. Find an orthonormal basis \(U_{r}\) for the range of \(P\) (e.g. eigenvectors with eigenvalue 1).  
4. Set \(W = U_{r}^T\).  

The columns of \(W\) are the coordinates of the projected standard basis vectors, and the Gram matrix \(W^T W = P\) has the chosen nullspace.

---

### Why this works
The key phenomenon is that **the Gram matrix \(W^T W\) completely encodes the geometry** (inner products) of the feature embedding. By forcing the nullspace to be spanned by \((1,1,\ldots,1)\), we make the sum of all embedding vectors zero, which — under equal‑norm constraints — forces a perfectly symmetric, equi‑angular arrangement: the regular simplex. This minimises the maximum pairwise interference and is the natural starting point for representing equally important, equally sparse features in superposition.