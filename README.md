# Point-mass point-to-point with obstacle avoidance — fatrop low-level interface

A minimal, self-contained example of how to write **your own** optimal control
problem (OCP) directly against [fatrop](https://github.com/meco-group/fatrop)'s
**low-level C++ interface** (`fatrop::OcpAbstract`). You hand fatrop the problem
dimensions and the function / Jacobian / Hessian evaluations yourself — no
CasADi, no code generation — so you have full control over every block of the
KKT system, including the **constraint Jacobian** and the **Lagrangian Hessian**.

The toy problem: a 2D point-mass robot has to move from a start pose to a goal
pose with minimum control effort, while staying outside a single static circular
(spherical-in-2D) obstacle.

```
        y
        ^
    2   |                    * goal (2,2), at rest
        |                 .
        |              .
        |         ( O )          <- obstacle: centre (1,1), keep-out r = 0.5
        |      .
        |   .
    0   | S  ___________________> x
        start (0,0), at rest
```

---

## 1. The problem

| symbol | meaning | size |
| ------ | ------- | ---- |
| `x = [px, py, vx, vy]` | state (position, velocity) | `nx = 4` |
| `u = [fx, fy]` | input (force, mass-normalised) | `nu = 2` |

**Dynamics** (explicit Euler, fully *linear* — a holonomic double integrator):

```
px+ = px + dt*vx
py+ = py + dt*vy
vx+ = vx + dt*fx/m
vy+ = vy + dt*fy/m
```

**Cost** — control effort: `sum_k (fx^2 + fy^2)`.

**Equality constraints** — pin the boundary states: `x_0 = x_start`, `x_{K-1} = x_goal`.

**Inequality constraint** — obstacle avoidance, imposed at every stage:

```
(px - cx)^2 + (py - cy)^2  >=  r_safe^2
```

This last constraint is the interesting one: it is **nonlinear in the state**, so
its Jacobian is state-dependent and it contributes curvature to the Lagrangian
Hessian. Everything else is linear / quadratic, which keeps the derivative
bookkeeping easy to follow.

---

## 2. How the low-level interface works

You implement the pure-virtual methods of `fatrop::OcpAbstract` (see
`<fatrop/ocp/ocp_abstract.hpp>`). They fall into three groups:

* **dimensions** — `get_nx`, `get_nu`, `get_ng`, `get_ng_ineq`, `get_horizon_length`
* **values** — `eval_L`, `eval_b`, `eval_g`, `eval_gineq`, `get_bounds`
* **derivatives** — `eval_rq` (objective gradient), `eval_BAbt` (dynamics
  Jacobian), `eval_Ggt` (equality-constraint Jacobian), `eval_Ggt_ineq`
  (**inequality-constraint Jacobian**), `eval_RSQrqt` (**Lagrangian Hessian** +
  gradient)

### Conventions you must respect

* Per stage `k` the decision variables are ordered `(u_k, x_k)`.
* The `…t` matrices (`BAbt`, `RSQrqt`, `Ggt`, `Ggt_ineq`) are passed
  **transposed** in **column-major BLASFEO** storage with shape
  `(nu + nx + 1) × (·)`:

  ```
  row block [0      .. nu-1   ]  = derivative w.r.t. inputs  u_k
  row block [nu     .. nu+nx-1]  = derivative w.r.t. states  x_k
  row        nu+nx               = reserved for fatrop (only zeroed, never written)
  ```

* Dynamics are written as the residual `b = -x_{k+1} + f(x_k, u_k)` (`eval_b`).
* The Lagrangian whose Hessian you assemble in `eval_RSQrqt` is

  ```
  L = obj_scale * cost
    + lam_dyn   . ( -x_{k+1} + f(x_k,u_k) )
    + lam_eq    . g_eq
    + lam_ineq  . g_ineq
  ```

  so every constraint contributes `(its multiplier) * (its Hessian)`.

### Where the obstacle Jacobian / Hessian live

With `g = (px-cx)^2 + (py-cy)^2`:

* **Jacobian** (`eval_Ggt_ineq`): `dg/dpx = 2(px-cx)`, `dg/dpy = 2(py-cy)`,
  written into the state rows of the transposed matrix.
* **Hessian** (`eval_RSQrqt`): `d^2g/dpx^2 = d^2g/dpy^2 = 2`, added as
  `lam_ineq * 2` on the `px` and `py` diagonal entries of the Hessian block.

These two methods are exactly the place to edit if you want to change the
constraint Jacobian / Hessian.

---

## 3. Code structure

```
lse_fatrop/
├── CMakeLists.txt              top-level build; pulls in fatrop (+ BLASFEO)
├── README.md                   this file
├── plot_trajectory.py          optional: plot trajectory.csv with matplotlib
└── src/
    ├── point_mass_obstacle_ocp.hpp   OcpAbstract subclass — single circular
    │                                 obstacle (the place to study first).
    ├── main.cpp                      driver for the circular-obstacle example.
    ├── lse.hpp                       CasADi-free analytical log-sum-exp:
    │                                 smooth_min / smooth_max + their exact
    │                                 gradient & Hessian (see §7).
    ├── point_mass_lse_ocp.hpp        OcpAbstract subclass — convex-polygon
    │                                 obstacle via an LSE smooth-max constraint,
    │                                 Jacobian/Hessian chained analytically.
    └── main_lse.cpp                  driver for the LSE example.
```

Two executables are produced: `point_mass_obstacle` (§5) and
`point_mass_lse_obstacle` (§7).

The solve life cycle in `main.cpp`:

1. instantiate the user problem — `PointMassObstacleOcp`
2. wrap it (`NlpOcp`) and build the interior-point algorithm (`IpAlgBuilder<OcpType>`)
3. set a couple of solver options (`OptionRegistry`)
4. solve (`ipalg->optimize()`)
5. pull the primal trajectory out of the solver's `IpData`
6. report + dump `trajectory.csv`

---

## 4. Building

### Prerequisites

* A C++17 compiler (GCC ≥ 9 / Clang ≥ 10)
* CMake ≥ 3.14
* A fatrop **source checkout** (the
  [v1.1.0 API](https://github.com/meco-group/fatrop) or newer, i.e. the one that
  ships `include/fatrop/ocp/ocp_abstract.hpp`)
* Internet access on the first configure — BLASFEO is fetched and built
  automatically by fatrop (`WITH_BUILD_BLASFEO`)

### Default (recommended): build against a fatrop source tree

By default the build expects a fatrop checkout next to this repo (`../fatrop`).
Point `FATROP_ROOT` elsewhere if needed. fatrop and BLASFEO are compiled as part
of the build — nothing has to be installed first.

```bash
cd lse_fatrop
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release        # uses ../fatrop
#   or, if fatrop lives elsewhere:
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFATROP_ROOT=/path/to/fatrop
cmake --build build -j
```

> The first configure clones BLASFEO via CMake `FetchContent`; subsequent builds
> are incremental.

### Alternative: build against an installed fatrop

If you already have fatrop **and** BLASFEO installed (so that
`find_package(fatrop)` and `find_package(blasfeo)` work):

```bash
cmake -S . -B build -DUSE_INSTALLED_FATROP=ON -DCMAKE_PREFIX_PATH="/path/to/install"
cmake --build build -j
```

---

## 5. Running

```bash
./build/point_mass_obstacle
```

Expected output (abridged):

```
=== Point-mass point-to-point with circular obstacle ===
Return flag : 0  (converged=1)
Iterations  : ...
Obstacle    : centre (1.0000, 1.0000), keep-out radius 0.5000

k= 0  p=( 0.0000,  0.0000)  v=( ... )  dist=1.4142
k= 5  p=( ... )  ...
...
k=40  p=( 2.0000,  2.0000)  v=( 0.0000,  0.0000)  dist=1.4142

Minimum clearance to obstacle centre : 0.50xx  (required >= 0.5000)
Trajectory written to trajectory.csv
```

The reported **minimum clearance** is `>= r_safe`, confirming the path bends
around the obstacle. The exit code is `0` on a converged solve, `1` otherwise.

### Plotting (optional)

```bash
python3 plot_trajectory.py build/trajectory.csv     # or wherever you ran it
```

(Requires `matplotlib`. Run the executable from your build dir so the CSV is
written there, or pass the CSV path explicitly.)

---

## 6. Making it your own

* **Move the obstacle / change the keep-out distance** — edit the `obstacle` /
  `r_safe` values in `src/main.cpp`, or the start/goal/horizon there.
* **Change the constraint Jacobian** — edit `eval_Ggt_ineq` (inequalities) or
  `eval_Ggt` (equalities) in `src/point_mass_obstacle_ocp.hpp`.
* **Change the Hessian** — edit `eval_RSQrqt`; remember to add
  `multiplier * constraint_Hessian` for every nonlinear constraint, and keep the
  cost curvature in the input block.
* **Add a second obstacle** — bump `get_ng_ineq` to 2 and fill a second column in
  `eval_gineq` / `eval_Ggt_ineq`, a second `(lower, upper)` pair in `get_bounds`,
  and a second curvature term in `eval_RSQrqt`.

> Tip: if you get the Jacobian/Hessian analytics wrong, fatrop's inertia
> correction will usually still converge (just more slowly) — so a good sanity
> check when extending this is to compare iteration counts against a
> finite-difference Hessian.

---

## 7. Log-sum-exp (LSE) obstacle example — analytical Jacobian & Hessian

`point_mass_lse_obstacle` solves the *same* point-mass point-to-point problem but
replaces the single quadratic obstacle with a **convex polygonal obstacle**, kept
out via a **log-sum-exp smoothing** of the half-plane distances. This is the LSE
formulation used in `embeddedcbf/lse_cbf_mpc_fatrop.py` / `helpers.py`, except the
constraint value, **Jacobian and Hessian are all assembled from closed-form
analytical derivatives** (`src/lse.hpp`) — **no CasADi graph** is built or
traversed for the constraint.

### The smoothing (identical to `helpers.py`)

```
smooth_max(e, a) =  (1/a) log sum_i exp( a e_i)
smooth_min(e, a) = -(1/a) log sum_i exp(-a e_i)   ( = -smooth_max(-e, a) )
```

(implemented with the standard max-shift for numerical stability). `a` is the
sharpness; following the reference code we use `a = log(n_e) / max_approx`.

### Analytical derivatives (`src/lse.hpp`)

With the softmax weights `p = softmax(a·e)` and `q = softmax(-a·e)`:

| function | gradient wrt `e` | Hessian wrt `e` |
| -------- | ---------------- | --------------- |
| `smooth_max` | `p` | `a (diag(p) − p pᵀ)`  (PSD) |
| `smooth_min` | `q` | `−a (diag(q) − q qᵀ)` (NSD) |

The gradient is literally the softmax weight vector; the Hessian is `a` times the
covariance of that categorical distribution. `lse.hpp` exposes
`smooth_max` / `smooth_min` (values), `smooth_max_grad` / `smooth_min_grad`
(value + weights), and `smooth_max_hess_e` / `smooth_min_hess_e` (Hessian
entries).

### Wiring it into the OCP (`src/point_mass_lse_ocp.hpp`)

The obstacle `{ p : A_l·p ≤ b_l }` gives per-half-plane "outside" amounts
`e_l(p) = A_l·p − b_l`. Collision-free ⇔ `max_l e_l ≥ 0`, smoothed to

```
g_ineq(p) = smooth_max(e(p), a)  ≥  margin
```

Because `e` is **affine** in the position, the chain rule is exact and compact
(`J = de/dz` has the constant rows `A_l` in the `px,py` columns):

```
eval_gineq    :  g       = smooth_max(e)
eval_Ggt_ineq :  ∇_z g   = Jᵀ p              -> dg/dpx = Σ_l p_l A_l[0],  dg/dpy = Σ_l p_l A_l[1]
eval_RSQrqt   :  ∇²_z g  = Jᵀ [a(diag(p)−ppᵀ)] J   (added as lam_ineq · ∇²_z g)
```

Run it:

```bash
./build/point_mass_lse_obstacle      # writes trajectory_lse.csv
python3 plot_trajectory.py build/trajectory_lse.csv   # (draws the keep-out circle;
                                                       #  the LSE obstacle is a square)
```

The driver also prints `true_max_outside` (the exact, non-smoothed
`max_l e_l`); `≥ 0` at every knot confirms the trajectory is collision-free.

### Extending to a nonlinear `e(z)` (the full CBF case)

In `lse_cbf_mpc_fatrop.py` the `e_l` are signed distances of **rotated robot
vertices**, so `e` is *nonlinear* in the state (it contains `cos θ`, `sin θ`).
The only change to the recipe is that the chain-rule Hessian gains the
second-order term:

```
∇²_z g = Jᵀ H_e J  +  Σ_i (grad_e)_i ∇²_z e_i
```

i.e. you add each `e_i`'s own Hessian weighted by the LSE gradient component
`p_i` (or `q_i`). `lse.hpp` already returns those weights, so you only need the
analytical `∇²_z e_i` of your geometry. Switching the keep-out to a conservative
keep-**in** constraint is just `smooth_max → smooth_min` (and `≥ margin` on the
min), using the `smooth_min*` helpers.
