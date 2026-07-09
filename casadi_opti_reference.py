#!/usr/bin/env python3
#
# casadi_opti_reference.py
#
# Minimal CasADi Opti reference for the two hand-written fatrop OCPs in this repo
# (point_mass_obstacle_ocp.hpp and point_mass_lse_ocp.hpp). Same dynamics, cost
# and boundary conditions; only the obstacle inequality differs. Flip USE_LSE to
# switch between the circular obstacle and the LSE smooth-max polygon obstacle.
#
# Purpose: a comparison baseline against the low-level fatrop C++ interface, so
# nothing fancy. Uses fatrop as the solver via Opti's structure detection.
#
import casadi as ca

# ---- pick which problem to solve ------------------------------------------
USE_LSE = False   # False -> circular obstacle, True -> LSE smooth-max polygon

# ---- shared problem definition (matches main.cpp / main_lse.cpp) ----------
K = 20           # knots -> N = K-1 intervals
N = K - 1
dt = 0.1          # s
mass = 1.0

x_start = ca.DM([0.0, 0.0, 0.0, 0.0])   # at rest in origin
x_goal = ca.DM([2.0, 2.0, 0.0, 0.0])    # at rest in (2, 2)

# circular obstacle (point_mass_obstacle_ocp.hpp)
cx, cy = 1.0, 1.0
r_safe = 0.3

c2x, c2y = 0.5, 0.6
r2_safe = 0.2

# # LSE square obstacle (point_mass_lse_ocp.hpp): { p : A_l . p <= b_l }
# h = 0.45
# obs_A = ca.DM([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
# obs_b = ca.DM([cx + h, -(cx - h), cy + h, -(cy - h)])
margin = 2e-2
alpha = ca.log(2) / margin   # sharpness = log(n_e) / max_approx
# margin = 0.05


def smooth_max(e):
    # (1/a) log sum exp(a e), max-shifted for numerical stability (see lse.hpp)
    m = ca.mmax(e)
    return m + ca.log(ca.sum1(ca.exp(alpha * (e - m)))) / alpha

def smooth_min(e):
    return -smooth_max(-e)


# ---- build the OCP with Opti ----------------------------------------------
opti = ca.Opti()

# per-stage decision variables (interleaved so fatrop can detect the structure)
X = []
U = []
for k in range(N):
    Xk = opti.variable(4)
    Uk = opti.variable(2)
    X.append(Xk)
    U.append(Uk)
Xk = opti.variable(4)
X.append(Xk)


def dynamics(x, u):  # explicit Euler double integrator
    return ca.vertcat(x[0] + dt * x[2],
                      x[1] + dt * x[3],
                      x[2] + dt * u[0] / mass,
                      x[3] + dt * u[1] / mass)


cost = 0
for k in range(N):
    opti.subject_to(X[k + 1] == dynamics(X[k], U[k]))   # dynamics gap
    cost += U[k][0] ** 2 + U[k][1] ** 2                 # control effort

    if k == 0:
        opti.subject_to(X[0] == x_start)                 # boundary: start

    px, py = X[k][0], X[k][1]
    if USE_LSE:
        # e = obs_A @ ca.vertcat(px, py) - obs_b
        e = ca.vertcat((px - cx) ** 2 + (py - cy) ** 2 - r_safe ** 2, (px - c2x) ** 2 + (py - c2y) ** 2 - r2_safe ** 2)
        opti.subject_to(smooth_min(ca.vec(e)) >= margin)
    else:
        # e = obs_A @ ca.vertcat(px, py) - obs_b
        # for i in range(e.shape[0]):
        #     opti.subject_to(e[i] >= 0)
        opti.subject_to((px - cx) ** 2 + (py - cy) ** 2 >= r_safe ** 2)
        opti.subject_to((px - c2x) ** 2 + (py - c2y) ** 2 >= r2_safe ** 2)
    
opti.subject_to(X[N] == x_goal)              # boundary: goal

opti.minimize(cost)

# straight-line warm start
for k in range(K):
    a = k / (K - 1)
    opti.set_initial(X[k], (1 - a) * x_start + a * x_goal)

# ---- solve with fatrop -----------------------------------------------------
opti.solver("fatrop", {"structure_detection": "auto", "expand": True, 'debug': True},
            {"max_iter": 40, "tol": 1e-4})

sol = opti.solve()


### Plot sparsity pattern
J = ca.jacobian(opti.g, opti.x).sparsity()
lag = opti.f + ca.dot(opti.lam_g, opti.g)
H = ca.hessian(lag, opti.x)[0].sparsity()
func_h = ca.Function('h', [opti.x, opti.lam_g], [ca.hessian(opti.f + ca.dot(opti.lam_g, opti.g), opti.x)[0]]).expand()
print("Number of decision variables: ", opti.x.shape[0])
print("Number of constraints: ", opti.g.shape[0])
print("Jacobian nonzeros: ", J.nnz())
print("Hessian nonzeros: ", H.nnz())
print("Hessian nodes: ", func_h.n_nodes())

func_j = ca.Function('j', [opti.x], [ca.jacobian(opti.g, opti.x)]).expand()
print("Jacobian nodes: ", func_j.n_nodes())

# ---- report ----------------------------------------------------------------
name = "LSE smooth-max polygon" if USE_LSE else "circular"
print(f"\n=== Point-mass p2p with {name} obstacle (CasADi Opti + fatrop) ===")
px = [float(sol.value(X[k][0])) for k in range(K)]
py = [float(sol.value(X[k][1])) for k in range(K)]
min_clear = float("inf")
for k in range(K):
    dist = ((px[k] - cx) ** 2 + (py[k] - cy) ** 2) ** 0.5
    min_clear = min(min_clear, dist)
    if k % 5 == 0 or k == K - 1:
        print(f"k={k:2d}  p=({px[k]:7.4f}, {py[k]:7.4f})  dist_to_centre={dist:.4f}")
print(f"\nMinimum clearance to obstacle centre : {min_clear:.4f}")

# ---- simple visualisation --------------------------------------------------
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(px, py, "o-", ms=3, color="tab:blue", label="trajectory")
ax.plot(*x_start[:2].full().ravel(), "gs", ms=10, label="start")
ax.plot(*x_goal[:2].full().ravel(), "r*", ms=14, label="goal")
ax.add_patch(plt.Circle((cx, cy), r_safe, color="0.6", label="keep-out"))
ax.add_patch(plt.Circle((c2x, c2y), r2_safe, color="0.6", label="keep-out"))
ax.set_aspect("equal")
ax.set_xlabel("px")
ax.set_ylabel("py")
ax.set_title(f"Point-mass p2p ({name} obstacle)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("casadi_opti_reference.png", dpi=120)
print("Plot written to casadi_opti_reference.png")
plt.show()
