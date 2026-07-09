//
// point_mass_lse_ocp.hpp
//
// Point-mass point-to-point OCP whose obstacle-avoidance constraint is a
// Log-Sum-Exp (LSE) smooth-min of per-obstacle signed distances — the same
// smoothing used in embeddedcbf/lse_cbf_mpc_fatrop.py, but with the constraint
// value, Jacobian and Hessian all assembled from the *analytical* LSE
// derivatives in lse.hpp. No CasADi graph is involved for the constraint.
//
// Dynamics / cost are identical to point_mass_obstacle_ocp.hpp (a holonomic 2D
// double integrator, minimum control effort). Only the inequality constraint
// changes, so the LSE machinery is the sole new ingredient.
//
// Obstacle constraint ---------------------------------------------------------
//
//   There are n_e circular obstacles. For obstacle l the signed "outside" amount
//   (matching casadi_opti_reference.py) is the squared-distance slack
//
//       e_l(p) = (px - cx_l)^2 + (py - cy_l)^2 - r_l^2      (e_l >= 0 <=> outside l)
//
//   The point is collision-free iff it is outside *every* obstacle, i.e.
//   min_l e_l(p) >= 0. We smooth that min with the LSE smooth_min and keep a
//   margin:
//
//       g_ineq(p) = smooth_min(e(p), alpha) >= margin
//
//   Because smooth_min <= min, this is the *conservative* side of the keep-out
//   (it under-estimates the true clearance), so enforcing it keeps the point
//   strictly outside all obstacles. Larger alpha -> tighter approximation of the
//   true min, stiffer Hessian.
//
// Chain rule (e is now *quadratic* in p, so the second-order e-term stays!) ----
//
//   With z = (u_k, x_k) and J = de/dz (row l is [2(px-cx_l), 2(py-cy_l)] in the
//   px,py columns, else 0), and q = softmax(-alpha e) = d smooth_min / d e:
//     grad_z g = J^T q
//     hess_z g = J^T [hess_e] J  +  sum_l q_l * d^2 e_l / dz^2
//   where hess_e = -alpha(diag(q)-qq^T) (lse::smooth_min_hess_e) and each
//   d^2 e_l/dz^2 = 2*I on (px,py). Since sum_l q_l = 1 the second term is just
//   2*I on the (px,py) diagonal. The Lagrangian Hessian gets lam_ineq * hess_z g.
//

#ifndef LSE_FATROP_POINT_MASS_LSE_OCP_HPP
#define LSE_FATROP_POINT_MASS_LSE_OCP_HPP

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include <fatrop/fatrop.hpp>

#include "lse.hpp"

namespace lse_fatrop
{
    using fatrop::Index;
    using fatrop::Scalar;
    // Note: MAT is a macro (-> blasfeo_dmat, a global type), not a fatrop:: name,
    // so it is used unqualified rather than pulled in with a using-declaration.

    static constexpr Index NX_LSE = 4; // [px, py, vx, vy]
    static constexpr Index NU_LSE = 2; // [fx, fy]

    /// 2D point-mass p2p problem with an LSE-smoothed multi-circle obstacle.
    class PointMassLseOcp : public fatrop::OcpAbstract
    {
    public:
        /// @param K         number of stages
        /// @param dt        integration step [s]
        /// @param mass      point-mass [kg]
        /// @param x_start   start state [px,py,vx,vy]
        /// @param x_goal    goal  state [px,py,vx,vy]
        /// @param centres   obstacle centres [cx,cy] (n_e of them)
        /// @param radii     obstacle radii            (n_e of them)
        /// @param alpha     LSE sharpness (e.g. log(n_e)/margin)
        /// @param margin    required smooth_min value (keep-out margin)
        PointMassLseOcp(Index K, Scalar dt, Scalar mass,
                        const std::array<Scalar, NX_LSE> &x_start,
                        const std::array<Scalar, NX_LSE> &x_goal,
                        const std::vector<std::array<Scalar, 2>> &centres,
                        const std::vector<Scalar> &radii, Scalar alpha, Scalar margin)
            : K_(K), dt_(dt), m_(mass), x_start_(x_start), x_goal_(x_goal),
              centres_(centres), radii_(radii), alpha_(alpha), margin_(margin),
              n_e_(static_cast<Index>(radii.size()))
        {
        }

        // --- dimensions ----------------------------------------------------------
        Index get_horizon_length() const override { return K_; }
        Index get_nx(const Index /*k*/) const override { return NX_LSE; }
        Index get_nu(const Index k) const override { return (k == K_ - 1) ? 0 : NU_LSE; }
        Index get_ng(const Index k) const override
        {
            return (k == 0 || k == K_ - 1) ? NX_LSE : 0;
        }
        Index get_ng_ineq(const Index /*k*/) const override { return 1; } // one LSE constraint

        // --- dynamics (linear double integrator) ---------------------------------
        Index eval_BAbt(const Scalar * /*states_kp1*/, const Scalar * /*inputs_k*/,
                        const Scalar * /*states_k*/, MAT *res, const Index /*k*/) override
        {
            fatrop::blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            fatrop::blasfeo_matel_wrap(res, 0, 2) = dt_ / m_; // B^T : d(vx')/d(fx)
            fatrop::blasfeo_matel_wrap(res, 1, 3) = dt_ / m_; // B^T : d(vy')/d(fy)
            fatrop::blasfeo_diare_wrap(NX_LSE, 1.0, res, NU_LSE, 0); // A^T identity
            fatrop::blasfeo_matel_wrap(res, NU_LSE + 2, 0) = dt_;    // A^T : d(px')/d(vx)
            fatrop::blasfeo_matel_wrap(res, NU_LSE + 3, 1) = dt_;    // A^T : d(py')/d(vy)
            return 0;
        }

        Index eval_b(const Scalar *states_kp1, const Scalar *inputs_k,
                     const Scalar *states_k, Scalar *res, const Index /*k*/) override
        {
            res[0] = -states_kp1[0] + states_k[0] + dt_ * states_k[2];
            res[1] = -states_kp1[1] + states_k[1] + dt_ * states_k[3];
            res[2] = -states_kp1[2] + states_k[2] + dt_ * inputs_k[0] / m_;
            res[3] = -states_kp1[3] + states_k[3] + dt_ * inputs_k[1] / m_;
            return 0;
        }

        // --- objective (minimum control effort) ----------------------------------
        Index eval_L(const Scalar *objective_scale, const Scalar *inputs_k,
                     const Scalar * /*states_k*/, Scalar *res, const Index k) override
        {
            *res = (k == K_ - 1) ? 0.0
                                 : objective_scale[0] * (inputs_k[0] * inputs_k[0] +
                                                         inputs_k[1] * inputs_k[1]);
            return 0;
        }

        Index eval_rq(const Scalar *objective_scale, const Scalar *inputs_k,
                      const Scalar * /*states_k*/, Scalar *res, const Index k) override
        {
            const Index nu = get_nu(k);
            for (Index i = 0; i < nu + NX_LSE; ++i)
                res[i] = 0.0;
            if (nu > 0)
            {
                res[0] = 2.0 * objective_scale[0] * inputs_k[0];
                res[1] = 2.0 * objective_scale[0] * inputs_k[1];
            }
            return 0;
        }

        // Lagrangian Hessian = cost curvature (2*I on inputs) + lam_ineq * hess_z g.
        Index eval_RSQrqt(const Scalar *objective_scale, const Scalar * /*inputs_k*/,
                          const Scalar *states_k, const Scalar * /*lam_dyn_k*/,
                          const Scalar * /*lam_eq_k*/, const Scalar *lam_eq_ineq_k,
                          MAT *res, const Index k) override
        {
            fatrop::blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            const Index nu = get_nu(k);

            // cost curvature
            if (nu > 0)
                fatrop::blasfeo_diare_wrap(NU_LSE, 2.0 * objective_scale[0], res, 0, 0);

            // obstacle curvature: lam_ineq * ( J^T hess_e J + sum_l q_l d^2 e_l ).
            // q = softmax(-alpha e) are the smooth_min weights (grad w.r.t. e).
            Scalar e[kMaxE], q[kMaxE];
            eval_e(states_k, e);
            lse::smooth_min_grad(e, n_e_, alpha_, q);

            // J^T hess_e J on the (px,py) block, with J row l = [2(px-cx_l), 2(py-cy_l)].
            Scalar H[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
            for (Index i = 0; i < n_e_; ++i)
            {
                const Scalar Ji[2] = {2.0 * (states_k[0] - centres_[i][0]),
                                      2.0 * (states_k[1] - centres_[i][1])};
                for (Index j = 0; j < n_e_; ++j)
                {
                    const Scalar he = lse::smooth_min_hess_e(q, alpha_, i, j);
                    const Scalar Jj[2] = {2.0 * (states_k[0] - centres_[j][0]),
                                          2.0 * (states_k[1] - centres_[j][1])};
                    for (Index a = 0; a < 2; ++a)
                        for (Index b = 0; b < 2; ++b)
                            H[a][b] += Ji[a] * he * Jj[b];
                }
            }
            // second-order e-term: sum_l q_l * d^2 e_l/dp^2 = 2*I (sum_l q_l = 1).
            H[0][0] += 2.0;
            H[1][1] += 2.0;

            const Scalar lam = lam_eq_ineq_k[0];
            // px is local index (nu+0), py is (nu+1).
            fatrop::blasfeo_matel_wrap(res, nu + 0, nu + 0) += lam * H[0][0];
            fatrop::blasfeo_matel_wrap(res, nu + 1, nu + 1) += lam * H[1][1];
            fatrop::blasfeo_matel_wrap(res, nu + 0, nu + 1) += lam * H[0][1];
            fatrop::blasfeo_matel_wrap(res, nu + 1, nu + 0) += lam * H[1][0];
            return 0;
        }

        // --- equality constraints (boundary conditions) --------------------------
        Index eval_g(const Scalar * /*inputs_k*/, const Scalar *states_k, Scalar *res,
                     const Index k) override
        {
            if (k == 0)
                for (Index i = 0; i < NX_LSE; ++i)
                    res[i] = states_k[i] - x_start_[i];
            else if (k == K_ - 1)
                for (Index i = 0; i < NX_LSE; ++i)
                    res[i] = states_k[i] - x_goal_[i];
            return 0;
        }

        Index eval_Ggt(const Scalar * /*inputs_k*/, const Scalar * /*states_k*/, MAT *res,
                       const Index k) override
        {
            fatrop::blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            if (k == 0 || k == K_ - 1)
                fatrop::blasfeo_diare_wrap(NX_LSE, 1.0, res, get_nu(k), 0);
            return 0;
        }

        // --- LSE inequality constraint -------------------------------------------
        // g_ineq = smooth_min( e(p) ),  e_l = (px-cx_l)^2 + (py-cy_l)^2 - r_l^2.
        Index eval_gineq(const Scalar * /*inputs_k*/, const Scalar *states_k, Scalar *res,
                         const Index /*k*/) override
        {
            Scalar e[kMaxE];
            eval_e(states_k, e);
            res[0] = lse::smooth_min(e, n_e_, alpha_);
            return 0;
        }

        // g_ineq Jacobian (transposed). grad_z g = J^T q, with J nonzero only in
        // the (px,py) state rows. dg/dpx = sum_l q_l 2(px-cx_l),
        // dg/dpy = sum_l q_l 2(py-cy_l).
        Index eval_Ggt_ineq(const Scalar * /*inputs_k*/, const Scalar *states_k, MAT *res,
                            const Index k) override
        {
            fatrop::blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            Scalar e[kMaxE], q[kMaxE];
            eval_e(states_k, e);
            lse::smooth_min_grad(e, n_e_, alpha_, q);
            Scalar dpx = 0.0, dpy = 0.0;
            for (Index l = 0; l < n_e_; ++l)
            {
                dpx += q[l] * 2.0 * (states_k[0] - centres_[l][0]);
                dpy += q[l] * 2.0 * (states_k[1] - centres_[l][1]);
            }
            const Index nu = get_nu(k);
            fatrop::blasfeo_matel_wrap(res, nu + 0, 0) = dpx;
            fatrop::blasfeo_matel_wrap(res, nu + 1, 0) = dpy;
            return 0;
        }

        // smooth_min(e) >= margin  ==>  bounds [margin, +inf].
        Index get_bounds(Scalar *lower, Scalar *upper, const Index /*k*/) const override
        {
            lower[0] = margin_;
            upper[0] = std::numeric_limits<Scalar>::infinity();
            return 0;
        }

        // --- initial guess (straight-line warm start) ----------------------------
        Index get_initial_xk(Scalar *xk, const Index k) const override
        {
            const Scalar a = (K_ > 1) ? static_cast<Scalar>(k) / (K_ - 1) : 0.0;
            for (Index i = 0; i < NX_LSE; ++i)
                xk[i] = (1.0 - a) * x_start_[i] + a * x_goal_[i];
            return 0;
        }
        Index get_initial_uk(Scalar *uk, const Index k) const override
        {
            if (k == K_ - 1)
                return 0;
            uk[0] = 0.0;
            uk[1] = 0.0;
            return 0;
        }

        // --- accessors for the driver --------------------------------------------
        Index K() const { return K_; }
        Scalar dt() const { return dt_; }
        Scalar margin() const { return margin_; }
        Scalar alpha() const { return alpha_; }
        Index n_e() const { return n_e_; }
        const std::vector<std::array<Scalar, 2>> &centres() const { return centres_; }
        const std::vector<Scalar> &radii() const { return radii_; }

        // True (non-smoothed) worst-case slack: min_l e_l(p). >= 0 means the point
        // is genuinely outside every obstacle (collision-free).
        Scalar true_min_outside(const Scalar px, const Scalar py) const
        {
            Scalar m = std::numeric_limits<Scalar>::infinity();
            for (Index l = 0; l < n_e_; ++l)
            {
                const Scalar dx = px - centres_[l][0], dy = py - centres_[l][1];
                m = std::min(m, dx * dx + dy * dy - radii_[l] * radii_[l]);
            }
            return m;
        }

    private:
        static constexpr Index kMaxE = 32; // stack-buffer cap on number of obstacles

        // Evaluate e_l = (px-cx_l)^2 + (py-cy_l)^2 - r_l^2 into `buf`.
        // p = (states_k[0], states_k[1]) are (px, py).
        void eval_e(const Scalar *states_k, Scalar *buf) const
        {
            for (Index l = 0; l < n_e_; ++l)
            {
                const Scalar dx = states_k[0] - centres_[l][0];
                const Scalar dy = states_k[1] - centres_[l][1];
                buf[l] = dx * dx + dy * dy - radii_[l] * radii_[l];
            }
        }

        const Index K_;
        const Scalar dt_;
        const Scalar m_;
        const std::array<Scalar, NX_LSE> x_start_;
        const std::array<Scalar, NX_LSE> x_goal_;
        const std::vector<std::array<Scalar, 2>> centres_;
        const std::vector<Scalar> radii_;
        const Scalar alpha_;
        const Scalar margin_;
        const Index n_e_;
    };

} // namespace lse_fatrop

#endif // LSE_FATROP_POINT_MASS_LSE_OCP_HPP
