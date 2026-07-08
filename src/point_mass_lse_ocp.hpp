//
// point_mass_lse_ocp.hpp
//
// Point-mass point-to-point OCP whose obstacle-avoidance constraint is a
// Log-Sum-Exp (LSE) smooth-max of half-plane distances — the same smoothing
// used in embeddedcbf/lse_cbf_mpc_fatrop.py, but with the constraint value,
// Jacobian and Hessian all assembled from the *analytical* LSE derivatives in
// lse.hpp. No CasADi graph is involved for the constraint.
//
// Dynamics / cost are identical to point_mass_obstacle_ocp.hpp (a holonomic 2D
// double integrator, minimum control effort). Only the inequality constraint
// changes, so the LSE machinery is the sole new ingredient.
//
// Obstacle constraint ---------------------------------------------------------
//
//   A convex polygonal obstacle is the set  { p : A_l . p <= b_l  for all l }.
//   The signed "outside" amount for half-plane l is
//
//       e_l(p) = A_l . p - b_l            (e_l > 0  <=>  p is outside half-plane l)
//
//   The point is collision-free iff it is outside at least one half-plane, i.e.
//   max_l e_l(p) >= 0. We smooth that max with the LSE smooth_max and keep a
//   margin:
//
//       g_ineq(p) = smooth_max(e(p), alpha) >= margin
//
//   (As in the reference code, alpha = log(n_e)/max_approx controls sharpness;
//   larger alpha -> tighter approximation of the true max, stiffer Hessian.)
//
// Note on conservativeness: smooth_max >= max, so this is the *relaxed* side of
// the keep-out (it can allow a sliver of penetration ~ (1/alpha) log n_e). To be
// strictly conservative for a keep-IN constraint use smooth_min instead (see the
// README); the analytical pieces for both live in lse.hpp.
//
// Chain rule (e is affine in p, so the second-order e-term vanishes) ----------
//
//   With z = (u_k, x_k) and J = de/dz (rows = A_l in the px,py columns, else 0):
//     grad_z g = J^T p                       (p = softmax(alpha e), from lse.hpp)
//     hess_z g = J^T [alpha(diag(p)-pp^T)] J
//   The Lagrangian Hessian then gets  lam_ineq * hess_z g  added in eval_RSQrqt.
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

    /// 2D point-mass p2p problem with an LSE-smoothed convex-polygon obstacle.
    class PointMassLseOcp : public fatrop::OcpAbstract
    {
    public:
        /// @param K         number of stages
        /// @param dt        integration step [s]
        /// @param mass      point-mass [kg]
        /// @param x_start   start state [px,py,vx,vy]
        /// @param x_goal    goal  state [px,py,vx,vy]
        /// @param obs_A     obstacle half-plane normals, row-major (n_e x 2)
        /// @param obs_b     obstacle half-plane offsets   (n_e)
        /// @param alpha     LSE sharpness (e.g. log(n_e)/max_approx)
        /// @param margin    required smooth_max value (keep-out margin)
        PointMassLseOcp(Index K, Scalar dt, Scalar mass,
                        const std::array<Scalar, NX_LSE> &x_start,
                        const std::array<Scalar, NX_LSE> &x_goal,
                        const std::vector<std::array<Scalar, 2>> &obs_A,
                        const std::vector<Scalar> &obs_b, Scalar alpha, Scalar margin)
            : K_(K), dt_(dt), m_(mass), x_start_(x_start), x_goal_(x_goal), obs_A_(obs_A),
              obs_b_(obs_b), alpha_(alpha), margin_(margin),
              n_e_(static_cast<Index>(obs_b.size()))
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

            // obstacle curvature: lam_ineq * J^T hess_e J, with e affine in (px,py).
            // smooth_max_grad fills the softmax weights p from the e-values e(states_k).
            Scalar p[kMaxE];
            lse::smooth_max_grad(eval_e(states_k, p_e_), n_e_, alpha_, p);

            // Contract the e-Hessian with the constant rows A_l over (px,py):
            //   H[a][b] = sum_{i,j} A_i[a] * hess_e(i,j) * A_j[b],   a,b in {0:px, 1:py}
            Scalar H[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
            for (Index i = 0; i < n_e_; ++i)
                for (Index j = 0; j < n_e_; ++j)
                {
                    const Scalar he = lse::smooth_max_hess_e(p, alpha_, i, j);
                    for (Index a = 0; a < 2; ++a)
                        for (Index b = 0; b < 2; ++b)
                            H[a][b] += obs_A_[i][a] * he * obs_A_[j][b];
                }

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
        // g_ineq = smooth_max( e(p) ),  e_l = A_l . p - b_l.
        Index eval_gineq(const Scalar * /*inputs_k*/, const Scalar *states_k, Scalar *res,
                         const Index /*k*/) override
        {
            res[0] = lse::smooth_max(eval_e(states_k, p_e_), n_e_, alpha_);
            return 0;
        }

        // g_ineq Jacobian (transposed). grad_z g = J^T p, with J nonzero only in
        // the (px,py) state rows. dg/dpx = sum_l p_l A_l[0], dg/dpy = sum_l p_l A_l[1].
        Index eval_Ggt_ineq(const Scalar * /*inputs_k*/, const Scalar *states_k, MAT *res,
                            const Index k) override
        {
            fatrop::blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            Scalar p[kMaxE];
            lse::smooth_max_grad(eval_e(states_k, p_e_), n_e_, alpha_, p);
            Scalar dpx = 0.0, dpy = 0.0;
            for (Index l = 0; l < n_e_; ++l)
            {
                dpx += p[l] * obs_A_[l][0];
                dpy += p[l] * obs_A_[l][1];
            }
            const Index nu = get_nu(k);
            fatrop::blasfeo_matel_wrap(res, nu + 0, 0) = dpx;
            fatrop::blasfeo_matel_wrap(res, nu + 1, 0) = dpy;
            return 0;
        }

        // smooth_max(e) >= margin  ==>  bounds [margin, +inf].
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
        const std::vector<std::array<Scalar, 2>> &obs_A() const { return obs_A_; }
        const std::vector<Scalar> &obs_b() const { return obs_b_; }
        Scalar alpha() const { return alpha_; }

        // Signed half-plane "outside" amount; >= 0 everywhere means truly collision free.
        Scalar true_max_outside(const Scalar px, const Scalar py) const
        {
            Scalar m = -std::numeric_limits<Scalar>::infinity();
            for (Index l = 0; l < n_e_; ++l)
                m = std::max(m, obs_A_[l][0] * px + obs_A_[l][1] * py - obs_b_[l]);
            return m;
        }

    private:
        static constexpr Index kMaxE = 32; // stack-buffer cap on number of half-planes

        // Evaluate e_l = A_l . p - b_l into the provided buffer and return it.
        // p = (states_k[0], states_k[1]) are (px, py).
        const Scalar *eval_e(const Scalar *states_k, Scalar *buf) const
        {
            for (Index l = 0; l < n_e_; ++l)
                buf[l] = obs_A_[l][0] * states_k[0] + obs_A_[l][1] * states_k[1] - obs_b_[l];
            return buf;
        }

        const Index K_;
        const Scalar dt_;
        const Scalar m_;
        const std::array<Scalar, NX_LSE> x_start_;
        const std::array<Scalar, NX_LSE> x_goal_;
        const std::vector<std::array<Scalar, 2>> obs_A_;
        const std::vector<Scalar> obs_b_;
        const Scalar alpha_;
        const Scalar margin_;
        const Index n_e_;
        mutable Scalar p_e_[kMaxE]; // scratch for e-values (single-threaded use)
    };

} // namespace lse_fatrop

#endif // LSE_FATROP_POINT_MASS_LSE_OCP_HPP
