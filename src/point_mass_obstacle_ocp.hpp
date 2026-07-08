//
// point_mass_obstacle_ocp.hpp
//
// A minimal, fully hand-written Optimal Control Problem (OCP) for the fatrop
// *low-level* C++ interface (fatrop::OcpAbstract).
//
// Problem ---------------------------------------------------------------------
//
//   A 2D point-mass robot ("holonomic" double integrator) has to travel from a
//   start pose to a goal pose in minimum control effort, while staying outside
//   a single static circular/spherical obstacle.
//
//   state   x_k = [px, py, vx, vy]      (nx = 4)
//   input   u_k = [fx, fy]              (nu = 2)   (force, mass-normalised below)
//
//   dynamics (explicit Euler, fully *linear* -> "holonomic" point mass):
//       px_{k+1} = px + dt * vx
//       py_{k+1} = py + dt * vy
//       vx_{k+1} = vx + dt * fx / m
//       vy_{k+1} = vy + dt * fy / m
//
//   cost:            sum_k ( fx^2 + fy^2 )                 (control effort)
//
//   equality constr: x_0   = x_start                       (boundary conditions)
//                    x_{K-1} = x_goal
//
//   inequality:      (px - cx)^2 + (py - cy)^2 >= r_safe^2 (obstacle avoidance)
//
// Why this is a good "low-level interface" demo --------------------------------
//
//   The obstacle-avoidance constraint is the interesting bit: it is *nonlinear*
//   in the state, so the user is forced to hand-code
//     * its value           -> eval_gineq()
//     * its Jacobian        -> eval_Ggt_ineq()       (the "constraint Jacobian")
//     * its Hessian (times
//       the multiplier)     -> eval_RSQrqt()         (the "constraint Hessian"
//                                                      contribution to the
//                                                      Lagrangian Hessian)
//   These are exactly the quantities the user asked to be able to change.
//   Everything else (linear dynamics, quadratic cost, linear boundary
//   constraints) is kept as simple as possible so the derivative bookkeeping
//   stands out.
//
// fatrop conventions you must respect -----------------------------------------
//
//   * Per stage k the decision variables are ordered as ( u_k , x_k ).
//   * All "...t" matrices (BAbt, RSQrqt, Ggt, Ggt_ineq) are passed *transposed*
//     and in column-major BLASFEO storage, with shape (nu + nx + 1) x (.).
//     Row block [0       .. nu-1     ] = derivative w.r.t. inputs  u_k
//     Row block [nu      .. nu+nx-1  ] = derivative w.r.t. states  x_k
//     Row        nu+nx                 = reserved for fatrop (DO NOT write it;
//                                        we only zero it).
//   * Dynamics must be written in the residual form  -x_{k+1} + f(x_k,u_k) = 0
//     (see eval_b).
//   * The Lagrangian whose Hessian we assemble in eval_RSQrqt is
//        L = obj_scale * cost
//          + lam_dyn   . ( -x_{k+1} + f(x_k,u_k) )
//          + lam_eq    . g_eq(x_k,u_k)
//          + lam_ineq  . g_ineq(x_k,u_k)
//     so each constraint contributes (its multiplier) * (its Hessian).
//

#ifndef LSE_FATROP_POINT_MASS_OBSTACLE_OCP_HPP
#define LSE_FATROP_POINT_MASS_OBSTACLE_OCP_HPP

#include <array>
#include <limits>

#include <fatrop/fatrop.hpp>

namespace lse_fatrop
{
    using fatrop::Index;
    using fatrop::Scalar;
    // Note: MAT is a macro (-> blasfeo_dmat, a global type), not a fatrop:: name,
    // so it is used unqualified rather than pulled in with a using-declaration.

    // Convenience: state / input sizes (constant over the horizon here).
    static constexpr Index NX = 4; // [px, py, vx, vy]
    static constexpr Index NU = 2; // [fx, fy]

    /// 2D point-mass point-to-point problem with one static circular obstacle.
    ///
    /// Derives from fatrop::OcpAbstract (dynamic polymorphism). Every virtual
    /// below maps one-to-one onto a quantity fatrop needs: dimensions, function
    /// values, and the (transposed) Jacobian / Hessian blocks.
    class PointMassObstacleOcp : public fatrop::OcpAbstract
    {
    public:
        /// @param K        number of stages (knot points), horizon = K-1 intervals
        /// @param dt       integration step [s]
        /// @param mass     point-mass [kg]
        /// @param x_start  full start state [px,py,vx,vy]
        /// @param x_goal   full goal  state [px,py,vx,vy]
        /// @param obstacle [cx, cy]   obstacle centre
        /// @param r_safe   obstacle radius + robot radius + margin (the kept distance)
        PointMassObstacleOcp(Index K, Scalar dt, Scalar mass,
                             const std::array<Scalar, NX> &x_start,
                             const std::array<Scalar, NX> &x_goal,
                             const std::array<Scalar, 2> &obstacle, Scalar r_safe)
            : K_(K), dt_(dt), m_(mass), x_start_(x_start), x_goal_(x_goal),
              cx_(obstacle[0]), cy_(obstacle[1]), r_safe_(r_safe)
        {
        }

        // --- problem dimensions --------------------------------------------------

        Index get_horizon_length() const override { return K_; }

        Index get_nx(const Index /*k*/) const override { return NX; }

        // No input at the final stage (nothing to actuate after the last state).
        Index get_nu(const Index k) const override { return (k == K_ - 1) ? 0 : NU; }

        // Equality constraints only pin the boundary states.
        Index get_ng(const Index k) const override
        {
            if (k == 0 || k == K_ - 1)
                return NX; // fix all 4 components of the boundary state
            return 0;
        }

        // One obstacle inequality at every stage.
        Index get_ng_ineq(const Index /*k*/) const override { return 1; }

        // --- dynamics ------------------------------------------------------------

        // BAbt = [ B^T ; A^T ; b^T ]  (shape (nu+nx+1) x nx_{k+1}, column-major).
        // The dynamics are linear, so A and B are constant; b^T (last row) is
        // reserved for fatrop and left at zero.
        Index eval_BAbt(const Scalar * /*states_kp1*/, const Scalar * /*inputs_k*/,
                        const Scalar * /*states_k*/, MAT *res, const Index /*k*/) override
        {
            fatrop::blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0); // zero everything

            // --- B^T  (rows 0..nu-1, i.e. d f / d u) -----------------------------
            //   f affects vx,vy through fx,fy:  d vx_{k+1}/d fx = dt/m, etc.
            //   B = [[0,0],[0,0],[dt/m,0],[0,dt/m]]  ->  B^T row=input, col=state
            fatrop::blasfeo_matel_wrap(res, 0, 2) = dt_ / m_; // d(vx')/d(fx)
            fatrop::blasfeo_matel_wrap(res, 1, 3) = dt_ / m_; // d(vy')/d(fy)

            // --- A^T  (rows nu..nu+nx-1, i.e. d f / d x) -------------------------
            //   A = I + dt on (px<-vx),(py<-vy).  Placed at row offset nu = NU.
            fatrop::blasfeo_diare_wrap(NX, 1.0, res, NU, 0); // identity part of A^T
            fatrop::blasfeo_matel_wrap(res, NU + 2, 0) = dt_; // d(px')/d(vx)
            fatrop::blasfeo_matel_wrap(res, NU + 3, 1) = dt_; // d(py')/d(vy)
            return 0;
        }

        // Dynamics residual in the required form  b = -x_{k+1} + f(x_k,u_k).
        Index eval_b(const Scalar *states_kp1, const Scalar *inputs_k,
                     const Scalar *states_k, Scalar *res, const Index /*k*/) override
        {
            res[0] = -states_kp1[0] + states_k[0] + dt_ * states_k[2];
            res[1] = -states_kp1[1] + states_k[1] + dt_ * states_k[3];
            res[2] = -states_kp1[2] + states_k[2] + dt_ * inputs_k[0] / m_;
            res[3] = -states_kp1[3] + states_k[3] + dt_ * inputs_k[1] / m_;
            return 0;
        }

        // --- objective -----------------------------------------------------------

        // Scalar objective at stage k:  obj_scale * (fx^2 + fy^2).
        Index eval_L(const Scalar *objective_scale, const Scalar *inputs_k,
                     const Scalar * /*states_k*/, Scalar *res, const Index k) override
        {
            if (k == K_ - 1)
                *res = 0.0; // no input at terminal stage
            else
                *res = objective_scale[0] *
                       (inputs_k[0] * inputs_k[0] + inputs_k[1] * inputs_k[1]);
            return 0;
        }

        // Gradient of the objective (NOT the Lagrangian) w.r.t. (u_k, x_k).
        Index eval_rq(const Scalar *objective_scale, const Scalar *inputs_k,
                      const Scalar * /*states_k*/, Scalar *res, const Index k) override
        {
            const Index nu = get_nu(k);
            // states never appear in the cost -> their gradient block is zero.
            for (Index i = 0; i < nu + NX; ++i)
                res[i] = 0.0;
            if (nu > 0)
            {
                res[0] = 2.0 * objective_scale[0] * inputs_k[0];
                res[1] = 2.0 * objective_scale[0] * inputs_k[1];
            }
            return 0;
        }

        // Hessian of the Lagrangian w.r.t. (u_k, x_k), transposed & stored as the
        // top (nu+nx) x (nu+nx) block of res (last row reserved -> left at zero).
        //
        // Contributions:
        //   * cost:      d^2/du^2 (fx^2+fy^2) = 2*I_2   (in the input block)
        //   * dynamics:  linear  -> 0
        //   * obstacle:  lam_ineq * d^2 g_ineq / dx^2,  with
        //                g_ineq = (px-cx)^2 + (py-cy)^2  ->  Hessian = 2*I on (px,py)
        Index eval_RSQrqt(const Scalar *objective_scale, const Scalar * /*inputs_k*/,
                          const Scalar * /*states_k*/, const Scalar * /*lam_dyn_k*/,
                          const Scalar * /*lam_eq_k*/, const Scalar *lam_eq_ineq_k,
                          MAT *res, const Index k) override
        {
            fatrop::blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            const Index nu = get_nu(k);

            // cost curvature (input block only)
            if (nu > 0)
                fatrop::blasfeo_diare_wrap(NU, 2.0 * objective_scale[0], res, 0, 0);

            // obstacle curvature: add lam_ineq * 2 on the px and py diagonal entries.
            // px is local index (nu + 0), py is (nu + 1).
            const Scalar lam = lam_eq_ineq_k[0];
            fatrop::blasfeo_matel_wrap(res, nu + 0, nu + 0) += 2.0 * lam;
            fatrop::blasfeo_matel_wrap(res, nu + 1, nu + 1) += 2.0 * lam;
            return 0;
        }

        // --- equality constraints (boundary conditions) --------------------------

        // g_eq value:  x_0 - x_start   (k==0)   or   x_{K-1} - x_goal (k==K-1).
        Index eval_g(const Scalar * /*inputs_k*/, const Scalar *states_k, Scalar *res,
                     const Index k) override
        {
            if (k == 0)
                for (Index i = 0; i < NX; ++i)
                    res[i] = states_k[i] - x_start_[i];
            else if (k == K_ - 1)
                for (Index i = 0; i < NX; ++i)
                    res[i] = states_k[i] - x_goal_[i];
            return 0;
        }

        // g_eq Jacobian (transposed). The constraint is x = const, so the Jacobian
        // w.r.t. the state is the identity; w.r.t. the input it is zero. The state
        // block starts at row offset nu.
        Index eval_Ggt(const Scalar * /*inputs_k*/, const Scalar * /*states_k*/, MAT *res,
                       const Index k) override
        {
            fatrop::blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            if (k == 0 || k == K_ - 1)
                fatrop::blasfeo_diare_wrap(NX, 1.0, res, get_nu(k), 0);
            return 0;
        }

        // --- inequality constraint (obstacle) ------------------------------------

        // g_ineq value:  (px-cx)^2 + (py-cy)^2.
        Index eval_gineq(const Scalar * /*inputs_k*/, const Scalar *states_k, Scalar *res,
                         const Index /*k*/) override
        {
            const Scalar dx = states_k[0] - cx_;
            const Scalar dy = states_k[1] - cy_;
            res[0] = dx * dx + dy * dy;
            return 0;
        }

        // g_ineq Jacobian (transposed). Nonzero only w.r.t. px, py:
        //   d g / d px = 2 (px - cx),  d g / d py = 2 (py - cy).
        // Placed in the state block (row offset nu), single column 0.
        Index eval_Ggt_ineq(const Scalar * /*inputs_k*/, const Scalar *states_k, MAT *res,
                            const Index k) override
        {
            fatrop::blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            const Index nu = get_nu(k);
            fatrop::blasfeo_matel_wrap(res, nu + 0, 0) = 2.0 * (states_k[0] - cx_);
            fatrop::blasfeo_matel_wrap(res, nu + 1, 0) = 2.0 * (states_k[1] - cy_);
            return 0;
        }

        // Bounds on g_ineq:  r_safe^2 <= g_ineq <= +inf.
        Index get_bounds(Scalar *lower, Scalar *upper, const Index /*k*/) const override
        {
            lower[0] = r_safe_ * r_safe_;
            upper[0] = std::numeric_limits<Scalar>::infinity(); // one-sided
            return 0;
        }

        // --- initial guess -------------------------------------------------------

        // Straight-line interpolation of the position from start to goal, velocities
        // linearly blended too. This warm start may cut *through* the obstacle; the
        // solver then pushes it out thanks to the inequality above.
        Index get_initial_xk(Scalar *xk, const Index k) const override
        {
            const Scalar a = (K_ > 1) ? static_cast<Scalar>(k) / (K_ - 1) : 0.0;
            for (Index i = 0; i < NX; ++i)
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

        // --- accessors used by the driver for reporting --------------------------
        Index K() const { return K_; }
        Scalar dt() const { return dt_; }
        Scalar cx() const { return cx_; }
        Scalar cy() const { return cy_; }
        Scalar r_safe() const { return r_safe_; }

    private:
        const Index K_;
        const Scalar dt_;
        const Scalar m_;
        const std::array<Scalar, NX> x_start_;
        const std::array<Scalar, NX> x_goal_;
        const Scalar cx_;
        const Scalar cy_;
        const Scalar r_safe_;
    };

} // namespace lse_fatrop

#endif // LSE_FATROP_POINT_MASS_OBSTACLE_OCP_HPP
