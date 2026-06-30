//
// lse.hpp
//
// CasADi-free, analytical Log-Sum-Exp (LSE) smooth-min / smooth-max with their
// exact first- and second-order derivatives.
//
// This reproduces the formulation used in
//   embeddedcbf/helpers.py :  smooth_max / smooth_min
//   embeddedcbf/lse_cbf_mpc_fatrop.py
// but evaluates the value, Jacobian and Hessian from closed-form expressions —
// no CasADi graph is built or traversed for the constraint.
//
// Definitions (helpers.py) ----------------------------------------------------
//
//   basic(z)        = max_i z_i + log sum_i exp(z_i - max_j z_j)   == log sum_i exp(z_i)
//   smooth_max(e,a) = (1/a) * basic(a*e)        =  (1/a) log sum_i exp( a*e_i)
//   smooth_min(e,a) = -smooth_max(-e,a)         = -(1/a) log sum_i exp(-a*e_i)
//
// The "max_i z_i" subtraction is the standard log-sum-exp shift: it does not
// change the mathematical value, it only keeps exp() from overflowing. We keep
// it here for the same numerical-stability reason.
//
// Derivatives -----------------------------------------------------------------
//
// Let p = softmax(a*e),  i.e.  p_i = exp(a*e_i) / sum_j exp(a*e_j).  Then
//
//   d smooth_max / d e_i        = p_i                          (a probability vector)
//   d^2 smooth_max / d e_i d e_j = a * ( diag(p) - p p^T )_ij
//
// Let q = softmax(-a*e),  i.e.  q_i = exp(-a*e_i) / sum_j exp(-a*e_j).  Then
//
//   d smooth_min / d e_i        = q_i
//   d^2 smooth_min / d e_i d e_j = -a * ( diag(q) - q q^T )_ij
//
// Both p and q sum to 1 and are non-negative; the e-Hessians are (a times) the
// covariance of a categorical distribution, hence PSD for smooth_max and NSD
// for smooth_min.
//
// Chaining to decision variables ----------------------------------------------
//
// If the constraint argument is a function e = e(z) of the decision variables z
// (here z = (u_k, x_k)), with Jacobian  J = de/dz  (n_e x n_z), and we define the
// scalar constraint  h(z) = smooth_*(e(z)), then by the chain rule
//
//   grad_z h = J^T * grad_e
//   hess_z h = J^T * hess_e * J  +  sum_i (grad_e)_i * d^2 e_i / dz^2
//
// The second term vanishes when e is affine in z (our example). For a nonlinear
// e (e.g. the rotated robot vertices of the CBF example) you simply add the
// (grad_e)_i-weighted Hessians of each e_i — see the README.
//

#ifndef LSE_FATROP_LSE_HPP
#define LSE_FATROP_LSE_HPP

#include <cmath>

#include <fatrop/context/context.hpp> // fatrop::Scalar, fatrop::Index

namespace lse_fatrop
{
    namespace lse
    {
        using fatrop::Index;
        using fatrop::Scalar;

        /// Numerically-stable softmax with inverse-temperature `beta`:
        ///     w_i = exp(beta*e_i) / sum_j exp(beta*e_j)
        /// and (as a by-product) the log-sum-exp value
        ///     lse = (1/beta) * log sum_j exp(beta*e_j).
        ///
        /// Used with beta = +alpha for smooth_max (w == p) and beta = -alpha for
        /// smooth_min (w == q). `w` must point to n writable Scalars.
        ///
        /// @return the smooth value (1/beta)*log sum exp(beta*e).
        inline Scalar softmax_with_value(const Scalar *e, const Index n, const Scalar beta,
                                         Scalar *w)
        {
            // shift by the max of (beta*e) for stability
            Scalar zmax = beta * e[0];
            for (Index i = 1; i < n; ++i)
            {
                const Scalar zi = beta * e[i];
                if (zi > zmax)
                    zmax = zi;
            }
            Scalar sum = 0.0;
            for (Index i = 0; i < n; ++i)
            {
                w[i] = std::exp(beta * e[i] - zmax); // un-normalised weight
                sum += w[i];
            }
            const Scalar inv_sum = 1.0 / sum;
            for (Index i = 0; i < n; ++i)
                w[i] *= inv_sum; // normalise -> softmax
            // (1/beta) * log sum exp(beta*e) = (1/beta) * ( zmax + log(sum) )
            return (zmax + std::log(sum)) / beta;
        }

        /// smooth_max value only.  smooth_max(e,a) = (1/a) log sum exp(a*e).
        inline Scalar smooth_max(const Scalar *e, const Index n, const Scalar alpha)
        {
            Scalar zmax = e[0];
            for (Index i = 1; i < n; ++i)
                if (e[i] > zmax)
                    zmax = e[i];
            Scalar sum = 0.0;
            for (Index i = 0; i < n; ++i)
                sum += std::exp(alpha * (e[i] - zmax));
            return zmax + std::log(sum) / alpha;
        }

        /// smooth_min value only.  smooth_min(e,a) = -smooth_max(-e,a).
        inline Scalar smooth_min(const Scalar *e, const Index n, const Scalar alpha)
        {
            Scalar zmin = e[0];
            for (Index i = 1; i < n; ++i)
                if (e[i] < zmin)
                    zmin = e[i];
            Scalar sum = 0.0;
            for (Index i = 0; i < n; ++i)
                sum += std::exp(-alpha * (e[i] - zmin));
            return zmin - std::log(sum) / alpha;
        }

        /// Value + gradient (w.r.t. e) of smooth_max. `p` receives the softmax(a*e)
        /// weights, which ARE the gradient d smooth_max / d e.
        /// @return smooth_max value.
        inline Scalar smooth_max_grad(const Scalar *e, const Index n, const Scalar alpha,
                                      Scalar *p)
        {
            return softmax_with_value(e, n, +alpha, p);
        }

        /// Value + gradient (w.r.t. e) of smooth_min. `q` receives the softmax(-a*e)
        /// weights, which ARE the gradient d smooth_min / d e.
        /// @return smooth_min value.
        inline Scalar smooth_min_grad(const Scalar *e, const Index n, const Scalar alpha,
                                      Scalar *q)
        {
            // smooth_min value = (1/(-alpha)) log sum exp(-alpha e) == softmax_with_value's
            // return for beta=-alpha. The weights q are exactly d smooth_min / d e.
            return softmax_with_value(e, n, -alpha, q);
        }

        /// Hessian (w.r.t. e) entry of smooth_max:  a*(diag(p) - p p^T)_{ij}.
        inline Scalar smooth_max_hess_e(const Scalar *p, const Scalar alpha, const Index i,
                                        const Index j)
        {
            const Scalar diag = (i == j) ? p[i] : 0.0;
            return alpha * (diag - p[i] * p[j]);
        }

        /// Hessian (w.r.t. e) entry of smooth_min:  -a*(diag(q) - q q^T)_{ij}.
        inline Scalar smooth_min_hess_e(const Scalar *q, const Scalar alpha, const Index i,
                                        const Index j)
        {
            const Scalar diag = (i == j) ? q[i] : 0.0;
            return -alpha * (diag - q[i] * q[j]);
        }

    } // namespace lse
} // namespace lse_fatrop

#endif // LSE_FATROP_LSE_HPP
