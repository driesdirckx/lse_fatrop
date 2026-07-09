//
// nnz_report.hpp
//
// Count the total number of structural nonzeros in the constraint Jacobian and
// the Lagrangian Hessian that a hand-written fatrop OCP produces, summed over
// the whole horizon. This exists so the low-level C++ interface can be compared,
// number-for-number, against the CasADi Opti reference in
// casadi_opti_reference.py, which prints
//
//     jacobian(opti.g, opti.x).nnz()      ("Jacobian nonzeros")
//     hessian(lagrangian, opti.x).nnz()   ("Hessian nonzeros")
//
// "Structural" here means: evaluate every block at a generic, non-degenerate
// point (nonzero states / inputs / multipliers) and count the entries whose
// magnitude exceeds a tiny threshold. Evaluating away from special points avoids
// accidental cancellation, so what we count is the sparsity pattern the solver
// actually assembles.
//
// Jacobian pieces (per stage, w.r.t. the stage variables (u_k, x_k)):
//   * dynamics    : [B^T; A^T] stored in BAbt, PLUS the implicit -I coupling to
//                   x_{k+1} (fatrop stores it structurally; CasADi counts it, so
//                   we add nx_{k+1} per dynamics stage to match).
//   * equalities  : Ggt
//   * inequalities: Ggt_ineq
// Hessian piece (per stage): the (nu+nx) x (nu+nx) top-left block of RSQrqt.
// (The Lagrangian Hessian is block-diagonal per stage here because the dynamics
// are linear, so summing the per-stage blocks reproduces the global pattern.)
//

#ifndef LSE_FATROP_NNZ_REPORT_HPP
#define LSE_FATROP_NNZ_REPORT_HPP

#include <cmath>
#include <iostream>
#include <vector>

#include <fatrop/fatrop.hpp>
#include <fatrop/linear_algebra/matrix.hpp>

namespace lse_fatrop
{
    struct NnzCounts
    {
        long jacobian = 0;
        long hessian = 0;
    };

    inline NnzCounts count_nnz(fatrop::OcpAbstract &ocp)
    {
        using fatrop::Index;
        using fatrop::MatRealAllocated;
        using fatrop::Scalar;

        const Index K = ocp.get_horizon_length();
        const Scalar tol = 1e-12;
        const Scalar obj_scale = 1.0;

        // Fill a buffer with distinct, nonzero, non-degenerate values.
        auto fill = [](std::vector<Scalar> &v, Index n, Scalar base) {
            v.assign(n > 0 ? n : 1, 0.0);
            for (Index i = 0; i < n; ++i)
                v[i] = base + 0.13 * static_cast<Scalar>(i + 1);
        };
        auto count_block = [&](MatRealAllocated &M, Index rows, Index cols) {
            long nz = 0;
            for (Index i = 0; i < rows; ++i)
                for (Index j = 0; j < cols; ++j)
                    if (std::fabs(M(i, j)) > tol)
                        ++nz;
            return nz;
        };

        NnzCounts c;
        std::vector<Scalar> xk, xkp1, uk, lam_dyn, lam_eq, lam_ineq;

        for (Index k = 0; k < K; ++k)
        {
            const Index nx = ocp.get_nx(k);
            const Index nu = ocp.get_nu(k);
            const Index ng = ocp.get_ng(k);
            const Index ngi = ocp.get_ng_ineq(k);

            fill(xk, nx, 0.30);
            fill(uk, nu, 0.20);
            fill(lam_eq, ng, 1.0);
            fill(lam_ineq, ngi, 1.0);

            // --- Hessian: (nu+nx) x (nu+nx) top-left block of RSQrqt ------------
            {
                fill(lam_dyn, nx, 1.0);
                MatRealAllocated M(nu + nx + 1, nu + nx);
                ocp.eval_RSQrqt(&obj_scale, uk.data(), xk.data(), lam_dyn.data(),
                                lam_eq.data(), lam_ineq.data(), &M.mat(), k);
                c.hessian += count_block(M, nu + nx, nu + nx);
            }

            // --- equality Jacobian Ggt: (nu+nx) rows x ng cols -----------------
            if (ng > 0)
            {
                MatRealAllocated M(nu + nx + 1, ng);
                ocp.eval_Ggt(uk.data(), xk.data(), &M.mat(), k);
                c.jacobian += count_block(M, nu + nx, ng);
            }

            // --- inequality Jacobian Ggt_ineq: (nu+nx) rows x ng_ineq cols -----
            if (ngi > 0)
            {
                MatRealAllocated M(nu + nx + 1, ngi);
                ocp.eval_Ggt_ineq(uk.data(), xk.data(), &M.mat(), k);
                c.jacobian += count_block(M, nu + nx, ngi);
            }

            // --- dynamics Jacobian: BAbt [B^T;A^T] + implicit -I on x_{k+1} -----
            if (k < K - 1)
            {
                const Index nxp1 = ocp.get_nx(k + 1);
                fill(xkp1, nxp1, 0.40);
                MatRealAllocated M(nu + nx + 1, nxp1);
                ocp.eval_BAbt(xkp1.data(), uk.data(), xk.data(), &M.mat(), k);
                c.jacobian += count_block(M, nu + nx, nxp1);
                c.jacobian += nxp1; // d(dynamics residual)/d(x_{k+1}) = -I
            }
        }
        return c;
    }

    // Print the Jacobian / Hessian nonzero totals in a form directly comparable
    // to the CasADi reference output.
    inline void report_nnz(fatrop::OcpAbstract &ocp)
    {
        const NnzCounts c = count_nnz(ocp);
        std::cout << "Jacobian nonzeros (total) : " << c.jacobian << "\n";
        std::cout << "Hessian  nonzeros (total) : " << c.hessian << "\n";
    }

} // namespace lse_fatrop

#endif // LSE_FATROP_NNZ_REPORT_HPP
