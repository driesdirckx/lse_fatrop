//
// main_lse.cpp
//
// Driver for the LSE (log-sum-exp) obstacle example. Identical solver plumbing
// to main.cpp; the only difference is the OCP, whose obstacle constraint is a
// smooth-max of half-plane distances with analytically-derived Jacobian/Hessian
// (see point_mass_lse_ocp.hpp + lse.hpp).
//

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <fatrop/fatrop.hpp>

#include "point_mass_lse_ocp.hpp"

using namespace fatrop;
using lse_fatrop::NX_LSE;
using lse_fatrop::PointMassLseOcp;

int main()
{
    // ---- 1. problem definition ---------------------------------------------
    const Index K = 41;
    const Scalar dt = 0.1;
    const Scalar mass = 1.0;

    const std::array<Scalar, NX_LSE> x_start = {0.0, 0.0, 0.0, 0.0};
    const std::array<Scalar, NX_LSE> x_goal = {2.0, 2.0, 0.0, 0.0};

    // Axis-aligned square obstacle centred at (cx,cy), half-size h, as a polytope
    // { p : A_l . p <= b_l }.  e_l = A_l . p - b_l  ->  e_l > 0 means "outside l".
    const Scalar cx = 1.0, cy = 1.0, h = 0.45;
    const std::vector<std::array<Scalar, 2>> obs_A = {
        {1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}};
    const std::vector<Scalar> obs_b = {cx + h, -(cx - h), cy + h, -(cy - h)};

    // Sharpness from the reference code: alpha = log(n_e) / max_approx.
    const Scalar max_approx = 2e-2;
    const Scalar alpha = std::log(static_cast<Scalar>(obs_b.size())) / max_approx;
    const Scalar margin = 0.05; // keep-out margin on smooth_max(e)

    auto ocp = std::make_shared<PointMassLseOcp>(K, dt, mass, x_start, x_goal, obs_A,
                                                 obs_b, alpha, margin);

    // ---- 2./3. build solver -------------------------------------------------
    OptionRegistry options;
    IpAlgBuilder<OcpType> builder(std::make_shared<NlpOcp>(ocp));
    auto ipalg = builder.with_options_registry(&options).build();
    options.set_option("max_iter", Index(500));
    options.set_option("constr_viol_tol", Scalar(1e-8));

    // ---- 4. solve ----------------------------------------------------------
    const IpSolverReturnFlag ret = ipalg->optimize();
    const bool converged = (ret == IpSolverReturnFlag::Success ||
                            ret == IpSolverReturnFlag::StopAtAcceptablePoint);

    // ---- 5./6. extract + report --------------------------------------------
    auto data = builder.get_ipdata();
    const auto &info = data->info();
    const VecRealView &x = data->current_iterate().primal_x();

    std::cout << "\n=== Point-mass p2p with LSE smooth-max polygon obstacle ===\n";
    std::cout << "Return flag : " << int(ret) << "  (converged=" << converged << ")\n";
    std::cout << "Iterations  : " << data->iteration_number() << "\n";
    std::cout << "Obstacle    : square centre (" << cx << ", " << cy << "), half-size " << h
              << "\n";
    std::cout << "LSE alpha   : " << alpha << "   margin : " << margin << "\n\n";

    std::ofstream csv("trajectory_lse.csv");
    csv << "k,t,px,py,vx,vy,fx,fy,smooth_max,true_max_outside\n";

    std::cout << std::fixed << std::setprecision(4);
    Scalar min_true = std::numeric_limits<Scalar>::infinity();
    for (Index k = 0; k < K; ++k)
    {
        const Scalar *xk = x.data() + info.offsets_primal_x[k];
        std::vector<Scalar> e(obs_b.size());
        for (std::size_t l = 0; l < obs_b.size(); ++l)
            e[l] = obs_A[l][0] * xk[0] + obs_A[l][1] * xk[1] - obs_b[l];
        const Scalar sm =
            lse_fatrop::lse::smooth_max(e.data(), static_cast<Index>(e.size()), alpha);
        const Scalar tm = ocp->true_max_outside(xk[0], xk[1]);
        min_true = std::min(min_true, tm);

        Scalar fx = 0.0, fy = 0.0;
        if (k < K - 1)
        {
            const Scalar *uk = x.data() + info.offsets_primal_u[k];
            fx = uk[0];
            fy = uk[1];
        }
        csv << k << ',' << k * dt << ',' << xk[0] << ',' << xk[1] << ',' << xk[2] << ','
            << xk[3] << ',' << fx << ',' << fy << ',' << sm << ',' << tm << '\n';

        if (k % 5 == 0 || k == K - 1)
            std::cout << "k=" << std::setw(2) << k << "  p=(" << std::setw(7) << xk[0]
                      << ", " << std::setw(7) << xk[1] << ")  smooth_max=" << std::setw(8)
                      << sm << "  true_max_outside=" << std::setw(8) << tm
                      << (tm < 0.0 ? "  <-- inside obstacle!" : "") << '\n';
    }
    csv.close();

    std::cout << "\nMinimum true_max_outside over trajectory : " << min_true
              << "  (>= 0 means collision-free)\n";
    std::cout << "Trajectory written to trajectory_lse.csv\n";
    return converged ? 0 : 1;
}
