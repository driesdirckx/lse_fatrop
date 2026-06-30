//
// main.cpp
//
// Driver for the hand-written point-mass / obstacle OCP.
//
// It shows the full life cycle of the fatrop low-level interface:
//   1. instantiate the user OcpAbstract                  (PointMassObstacleOcp)
//   2. wrap it in an NlpOcp and build the IP algorithm    (IpAlgBuilder)
//   3. (optionally) set solver options                    (OptionRegistry)
//   4. solve                                              (ipalg->optimize())
//   5. pull the primal trajectory back out                (IpData / iterate)
//   6. report + dump a CSV that can be plotted
//

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

#include <fatrop/fatrop.hpp>

#include "point_mass_obstacle_ocp.hpp"

using namespace fatrop;
using lse_fatrop::NU;
using lse_fatrop::NX;
using lse_fatrop::PointMassObstacleOcp;

int main()
{
    // ---- 1. problem definition ---------------------------------------------
    const Index K = 41;     // 41 knots -> 40 intervals
    const Scalar dt = 0.1;  // s
    const Scalar mass = 1.0;

    const std::array<Scalar, NX> x_start = {0.0, 0.0, 0.0, 0.0}; // at rest in origin
    const std::array<Scalar, NX> x_goal = {2.0, 2.0, 0.0, 0.0};  // at rest in (2,2)
    const std::array<Scalar, 2> obstacle = {1.0, 1.0};           // dead centre of the path
    const Scalar r_safe = 0.5;                                   // distance to keep

    auto ocp = std::make_shared<PointMassObstacleOcp>(K, dt, mass, x_start, x_goal,
                                                       obstacle, r_safe);

    // ---- 2./3. build the interior-point algorithm --------------------------
    // NlpOcp adapts an OcpAbstract to the generic NLP the solver consumes.
    OptionRegistry options;
    IpAlgBuilder<OcpType> builder(std::make_shared<NlpOcp>(ocp));
    auto ipalg = builder.with_options_registry(&options).build();

    options.set_option("max_iter", Index(500));
    options.set_option("constr_viol_tol", Scalar(1e-8));

    // ---- 4. solve ----------------------------------------------------------
    const IpSolverReturnFlag ret = ipalg->optimize();
    const bool converged = (ret == IpSolverReturnFlag::Success ||
                            ret == IpSolverReturnFlag::StopAtAcceptablePoint);

    // ---- 5. extract the primal solution ------------------------------------
    // The primal vector is laid out per stage as (u_0,x_0, u_1,x_1, ..., x_{K-1}).
    // info.offsets_primal_x[k] / offsets_primal_u[k] give the flat index of x_k / u_k.
    auto data = builder.get_ipdata();
    const auto &info = data->info();
    const VecRealView &x = data->current_iterate().primal_x();

    std::cout << "\n=== Point-mass point-to-point with circular obstacle ===\n";
    std::cout << "Return flag : " << int(ret) << "  (converged=" << converged << ")\n";
    std::cout << "Iterations  : " << data->iteration_number() << "\n";
    std::cout << "Obstacle    : centre (" << ocp->cx() << ", " << ocp->cy()
              << "), keep-out radius " << ocp->r_safe() << "\n\n";

    // ---- 6. report + CSV ---------------------------------------------------
    std::ofstream csv("trajectory.csv");
    csv << "k,t,px,py,vx,vy,fx,fy,dist_to_obstacle\n";

    std::cout << std::fixed << std::setprecision(4);
    Scalar min_clearance = std::numeric_limits<Scalar>::infinity();
    for (Index k = 0; k < K; ++k)
    {
        const Scalar *xk = x.data() + info.offsets_primal_x[k];
        const Scalar dist =
            std::sqrt((xk[0] - ocp->cx()) * (xk[0] - ocp->cx()) +
                      (xk[1] - ocp->cy()) * (xk[1] - ocp->cy()));
        min_clearance = std::min(min_clearance, dist);

        Scalar fx = 0.0, fy = 0.0;
        const bool has_u = (k < K - 1);
        if (has_u)
        {
            const Scalar *uk = x.data() + info.offsets_primal_u[k];
            fx = uk[0];
            fy = uk[1];
        }

        csv << k << ',' << k * dt << ',' << xk[0] << ',' << xk[1] << ',' << xk[2]
            << ',' << xk[3] << ',' << fx << ',' << fy << ',' << dist << '\n';

        if (k % 5 == 0 || k == K - 1) // print every 5th knot to keep it short
        {
            std::cout << "k=" << std::setw(2) << k << "  p=(" << std::setw(7) << xk[0]
                      << ", " << std::setw(7) << xk[1] << ")  v=(" << std::setw(7)
                      << xk[2] << ", " << std::setw(7) << xk[3] << ")  dist=" << dist
                      << (dist + 1e-9 < ocp->r_safe() ? "  <-- inside!" : "") << '\n';
        }
    }
    csv.close();

    std::cout << "\nMinimum clearance to obstacle centre : " << min_clearance
              << "  (required >= " << ocp->r_safe() << ")\n";
    std::cout << "Trajectory written to trajectory.csv\n";

    return converged ? 0 : 1;
}
