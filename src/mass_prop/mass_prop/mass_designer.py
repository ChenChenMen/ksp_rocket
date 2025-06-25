"""Rocket Mass Design Tool."""

import numpy as np

from rocket_util.constant import G0
from rocket_util.logconfig import create_logger


LOG = create_logger(__name__)


class TSTODesign:
    """Two stage to orbit launch vehicle mass designer."""

    def __init__(self, delta_v, payload_mass):
        """Instantiate a TSTO design based on total delta v and payload mass.

        Variable convention assumes upper stage to be at the leftmost
        """
        # Input variables
        self.total_delta_v = delta_v
        self.target_payload_mass = payload_mass

        # Design iterating variables
        self.structural_ratio = None
        self.engine_isp = None

        # Delta v distribution solution space
        self.delta_v_distribution = None

        # Stage specific variables
        self.mass_prop = None

    def optimize_dv_distribution(self, structural_ratio: np.ndarray, engine_isp: np.ndarray):
        """Set up numerical optimization for input structural ratios (only for TSTO)."""
        # Check input consistency
        if structural_ratio.size != 2 or engine_isp.size != 2:
            raise ValueError(
                f"Except a two stage to orbit, but structural ratio size: {structural_ratio.size}, "
                f"and engine isp size: {engine_isp.size}. Should be exactly 2."
            )

        # Compute gradient and approximated hessian
        def get_grad_hess(percent_dv_stg2: float):
            """Obtain gradient vector and estimated hessian matrix."""
            percent_dv_distribution = np.array([percent_dv_stg2, 1 - percent_dv_stg2])
            # Set up numeric middle steps
            exhaust_velo = G0 * engine_isp / self.total_delta_v
            inv_mu_mass_ratio = np.exp(-percent_dv_distribution / exhaust_velo)
            mass_ratio_diff = inv_mu_mass_ratio - structural_ratio
            inv_mass_ratio_diff = 1 / mass_ratio_diff

            # Compute objective function value
            objective_func = np.prod(inv_mass_ratio_diff)

            # Compute gradient
            gradient_derived_parts = inv_mass_ratio_diff * inv_mu_mass_ratio / exhaust_velo
            gradient_parts = objective_func * gradient_derived_parts
            gradient = gradient_parts[0] - gradient_parts[1]

            # Compute hessian
            hessian_derived_parts = structural_ratio * inv_mu_mass_ratio / gradient_derived_parts
            hessian_parts = gradient_parts * hessian_derived_parts
            hessian = hessian_parts[0] - hessian_parts[1]

            return gradient, hessian

        # Initial guess of stage 2 delta v distribution to be 50%
        curr_percent_dv_stg2, prev_percent_dv_stg2 = 0.5, 2

        # Define tolerance
        atol, max_iter = 1e-7, 500
        # Adam coefficent setup
        learn_rate, gradmean_eff, gradvari_eff = 0.01, 0.9, 0.999
        est_gradmean, est_gradvari, epsilon = 0, 0, 1e-8
        for iter in range(max_iter):
            # Exit criteria check
            if np.abs(curr_percent_dv_stg2 - prev_percent_dv_stg2) < atol:
                percent_dv_stg1 = 1 - curr_percent_dv_stg2
                LOG.info(
                    f"Find minimization solution with delta v distribution in {iter} iterations -\n"
                    f"stage 2: {(curr_percent_dv_stg2 * 100):.2f}% {(curr_percent_dv_stg2 * self.total_delta_v):.2f} m/s\n"
                    f"stage 1: {(percent_dv_stg1 * 100):.2f}% {(percent_dv_stg1 * self.total_delta_v):.2f} m/s"
                )
                break

            elif iter == max_iter - 1:
                percent_dv_stg1 = 1 - curr_percent_dv_stg2
                LOG.warning(
                    f"Max iterations {max_iter} reached. Exiting the iterative optimization with\n"
                    f"stage 2: {(curr_percent_dv_stg2 * 100):.2f}% {(curr_percent_dv_stg2 * self.total_delta_v):.2f} m/s\n"
                    f"stage 1: {(percent_dv_stg1 * 100):.2f}% {(percent_dv_stg1 * self.total_delta_v):.2f} m/s"
                )
                break

            # Adaptive moment estimation
            prev_percent_dv_stg2 = curr_percent_dv_stg2
            grad, _ = get_grad_hess(curr_percent_dv_stg2)
            # Moment updates
            est_gradmean = gradmean_eff * est_gradmean + (1 - gradmean_eff) * grad
            est_gradvari = gradvari_eff * est_gradvari + (1 - gradvari_eff) * grad**2
            # Bias correction to moment
            cor_est_gradmean = est_gradmean / (1 - gradmean_eff)
            cor_est_gradvari = est_gradvari / (1 - gradvari_eff)
            curr_percent_dv_stg2 -= learn_rate * cor_est_gradmean / (np.sqrt(cor_est_gradvari) + epsilon)

        self.structural_ratio = structural_ratio
        self.engine_isp = engine_isp
        self.delta_v_distribution = np.array([curr_percent_dv_stg2, 1 - curr_percent_dv_stg2]) * self.total_delta_v
        # Update the mass properties
        self._evaluate_stage_mass()

    def _evaluate_stage_mass(self):
        """Evaluate stage mass based on current delta v distribution."""
        curr_total_mass = self.target_payload_mass
        mass_prop = np.zeros(shape=(self.delta_v_distribution.size, 3))
        # Iteratively compute mass property
        for stage_ind, (stage_delta_v, stage_struc_ratio, stage_isp) in enumerate(
            zip(self.delta_v_distribution, self.structural_ratio, self.engine_isp)
        ):
            # Compute the inverse of mass ratio
            inv_mu_mass_ratio = np.exp(-1 * stage_delta_v / G0 / stage_isp)

            # Compute stage total mass, structural mass, and propellant mass
            total_mass = curr_total_mass * (1 - stage_struc_ratio) / (inv_mu_mass_ratio - stage_struc_ratio)
            struc_mass = (total_mass - curr_total_mass) * stage_struc_ratio
            prop_mass = total_mass - struc_mass - curr_total_mass
            mass_prop[stage_ind, :] = [total_mass, struc_mass, prop_mass]
            curr_total_mass = total_mass

        # Record solved mass properties
        LOG.info(
            "Solved mass property table as shown: [total mass, structural mass, propellant mass]\n"
            + "\n".join([f"Stage {stage}: {stage_mass_prop}" for stage, stage_mass_prop in enumerate(mass_prop.tolist())])
        )
        self.mass_prop = mass_prop

    def __str__(self):
        """String representation of the TSTO vehicle designed state."""
        header = ["", "=" * 100, "TSTO Vehicle Delta V Designer", "=" * 100, ""]
        # Add design requirements
        content = [
            "DESIGN REQUIREMENTS",
            f"Total Delta V: {self.total_delta_v} km/s",
            f"Target Payload Mass: {self.target_payload_mass} kg",
            "",
            "HARDWARD REQUIREMENTS",
            f"Structural Ratio: stage 2 - {self.structural_ratio[0] * 100} %, stage 1 - {self.structural_ratio[1] * 100} %",
            f"Enginee Isp: stage 2 - {self.engine_isp[0]} s, stage 1 - {self.engine_isp[1]} s",
            "",
            "RESULT",
            f"Delta V Distribution: stage 2 - {self.delta_v_distribution[0]:.0f} km/s, stage 1 - {self.delta_v_distribution[1]:.0f} km/s",
        ]
        return "\n".join(header + content)


# Example test case that will be executed when running this file directly
if __name__ == "__main__":
    # Provide an example TSTO designer instance
    designer = TSTODesign(delta_v=9144, payload_mass=46)
    designer.optimize_dv_distribution(structural_ratio=np.array([0.096, 0.085]), engine_isp=np.array([370, 278]))
    LOG.info(designer)
