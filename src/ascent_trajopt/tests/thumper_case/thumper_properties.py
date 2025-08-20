"""Define Thumper launch vehicle properties.

Reference: https://aiaa.org/wp-content/uploads/2024/12/1st-place-missile-systems-design_2024_virginia-polytechnic-institute-and-state-university_design_report.pdf
"""

from rocket_util.constant import ConstantsProvider
from rocket_util.units import Q_


class ThumperStage1BurnProperties(ConstantsProvider):
    """Thumper launch vehicle properties."""

    def __init__(self, scaler=None):
        """Initialize Thumper properties with optional scaling."""
        super().__init__(scaler)
        self._stored_constant = {
            "max_thrust": 1.0e6,
            "dry_mass": Q_(244.6 + 68.5, "lb"),
            "isp": Q_(277.8, "sec"),
        }
