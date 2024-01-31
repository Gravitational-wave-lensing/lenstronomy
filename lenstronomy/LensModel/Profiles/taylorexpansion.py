__author__ = "Jason the awesome"

import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase

__all__ = ["TaylorExpansion"]


class TaylorExpansion(LensProfileBase):
    """A single mass sheet (external convergence)"""

    model_name = "TAYLOREXPANSION"
    param_names = ["K0", "K2", "ra_0", "dec_0"]
    lower_limit_default = {"K0": -10, "K2": -1, "ra_0": -100, "dec_0": -100}
    upper_limit_default = {"K0": 10, "K2": 1, "ra_0": 100, "dec_0": 100}

    def function(self, x, y, K0, K2, ra_0=0, dec_0=0):
        """Lensing potential.

        :param x: x-coordinate
        :param y: y-coordinate
        :param K0: (external) convergence
        :return: lensing potential
        """
        # theta, phi = param_util.cart2polar(x - ra_0, y - dec_0)
        x_ = x - ra_0
        y_ = y - dec_0
        f_ = 0.5 * (1-K0)*x_**2 + 0.5*y_**2+K2*x_*y_**2
        return f_

    def derivatives(self, x, y, K0, K2, ra_0=0, dec_0=0):
        """Deflection angle.

        :param x: x-coordinate
        :param y: y-coordinate
        :param K0: (external) convergence
        :return: deflection angles (first order derivatives)
        """
        x_ = x - ra_0
        y_ = y - dec_0
        f_x = K0 * x_ + K2 * y_**2
        f_y = K0 * y_ + 2*K2*x_*y_
        return f_x, f_y

    def hessian(self, x, y, K0, K2, ra_0=0, dec_0=0):
        """Hessian matrix.

        :param x: x-coordinate
        :param y: y-coordinate
        :param K0: external convergence
        :param ra_0: zero point of polynomial expansion (no deflection added)
        :param dec_0: zero point of polynomial expansion (no deflection added)
        :return: second order derivatives f_xx, f_xy, f_yx, f_yy
        """
        x_ = x - ra_0
        y_ = y - dec_0
        K0 = K0
        f_xx = K0
        f_yy = K0 + 2*K2*x_ 
        f_xy = 2*K2*y_
        return f_xx, f_xy, f_xy, f_yy
