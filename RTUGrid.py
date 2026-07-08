import jax
import jax.numpy as jnp
import numpy as np
# import lineax as lx

from functools import partial
from jax.tree_util import register_pytree_node_class


@jax.jit
def interp1d(x, xp, fp):
    # a interp1d function that set method='scan_unrolled'
    # in jnp.serchsorted for faster computer times
    i = jnp.clip(
        jnp.searchsorted(
            xp,
            x,
            side='right',
            # this method keyword is the main driver in computation time
            # for the RTU-grid.
            method='scan_unrolled'
        ),
        1,
        len(xp) - 1
    )
    df = fp[i] - fp[i - 1]
    dx = xp[i] - xp[i - 1]
    delta = x - xp[i - 1]
    epsilon = jnp.spacing(jnp.finfo(xp.dtype).eps)
    dx0 = jax.lax.abs(dx) <= epsilon  # Prevent NaN gradients when `dx` is small.
    f = jnp.where(
        dx0,
        fp[i - 1],
        fp[i - 1] + (delta / jnp.where(dx0, 1, dx)) * df
    )
    return f


def spline_invert(
    ip_x_low_res,
    ip_y_low_res,
    ip_dy_low_res,
    ip_delta_x,
    x
):
    k_right = jnp.digitize(
        x,
        ip_x_low_res,
        # this method keyword is the main driver in computation time
        # for the RTU-grid.
        method='scan_unrolled'
    )
    k_left = k_right - 1

    # jax's default out-of-bound index gives
    # correct result for point on the right most
    # edge of interpolation, no need to do anything
    # special for the boundary
    t = (x - ip_x_low_res[k_left]) / ip_delta_x[k_left]
    t2 = t**2
    t3 = t**3
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    term1 = ip_y_low_res[k_left] * h00
    term2 = ip_y_low_res[k_right] * h01
    term3 = (ip_dy_low_res[k_left]*h10 + ip_dy_low_res[k_right]*h11) * ip_delta_x[k_left]
    return term1 + term2 + term3


@register_pytree_node_class
class InvertPolySpline:
    @staticmethod
    def v_polyder(c):
        return jax.vmap(
            jnp.polyder,
            in_axes=1,
            out_axes=1
        )(c)

    @staticmethod
    def v_polyval(c, x):
        return jax.vmap(
            jnp.polyval,
            in_axes=(1, 1),
            out_axes=(1)
        )(c, x)

    def __init__(self, coefs, lower_bound, upper_bound, y_low_res):
        # coefs Nx2
        # lower_bound 1x2
        # upper_bound 1x2
        # low_res Mx2

        # polynomial to inverse
        self.coefs = coefs

        # get 1st derivative of polynomial
        self.dcoefs = InvertPolySpline.v_polyder(self.coefs)

        # The bounds of the CDF
        # below will always be 0
        # above will always be 1
        self.lower_bound = jax.lax.stop_gradient(jnp.atleast_2d(lower_bound))
        self.upper_bound = jax.lax.stop_gradient(jnp.atleast_2d(upper_bound))

        # low resolution grid of knots for spline approx to the inverse function
        # cubic spline needs the function, derivative, and delta_x at each node
        self.y_low_res = jax.lax.stop_gradient(y_low_res)
        self.x_low_res = InvertPolySpline.v_polyval(self.coefs, self.y_low_res)
        self.dy_low_res = 1 / InvertPolySpline.v_polyval(self.dcoefs, self.y_low_res)
        self.delta_x = jnp.diff(self.x_low_res, axis=0)

    def __repr__(self):
        return f'InvertPoly(coefs={self.coefs}, lower_bound={self.lower_bound}, upper_bound={self.upper_bound})'

    def tree_flatten(self):
        children = (
            self.coefs,
            self.dcoefs,
            self.y_low_res,
            self.x_low_res,
            self.dy_low_res,
            self.delta_x,
            self.lower_bound,
            self.upper_bound
        )
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, _, children):
        obj = object.__new__(InvertPolySpline)
        obj.coefs = children[0]
        obj.dcoefs = children[1]
        obj.y_low_res = children[2]
        obj.x_low_res = children[3]
        obj.dy_low_res = children[4]
        obj.delta_x = children[5]
        obj.lower_bound = children[6]
        obj.upper_bound = children[7]
        return obj

    def fwd_transform(self, x):
        y = jax.vmap(
            spline_invert,
            in_axes=(1, 1, 1, 1, 1),
            out_axes=(1)
        )(
            self.x_low_res,
            self.y_low_res,
            self.dy_low_res,
            self.delta_x,
            x
        )
        y = jnp.where(x <= self.lower_bound, 0.0, y)
        y = jnp.where(x >= self.upper_bound, 1.0, y)
        return jnp.clip(y, 0.0, 1.0)

    def rev_transform(self, y):
        return InvertPolySpline.v_polyval(self.coefs, y)


v_polyfit = jax.vmap(jnp.polyfit, in_axes=(1, 1, None, None, None, 1), out_axes=(1))


# For reference different polyfit methods were tested that used QR decomposition
# rather than the default SVD inside `jax.numpy.polyfit`, the tests did not show
# any significant speedup when running on a GPU.  Because of the weights the 
# matrix decomposition can't cached when the lensing mass changes, otherwise
# that would be one obvious speedup.


# def qr_polyfit(x, y, deg, weights):
#     order = deg + 1
#     lhs = jnp.vander(x, order)
#     lhs *= weights[:, jnp.newaxis]
#     rhs = y
#     rhs *= weights
#     scale = jnp.sqrt((lhs * lhs).sum(axis=0))
#     lhs /= scale[jnp.newaxis, :]
#     qr = jnp.linalg.qr(lhs)
#     return jax.scipy.linalg.solve_triangular(qr.R, jnp.dot(qr.Q.T, rhs)) / scale


# qr_v_polyfit = jax.jit(jax.vmap(qr_polyfit, in_axes=(1, 1, None, 1), out_axes=(1)), static_argnums=(2,))


# def lx_polyfit(x, y, deg, weights):
#     order = deg + 1
#     lhs = jnp.vander(x, order)
#     lhs *= weights[:, jnp.newaxis]
#     rhs = y
#     rhs *= weights
#     scale = jnp.sqrt((lhs * lhs).sum(axis=0))
#     lhs /= scale[jnp.newaxis, :]
#     # there is a "silent re-compile" triggered if `trow=True`, make sure it is `False`
#     sol = lx.linear_solve(lx.MatrixLinearOperator(lhs), rhs, solver=lx.QR(), throw=False)
#     return sol.value / scale


# lx_v_polyfit = jax.jit(jax.vmap(lx_polyfit, in_axes=(1, 1, None, 1), out_axes=(1)), static_argnums=(2,))

v_gradient = jax.vmap(jnp.gradient, in_axes=(1, 1), out_axes=1)


@partial(jax.jit, static_argnames=('deg'))
def create_transforms_spline(traced_points, deg=21, mesh_weight_map=None):
    N = traced_points.shape[0]  # // 2
    if mesh_weight_map is None:
        t = jnp.arange(1, N + 1) / (N + 1)
        t = jnp.stack([t, t], axis=1)
        sort_points = jnp.sort(traced_points, axis=0)  # [::2]
    else:
        sdx = jnp.argsort(traced_points, axis=0)
        sort_points = jnp.take_along_axis(traced_points, sdx, axis=0)
        t = jnp.stack([mesh_weight_map, mesh_weight_map], axis=1)
        t = jnp.take_along_axis(t, sdx, axis=0)
        t = jnp.cumsum(t, axis=0)

    # The CDF estimation needs to be a smooth function to avoid noise caused by
    # using a sub-set of traced points
    #
    # A polynomial is fit to the *inverse* CDF, this polynomial is inverted
    # numerically to get the smooth CDF function.
    #
    # The polynomial is fit to 'y' points at the Chebyshev nodes to avoid the
    # Runge phenomenon and to estimate the gradient of the CDF
    #
    # The gradient of the CDF is use as the weights for the polynomial fit
    # (e.g. where the CDF changes rapidly the weight is higher).  This
    # helps prevent overfitting for high degree polynomials and helps keep
    # log degree polynomials monotonic.

    # Use a number of Chebyshev nodes equal to the degree of the fit
    cheb_deg = deg + 1
    # calculate nodes and interpolated values at the nodes
    cheb_nodes = jax.lax.stop_gradient(
        ((jnp.cos((2 * jnp.arange(cheb_deg) + 1) * jnp.pi / (2 * cheb_deg))[::-1]) + 1) / 2
    )
    cy = jnp.stack([cheb_nodes, cheb_nodes], axis=1)
    cx = jax.vmap(interp1d, in_axes=(None, 1, 1), out_axes=1)(cheb_nodes, t, sort_points)

    # fit the polynomial with weights
    w = v_gradient(cy, cx)
    coefs = v_polyfit(cy, cx, deg, None, False, w)
    # coefs = lx_v_polyfit(cy, cx, deg, w)
    # coefs = qr_v_polyfit(cy, cx, deg, w)

    # invert the polynomial with custom class
    # use the Chebyshev nodes (with the 0 and 1 appended as the start and end points)
    # as the knots of a cubic spline
    knots = jnp.vstack([jnp.zeros(2), cy, jnp.ones(2)])
    inv_poly = InvertPolySpline(coefs, sort_points[0], sort_points[-1], knots)
    # return sort_points and t for debugging, in production we just need the transforms
    return inv_poly, sort_points, t


@register_pytree_node_class
class InvertInterp:
    @staticmethod
    def v_interp1d(xp, yp, x):
        return jax.vmap(
            interp1d,
            in_axes=(1, 1, 1),
            out_axes=(1)
        )(x, xp, yp)

    def __init__(self, sort_points, t):
        self.sort_points = sort_points
        self.t = t

    def __repr__(self):
        return f'InvertPoly(sort_points={self.sort_points}, t={self.t})'

    def tree_flatten(self):
        children = (
            self.sort_points,
            self.t
        )
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*(children + aux_data))

    def fwd_transform(self, x):
        y = InvertInterp.v_interp1d(
            self.sort_points,
            self.t,
            x
        )
        y = jnp.where(x <= self.sort_points[0], 0.0, y)
        y = jnp.where(x >= self.sort_points[-1], 1.0, y)
        return jnp.clip(y, 0.0, 1.0)

    def rev_transform(self, y):
        return InvertInterp.v_interp1d(
            self.t,
            self.sort_points,
            y
        )


def create_transforms_linear_interp(traced_points, mesh_weight_map=None, **_):
    if mesh_weight_map is None:
        sort_points = jnp.sort(traced_points, axis=0)[::2]
        N = sort_points.shape[0]
        t = jnp.arange(1, N + 1) / (N + 1)
        t = jnp.stack([t, t], axis=1)
    else:
        sdx = jnp.argsort(traced_points, axis=0)[::2]
        sort_points = jnp.take_along_axis(traced_points, sdx, axis=0)
        t = jnp.stack([mesh_weight_map, mesh_weight_map], axis=1)
        t = jnp.take_along_axis(t, sdx, axis=0)
        t = jnp.cumsum(t, axis=0)

    Ii = InvertInterp(sort_points, t)
    return Ii, sort_points, t
