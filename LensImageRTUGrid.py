import jax
import jax.numpy as jnp
import numpy as np
# import lineax as lx

from functools import partial
from herculens import LensImage, PixelGrid
from scipy.fft import next_fast_len
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


@register_pytree_node_class
class SimpleFFTConvolve():
    '''A simplified PSF class that provides an FFT based convolution as a jax pytree.
    This class only calculates the FFT of the kernel once and stores it rather than
    recalculated it each call.  As a pytree an instance of this class can be passed
    into `jax.jit`ed functions without issue.
    '''
    def __init__(self, kernel, output_shape):
        self.image_shape = output_shape
        self.kernel = kernel
        full_shape = tuple(s1 + s2 - 1 for s1, s2 in zip(self.image_shape, self.kernel.shape))
        self.fft_shape = tuple(next_fast_len(s) for s in full_shape)
        self.sp2 = jnp.fft.rfftn(self.kernel, self.fft_shape)
        self.start_indices = tuple(
            (full_size - out_size) // 2
            for full_size, out_size in zip(full_shape, self.image_shape)
        )

    def __repr__(self):
        return f'SimpleFFTConvolve(kernel={self.kernel}, image_shape={self.image_shape})'

    def tree_flatten(self):
        children = (            
            self.kernel,
            self.sp2
        )
        aux_data = (
            self.image_shape,
            self.fft_shape,
            self.start_indices
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(SimpleFFTConvolve)
        obj.image_shape = aux_data[0]
        obj.fft_shape = aux_data[1]
        obj.start_indices = aux_data[2]
        obj.kernel = children[0]
        obj.sp2 = children[1]
        return obj

    def convolution2d(self, image):
        sp1 = jnp.fft.rfftn(image, self.fft_shape)
        sp_conv = sp1 * self.sp2
        conv = jnp.fft.irfftn(sp_conv, self.fft_shape)
        return jax.lax.dynamic_slice(conv, self.start_indices, image.shape)


class LensImageRTUGrid(LensImage):
    def __init__(
            self,
            *args,
            rtu_grid_source=False,
            rtu_grid_source_size=50,
            rtu_grid_type='spline',
            rtu_poly_order=11,
            rtu_mesh_weights=None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self._rtu_grid_source = rtu_grid_source
        if self._rtu_grid_source is True and self.source_arc_mask is None:
            raise ValueError("An arc mask for the lensed source must be "
                             "provided with RTU-grid source")
        if self._rtu_grid_source is True and self.SourceModel.has_pixels is False:
            raise ValueError("A pixelated source must per provided with a RTU-grid source")
        if self._rtu_grid_source:
            self.rtu_grid_source_size = rtu_grid_source_size
            # flatten the mask for use with the RTU-grid transfrom
            self.source_arc_mask_flat = self.source_arc_mask.astype(bool).ravel()
            # mask the RTU mesh weights
            if rtu_mesh_weights is not None:
                self.rtu_mesh_weights_mask = rtu_mesh_weights[self.source_arc_mask_flat]
                # normalize mesh weights
                self.rtu_mesh_weights_mask = self.rtu_mesh_weights_mask / self.rtu_mesh_weights_mask.sum()
            else:
                self.rtu_mesh_weights_mask = None
            # move the original mask to a new name so it is not applied during the lens modeling
            self.source_arc_mask_old = self.source_arc_mask
            self.source_arc_mask = None
            # set pixel grid to be fixed from 0-1 with correct number of pixels
            n_pix = self.rtu_grid_source_size
            # provide a small 1e-5 buffer on all sides for points outside the mask to be traced onto
            pixel_width = (1 - 2e-5) / n_pix
            zero_point = 0.5 * pixel_width + 1e-5
            self.uniform_edges = jnp.linspace(1e-5, 1 - 1e-5, n_pix + 1)
            # uniform space pixel edges
            self.uniform_edges_stack = jnp.stack([self.uniform_edges, self.uniform_edges]).T
            pixel_grid = PixelGrid(
                n_pix,
                n_pix,
                pixel_width * np.eye(2),
                ra_at_xy_0=zero_point,
                dec_at_xy_0=zero_point
            )
            self.SourceModel.set_pixel_grid(pixel_grid, self.Grid.pixel_area)
            # uniform space pixel centers
            self.uniform_centers_stack = jnp.stack(self.SourceModel.pixel_grid.pixel_axes).T
            # order of the polynomial to fit to the inverse eCDF, larger values can lead to slower
            # modeling times, only used for `spline` inversion
            self.rtu_poly_order = rtu_poly_order
            # use either spline inversion or linear interpolation inversion
            if rtu_grid_type == 'spline':
                self.rtu_grid_class = create_transforms_spline
            elif rtu_grid_type == 'linear':
                self.rtu_grid_class = create_transforms_linear_interp
            else:
                raise ValueError('rtu_grid_type should be either "spline" or "linear".')

        # For memory efficiency supersampling is calculated one "observed grid" at a time
        # and the average is tracked as a running sum.  Define the "center offsets" for
        # each supersampling location so it can be looped over later on.
        self.deltas = self.get_deltas()
        self.centers = np.array(self.Grid.pixel_coordinates).reshape(2, -1)
        self.pixel_area = self.Grid.pixel_width**2

    def get_deltas(self):
        supersampling_factor = self.ImageNumerics.grid_supersampling_factor
        pixel_width = self.Grid.pixel_width
        N = 2 * supersampling_factor + 1
        sub_centers = jnp.linspace(-0.5, 0.5, N)[1:-1:2]
        sub_grid = jnp.array(jnp.meshgrid(sub_centers, sub_centers)).T.reshape(supersampling_factor**2, 2, 1)
        return sub_grid * pixel_width

    @partial(jax.jit, static_argnums=(0,))
    def uniform_transform(self, x_grid_src_mask, y_grid_src_mask):
        grid_src_mask = jnp.stack([x_grid_src_mask, y_grid_src_mask], axis=1)
        # standardize mask points
        # this removes mass-sheet-like transforms
        # e.g. all models that are the same up to a sift and uniform scaling of the
        # source will resolve to the same RTU-grid
        mu = grid_src_mask.mean(axis=0)
        scale = grid_src_mask.std(axis=0).min()
        grid_src_mask_standard = (grid_src_mask - mu) / scale
        transform, _, _ = self.rtu_grid_class(grid_src_mask_standard, deg=self.rtu_poly_order, mesh_weight_map=self.rtu_mesh_weights_mask)
        # transform bin centers back to source plane coords
        grid_coords_std = transform.rev_transform(self.uniform_centers_stack)
        grid_coords = grid_coords_std * scale + mu
        # return both the standardized coords (mass-sheet-like transform invariant)
        # and observed space coords for the source bin centers
        grid_coords_out = (grid_coords_std, grid_coords)
        # transform bin edges back to source plane coords
        grid_edges_std = transform.rev_transform(self.uniform_edges_stack)
        grid_edges = grid_edges_std * scale + mu
        # return both the standardized coords (mass sheet transform invariant)
        # and observed space coords for the source bin edges
        grid_edges_out = (grid_edges_std, grid_edges)
        return (transform, mu, scale), grid_coords_out, grid_edges_out

    def source_surface_brightness(
            self, kwargs_source, kwargs_lens,
            return_pixels_coords=False
    ):
        transform_params = None
        pixels_x_coord = None
        pixels_y_coord = None
        pixels_x_coord_in = None
        pixels_y_coord_in = None
        if self._rtu_grid_source:
            x_grid, y_grid = self.MassModel.ray_shooting(
                self.centers[0][self.source_arc_mask_flat],
                self.centers[1][self.source_arc_mask_flat],
                kwargs_lens
            )
            transform_params, grid_coords, grid_edges = self.uniform_transform(x_grid, y_grid)
            pixels_x_coord = grid_coords
            pixels_y_coord = grid_edges
        elif self._src_adaptive_grid:
            pixels_x_coord_in, pixels_y_coord_in, _ = self.adapt_source_coordinates(kwargs_lens)
            pixels_x_coord = pixels_x_coord_in
            pixels_y_coord = pixels_y_coord_in

        # use jax.checkpoint to keep memory usage low when taking reverse mode jacobian
        @jax.checkpoint
        def body(carry, delta):
            count, mean = carry
            # get the new source light for current sub pixels
            centers_shift = self.centers + delta
            x_grid_src, y_grid_src = self.MassModel.ray_shooting(
                centers_shift[0],
                centers_shift[1],
                kwargs_lens
            )
            if self._rtu_grid_source:
                transform, mu, sigma = transform_params
                pos = (jnp.stack([x_grid_src, y_grid_src], axis=1) - mu) / sigma
                x_grid_src, y_grid_src = transform.fwd_transform(pos).T
                out_of_x = jnp.bitwise_or(x_grid_src == 0, x_grid_src == 1)
                out_of_y = jnp.bitwise_or(y_grid_src == 0, y_grid_src == 1)
            new_value = self.SourceModel.surface_brightness(
                x_grid_src, y_grid_src, kwargs_source,
                pixels_x_coord=pixels_x_coord_in, pixels_y_coord=pixels_y_coord_in
            )
            if self._rtu_grid_source:
                # most of the interpolators don't fully mask out the "out of bounds" pixels
                # because the are so close to the bounds
                # this ensures the masking is done correctly
                new_value = jnp.where(
                    jnp.bitwise_or(out_of_x, out_of_y),
                    0.0,
                    new_value
                )
            # track a running mean
            new_value = new_value * self.pixel_area
            count += 1
            delta = new_value - mean
            # new_mean = mean + delta / new_count
            mean += delta / count
            return (count, mean), None

        # loop over each set of sub-pixels in the supersampled grid and track a running mean
        init = (0, jnp.zeros(self.Grid.num_pixel))
        (_, source_light) = jax.lax.scan(body, init, self.deltas)[0]

        if return_pixels_coords:
            return source_light, (pixels_x_coord, pixels_y_coord)
        return source_light

    def lens_surface_brightness(self, kwargs_lens_light):
        # use jax.checkpoint to keep memory usage low when taking reverse mode jacobian
        @jax.checkpoint
        def body(carry, delta):
            count, mean = carry
            # get the new lens light for current sub pixels
            centers_shift = self.centers + delta
            new_value = self.LensLightModel.surface_brightness(
                centers_shift[0],
                centers_shift[1],
                kwargs_lens_light
            )
            new_value = new_value * self.pixel_area
            # track running mean
            count += 1
            delta = new_value - mean
            # new_mean = mean + delta / new_count
            mean += delta / count
            return (count, mean), None

        # loop over each set of sub-pixels in the supersampled grid and track a running mean
        init = (0, jnp.zeros(self.Grid.num_pixel))
        (_, lens_light) = jax.lax.scan(body, init, self.deltas)[0]
        return lens_light

    @partial(jax.jit, static_argnums=(0, 5, 6))
    def model(
        self, PSF_class, kwargs_lens=None, kwargs_source=None, kwargs_lens_light=None,
        unconvolved=False, return_source_pixels_coords=False
    ):
        # PSF_class is passed in as a variable so it can be fit along side
        # the other parameters without triggering a re-compile
        model, adapted_source_pixels_coords = self.source_surface_brightness(
            kwargs_source, kwargs_lens,
            return_pixels_coords=True
        )
        model += self.lens_surface_brightness(
            kwargs_lens_light
        )
        model = model.reshape(self.Grid.num_pixel_axes)
        if not unconvolved:
            # The memory efferent supersampling constructs the "resized" model
            # already, so need to de-couple the convolution from "resize_and_convolve"
            model = PSF_class.convolution2d(model)
        if return_source_pixels_coords:
            return model, adapted_source_pixels_coords
        return model
