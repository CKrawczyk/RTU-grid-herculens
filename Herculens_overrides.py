import jax
import jax.numpy as jnp
import numpy as np

from herculens import PixelatedLight, LightModel, LensImage, PixelGrid
from herculens.LightModel.light_model import function_static_single
from RTUGrid import create_transforms_linear_interp, create_transforms_spline
from scipy.fft import next_fast_len
from jax.tree_util import register_pytree_node_class
from functools import partial


class LightModelRTU(LightModel):
    @property
    def pixel_is_rtu_grid(self):
        if not self.has_pixels:
            return False
        return self.func_list[self.pixelated_index].is_rtu_grid

    @partial(jax.jit, static_argnums=(0, 3))
    def pixel_rtu_uniform_transform(self, x_mask, y_mask, weights_mask):
        if self.pixel_is_rtu_grid:
            return self.func_list[self.pixelated_index].rtu_uniform_transform(
                x_mask, y_mask, weights_mask
            )
    
    def surface_brightness(
            self, x, y, kwargs, k=None,
            pixels_x_coord=None, pixels_y_coord=None,
            transform_params=None,
            return_as_list=False
        ):
            if isinstance(k, int):
                return self._surf_bright_single(
                    x, y, kwargs, k=k,
                    pixels_x_coord=pixels_x_coord,
                    pixels_y_coord=pixels_y_coord,
                    transform_params=transform_params,
                    return_as_list=return_as_list
                )
            elif self._single_profile_mode:
                return self._surf_bright_single(
                    x, y, kwargs, k=0,
                    pixels_x_coord=pixels_x_coord,
                    pixels_y_coord=pixels_y_coord,
                    transform_params=transform_params,
                    return_as_list=return_as_list
                )
            elif self._repeated_profile_mode:
                return self._surf_bright_repeated(
                    x, y, kwargs, k=k,
                    pixels_x_coord=pixels_x_coord,
                    pixels_y_coord=pixels_y_coord,
                    transform_params=transform_params,
                    return_as_list=return_as_list
                )
            else:
                return self._surf_bright_loop(x, y, kwargs, k=k,
                    pixels_x_coord=pixels_x_coord,
                    pixels_y_coord=pixels_y_coord,
                    transform_params=transform_params,
                    return_as_list=return_as_list
                )
            
    def _surf_bright_single(
            self, x, y, kwargs, k=None,
            pixels_x_coord=None, pixels_y_coord=None,
            transform_params=None,
            return_as_list=False
        ):
        if k == self.pixelated_index:
            flux = self.func_list[k].function(
                x, y, **kwargs[k],
                pixels_x_coord=pixels_x_coord, 
                pixels_y_coord=pixels_y_coord,
                transform_params=transform_params,
            )
        else:
            flux = self.func_list[k].function(x, y, **kwargs[k])
        if return_as_list:
            return [flux]
        return flux
        
    def _surf_bright_repeated(
            self, x, y, kwargs, k=None,
            pixels_x_coord=None, pixels_y_coord=None,
            transform_params=None,
            return_as_list=False
        ):
        if k is not None:
            raise NotImplementedError("Repeated profile mode not implemented "
                                      "specific profile k.")
        func = function_static_single(x, y, self.func_list[0].function)
        flux_list = [
            func(**kwargs[i]) for i in range(self._num_func)
        ]
        if return_as_list:
            return flux_list
        return jnp.sum(jnp.array(flux_list),axis=0)

    def _surf_bright_loop(
            self, x, y, kwargs_list, k=None,
            pixels_x_coord=None, pixels_y_coord=None,
            transform_params=None,
            return_as_list=False
        ):
        if return_as_list:
            flux = []
        else:
            flux = jnp.zeros_like(x)
        bool_list = self._bool_list(k)
        for i, func in enumerate(self.func_list):
            if bool_list[i]:
                if i == self.pixelated_index:
                    flux_i = func.function(
                        x, y, 
                        pixels_x_coord=pixels_x_coord, 
                        pixels_y_coord=pixels_y_coord,
                        transform_params=transform_params,
                        **kwargs_list[i]
                    )
                else:
                    flux_i = func.function(x, y, **kwargs_list[i])
                if return_as_list:
                    flux.append(flux_i)
                else:
                    flux += flux_i
        return flux


class PixelatedLightRTU(PixelatedLight):
    def __init__(
        self,
        *args,
        rtu_grid=False,
        rtu_grid_type='spline',
        rtu_poly_order=11,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._rtu_grid = rtu_grid
        self._rtu_poly_order = rtu_poly_order
        if rtu_grid_type == 'spline':
            self._rtu_grid_class = create_transforms_spline
        elif rtu_grid_type == 'linear':
            self._rtu_grid_class = create_transforms_linear_interp
        else:
            raise ValueError('rtu_grid_type should be either "spline" or "linear".')
    
    @property
    def is_rtu_grid(self):
        return self._rtu_grid

    def rtu_uniform_transform(self, x_mask, y_mask, weights_mask):
        grid_src_mask = jnp.stack([x_mask, y_mask], axis=1)
        mu = grid_src_mask.mean(axis=0)
        scale = grid_src_mask.std(axis=0).min()
        grid_src_mask_standard = (grid_src_mask - mu) / scale
        transform, _, _ = self._rtu_grid_class(
            grid_src_mask_standard,
            deg=self._rtu_poly_order,
            mesh_weight_map=weights_mask
        )

        # transform bin centers back to source plane coords
        grid_coords_std = transform.rev_transform(self._uniform_pixel_centers_stack)
        grid_coords = grid_coords_std * scale + mu
        # return both the standardized coords (mass-sheet-like transform invariant)
        # and observed space coords for the source bin centers
        grid_coords_out = (grid_coords_std, grid_coords)
        # transform bin edges back to source plane coords
        grid_edges_std = transform.rev_transform(self._uniform_pixel_edges_stack)
        grid_edges = grid_edges_std * scale + mu
        # return both the standardized coords (mass sheet transform invariant)
        # and observed space coords for the source bin edges
        grid_edges_out = (grid_edges_std, grid_edges)
        transform_params = (transform, mu, scale)
        return transform_params, grid_coords_out, grid_edges_out

    def function(
        self,
        x,
        y,
        pixels_x_coord=None,
        pixels_y_coord=None,
        pixels=None,
        transform_params=None
    ):
        if self._rtu_grid:
            transform, mu, sigma = transform_params
            pos = (jnp.stack([x, y], axis=1) - mu) / sigma
            x, y = transform.fwd_transform(pos).T
            # these are the "out of bounds" pixels that should be set to 0
            out_of_x = jnp.bitwise_or(x == 0, x == 1)
            out_of_y = jnp.bitwise_or(y == 0, y == 1)
        if self._interp_type == 'fast_bilinear':
            f = self._function_fast(x, y, pixels_x_coord, pixels_y_coord, pixels)
        elif self._interp_type in ['bilinear', 'bicubic']:
            f = self._function_std(x, y, pixels_x_coord, pixels_y_coord, pixels)
        if self._rtu_grid:
            # correctly set "out of bounds" pixels to 0
            f = jnp.where(
                jnp.bitwise_or(out_of_x, out_of_y),
                0.0,
                f
            )
        # normalize for correct units when evaluated by LensImage methods
        return f / self._data_pixel_area
    
    def set_pixel_grid(self, pixel_grid, data_pixel_area):
        super().set_pixel_grid(pixel_grid, data_pixel_area)

        # calculate and store the pixel centers and edges in original coords
        self._uniform_pixel_centers_stack = jnp.stack(self.pixel_grid.pixel_axes).T
        half_pixel_width = 0.5 * self.pixel_grid.pixel_width
        self._uniform_pixel_edges_stack = jnp.r_[
            self._uniform_pixel_centers_stack - half_pixel_width,
            self._uniform_pixel_centers_stack[-1:] + half_pixel_width
        ]

    def derivatives(self, x, y, pixels_x_coord=None, pixels_y_coord=None, pixels=None, transform_params=None):
        if self._deriv_type == 'interpol':
            f_x, f_y = self._derivatives_interpol(x, y, pixels_x_coord, pixels_y_coord, pixels, transform_params)
        elif self._deriv_type == 'autodiff':
            f_x, f_y = self._derivatives_autodiff(x, y, pixels_x_coord, pixels_y_coord, pixels, transform_params)
        # normalize for correct units when evaluated by LensImage methods
        return f_x / self._data_pixel_area, f_y / self._data_pixel_area


    def _derivatives_autodiff(self, x, y, pixels_x_coord, pixels_y_coord, pixels, transform_params):
        def function(params):
            res = self.function(
                params[0], params[1], 
                pixels_x_coord=pixels_x_coord, pixels_y_coord=pixels_y_coord,
                pixels=pixels, transform_params=transform_params)
            if self._interp_type != 'fast_bilinear':
                res = res[0]
            return res
        grad_func = jax.grad(function)
        param_array = jnp.array([x.flatten(), y.flatten()]).T
        res = jax.vmap(grad_func)(param_array)
        f_x = res[:, 0].reshape(*x.shape)
        f_y = res[:, 1].reshape(*x.shape)
        return f_x, f_y


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
    def __init__(self, *args, rtu_mesh_weights=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._src_rtu_grid = self.SourceModel.pixel_is_rtu_grid

        if self._src_rtu_grid is True and self.source_arc_mask is None:
            raise ValueError("An arc mask for the lensed source must be "
                             "provided with RTU-grid source")
        if self._src_rtu_grid:
            # flatten the mask for use with the RTU-grid transform
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
            n_pix = self.SourceModel.pixel_grid_settings['num_pixels']
            # provide a small 1e-5 buffer on all sides for points outside the mask to be traced onto
            pixel_width = (1 - 2e-5) / n_pix
            zero_point = 0.5 * pixel_width + 1e-5
            pixel_grid = PixelGrid(
                n_pix,
                n_pix,
                pixel_width * np.eye(2),
                ra_at_xy_0=zero_point,
                dec_at_xy_0=zero_point
            )
            self.SourceModel.set_pixel_grid(pixel_grid, self.Grid.pixel_area)

    def adapt_rtu_transform(self, kwargs_lens):
        x_grid_img, y_grid_img = self.Grid.pixel_coordinates
        x_grid_src, y_grid_src = self.MassModel.ray_shooting(
            x_grid_img.ravel()[self.source_arc_mask_flat],
            y_grid_img.ravel()[self.source_arc_mask_flat],
            kwargs_lens
        )
        return self.SourceModel.pixel_rtu_uniform_transform(
            x_grid_src, y_grid_src, self.rtu_mesh_weights_mask
        )

    def eval_source_surface_brightness(
            self, x, y, kwargs_source, kwargs_lens=None, 
            k=None, k_lens=None, de_lensed=False,
            adapted_pixels_coords=None, 
            return_pixels_coords=False,
            return_as_list=False
        ):
        transform_params = None
        pixels_x_coord = None
        pixels_y_coord = None
        if self._src_rtu_grid:
            transform_params, grid_coords, grid_edges = self.adapt_rtu_transform(kwargs_lens)
        elif self._src_adaptive_grid:
            if adapted_pixels_coords is None:
                pixels_x_coord, pixels_y_coord, _ = self.adapt_source_coordinates(kwargs_lens)
            else:
                pixels_x_coord, pixels_y_coord = adapted_pixels_coords

        if de_lensed is True:
            source_light = self.SourceModel.surface_brightness(
                x, y, kwargs_source, k=k,
                pixels_x_coord=pixels_x_coord, pixels_y_coord=pixels_y_coord,
                transform_params=transform_params,
                return_as_list=return_as_list
            )
        else:
            x_grid_src, y_grid_src = self.MassModel.ray_shooting(x, y, kwargs_lens, k=k_lens)
            source_light = self.SourceModel.surface_brightness(
                x_grid_src, y_grid_src, kwargs_source, k=k,
                pixels_x_coord=pixels_x_coord, pixels_y_coord=pixels_y_coord,
                transform_params=transform_params
            )
        if return_pixels_coords:
            if self._src_rtu_grid:
                coords = (grid_coords, grid_edges)
            else:
                coords = (pixels_x_coord, pixels_y_coord)
            return source_light, coords
        return source_light


class LensImageRTUGridLowMem(LensImageRTUGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # For memory efficiency supersampling is calculated one "observed grid" at a time
        # and the average is tracked as a running sum.  Define the "center offsets" for
        # each supersampling location so it can be looped over later on.
        self.deltas = self.get_deltas()
        self.pixel_area = self.Grid.pixel_width**2
        self.centers = np.array(self.Grid.pixel_coordinates).reshape(2, -1)

    def get_deltas(self):
        supersampling_factor = self.ImageNumerics.grid_supersampling_factor
        pixel_width = self.Grid.pixel_width
        N = 2 * supersampling_factor + 1
        sub_centers = jnp.linspace(-0.5, 0.5, N)[1:-1:2]
        sub_grid = jnp.array(jnp.meshgrid(sub_centers, sub_centers)).T.reshape(supersampling_factor**2, 2, 1)
        return sub_grid * pixel_width

    def source_surface_brightness(
            self, kwargs_source, kwargs_lens,
            return_pixels_coords=False
    ):
        transform_params = None
        pixels_x_coord = None
        pixels_y_coord = None
        if self._src_rtu_grid:
            transform_params, grid_coords, grid_edges = self.adapt_rtu_transform(kwargs_lens)
        elif self._src_adaptive_grid:
            pixels_x_coord, pixels_y_coord, _ = self.adapt_source_coordinates(kwargs_lens)

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
            new_value = self.SourceModel.surface_brightness(
                x_grid_src, y_grid_src, kwargs_source,
                pixels_x_coord=pixels_x_coord, pixels_y_coord=pixels_y_coord,
                transform_params=transform_params
            )
            # track a running mean
            new_value = new_value * self.pixel_area
            count += 1
            delta_value = new_value - mean
            # new_mean = mean + delta / new_count
            mean += delta_value / count
            return (count, mean), None

        # loop over each set of sub-pixels in the supersampled grid and track a running mean
        init = (0, jnp.zeros(self.Grid.num_pixel))
        (_, source_light) = jax.lax.scan(body, init, self.deltas)[0]

        if return_pixels_coords:
            if self._src_rtu_grid:
                coords = (grid_coords, grid_edges)
            else:
                coords = (pixels_x_coord, pixels_y_coord)
            return source_light, coords
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

