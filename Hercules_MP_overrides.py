import jax
import jax.numpy as jnp
import numpy as np

from herculens import MPLightModel, MPLensImage, PixelGrid
from functools import partial


class MPLightModelRTU(MPLightModel):
    @property
    def pixel_is_rtu_grid(self):
        return [light_model.pixel_is_rtu_grid if light_model is not None else False for light_model in self.light_models]

    def pixel_rtu_uniform_transform(self, x_plane, y_plane, mask_plane, weights_plane):
        output = []
        for jdx, plane in enumerate(self.light_models):
            o = None
            if (plane is not None):
                if (plane.pixel_is_rtu_grid):
                    w = None
                    if weights_plane[jdx] is not None:
                        w = weights_plane[jdx][mask_plane[jdx]]
                    o = plane.pixel_rtu_uniform_transform(
                        x_plane[jdx][mask_plane[jdx]],
                        y_plane[jdx][mask_plane[jdx]],
                        w
                    )
            output.append(o)
        return output

    @partial(jax.jit, static_argnums=(0, 7))
    def surface_brightness(
        self,
        x,
        y,
        kwargs_list,
        pixels_x_coord,
        pixels_y_coord,
        transform_params=None,
        k=None,
    ):
        k = self.k_expand(k)
        flux = []
        for j in range(self.number_light_planes):
            if self.has_light[j]:
                flux.append(
                    self.light_models[j].surface_brightness(
                        x[j], y[j], kwargs_list[j],
                        k=k[j],
                        pixels_x_coord=pixels_x_coord[j],
                        pixels_y_coord=pixels_y_coord[j],
                        transform_params=transform_params[j]
                    )
                )
            else:
                flux.append(jnp.zeros_like(x[j]))
        return jnp.stack(flux)


class MPLensImageRTUGrid(MPLensImage):
    def __init__(self, *args, rtu_mesh_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.centers = np.array(self.Grid.pixel_coordinates).reshape(2, -1)
        self._src_rtu_grid = self.MPLightModel.pixel_is_rtu_grid
        self.rtu_mesh_weights_mask = [None] * self.MPLightModel.number_light_planes
        self.source_arc_masks_old = self.source_arc_masks
        for i, has_rtu in enumerate(self._src_rtu_grid):
            if has_rtu and (rtu_mesh_weights is not None):
                w = np.where(self.source_arc_masks_old[i], rtu_mesh_weights[i], 0.0).ravel()
                self.rtu_mesh_weights_mask[i] = w / w.sum()
                # remove the original mask and replace with one based on the weights value
                self.source_arc_masks[i] = (w > 0).reshape(self.Grid.num_pixel_axes)
            if has_rtu:
                n_pix = self.MPLightModel.light_models[i].pixel_grid_settings['num_pixels']
                pixel_width = (1 - 2e-5) / n_pix
                zero_point = 0.5 * pixel_width + 1e-5
                pixel_grid = PixelGrid(
                    n_pix,
                    n_pix,
                    pixel_width * np.eye(2),
                    ra_at_xy_0=zero_point,
                    dec_at_xy_0=zero_point
                )
                self.MPLightModel.light_models[i].set_pixel_grid(pixel_grid, self.Grid.pixel_area)
        ssf = self.ImageNumerics.grid_supersampling_factor

        self.source_arc_masks_flat = self.source_arc_masks.reshape(
            self.MPLightModel.number_light_planes,
            -1
        ).astype(bool)
        # get masks in super sampled space
        s_ones = np.ones([ssf, ssf])
        self.source_arc_masks_ss = np.stack([
            np.kron(m, s_ones) for m in self.source_arc_masks
        ])
        # flatten the super sampled masks
        self._source_arc_masks_flat = self.source_arc_masks_ss.reshape(
            self.MPLightModel.number_light_planes,
            -1
        )
        self.rtu_mesh_weights_mask = np.array(self.rtu_mesh_weights_mask)
        self.centers = np.array(self.Grid.pixel_coordinates).reshape(2, -1)

    @partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    def model(
        self,
        eta_flat=None,
        kwargs_mass=None,
        kwargs_light=None,
        supersampled=False,
        unconvolved=False,
        k_mass=None,
        k_light=None,
        k_planes=None,
        apply_mask=True,
        return_pixel_scale=False,
        return_pixels_coords=False,
        point_source_add=False,
        kwargs_point_source=None,
    ):
        ra_grid_img, dec_grid_img = self.ImageNumerics.coordinates_evaluate
        transform_params = [None] * self.MPLightModel.number_light_planes
        if any(self._src_rtu_grid):
            ra_centers_planes, dec_centers_planes = self.MPMassModel.ray_shooting(
                self.centers[0],
                self.centers[1],
                eta_flat,
                kwargs_mass
            )
            transform_outputs = self.MPLightModel.pixel_rtu_uniform_transform(
                ra_centers_planes,
                dec_centers_planes,
                self.source_arc_masks_flat,
                self.rtu_mesh_weights_mask
            )
            transform_params = [t[0] if t is not None else None for t in transform_outputs]
            grid_coords = [t[1] if t is not None else None for t in transform_outputs]
            grid_edges = [t[2] if t is not None else None for t in transform_outputs]

        # pixel grid positions on each mass plane (including the lens plane)
        ra_grid_planes, dec_grid_planes = self.MPMassModel.ray_shooting(
            ra_grid_img,
            dec_grid_img,
            eta_flat,
            kwargs_mass,
            k=k_mass
        )
        # (masked) light contribution from each plane
        pixels_x_coord, pixels_y_coord, _ = self.adapt_source_coordinates(
            ra_grid_planes,
            dec_grid_planes
        )
        light_planes = self.MPLightModel.surface_brightness(
            ra_grid_planes,
            dec_grid_planes,
            kwargs_light,
            pixels_x_coord,
            pixels_y_coord,
            k=k_light,
            transform_params=transform_params
        )
        if apply_mask:
            light_planes = light_planes * self._source_arc_masks_flat
        k_planes = self.k_extend(k_planes, len(light_planes))
        model = light_planes[k_planes].sum(axis=0)
        if not supersampled:
            model = self.ImageNumerics.re_size_convolve(model, unconvolved=unconvolved)
            if point_source_add:
                model = model + self.point_source_image(
                    kwargs_point_source, eta_flat, kwargs_mass
                )
        if return_pixels_coords:
            if any(self._src_rtu_grid):
                return model, (grid_coords, grid_edges)
            else:
                return model, None
        if return_pixel_scale:
            pixel_scale = [x[1] - x[0] if x is not None else None for x in pixels_x_coord]
            return model, pixel_scale
        else:
            return model


class MPLensImageRTUGridLowMem(MPLensImageRTUGrid):
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

    @partial(jax.jit, static_argnums=(0, 5, 6, 7))
    def model(
        self,
        PSF_class,
        eta_flat=None,
        kwargs_mass=None,
        kwargs_light=None,
        unconvolved=False,
        apply_mask=True,
        return_pixels_coords=False
    ):
        transform_params = [None] * self.MPLightModel.number_light_planes
        ra_centers_planes, dec_centers_planes = self.MPMassModel.ray_shooting(
            self.centers[0],
            self.centers[1],
            eta_flat,
            kwargs_mass
        )
        # all sub-pixels need to resolve to the same grid if adaptive
        # calculated in once up front on the trance of the pixel centers
        pixels_x_coord, pixels_y_coord, _ = self.adapt_source_coordinates(
            ra_centers_planes,
            dec_centers_planes
        )
        if any(self._src_rtu_grid):
            transform_outputs = self.MPLightModel.pixel_rtu_uniform_transform(
                ra_centers_planes,
                dec_centers_planes,
                self.source_arc_masks_flat,
                self.rtu_mesh_weights_mask
            )
            transform_params = [t[0] if t is not None else None for t in transform_outputs]
            grid_coords = [t[1] if t is not None else None for t in transform_outputs]
            grid_edges = [t[2] if t is not None else None for t in transform_outputs]


        # use jax.checkpoint to keep memory usage low when taking reverse mode jacobian
        @jax.checkpoint
        def body(carry, delta):
            count, mean = carry
            # get the new source light for current sub pixels
            centers_shift = self.centers + delta
            ra_grid_planes, dec_grid_planes = self.MPMassModel.ray_shooting(
                centers_shift[0],
                centers_shift[1],
                eta_flat,
                kwargs_mass,
            )
            new_value = self.MPLightModel.surface_brightness(
                ra_grid_planes,
                dec_grid_planes,
                kwargs_light,
                pixels_x_coord,
                pixels_y_coord,
                transform_params=transform_params
            )
            # apply mask if needed before summing
            if apply_mask:
                new_value = new_value * self.source_arc_masks_flat
            # track a running mean on the sum of the planes
            new_value = new_value.sum(axis=0) * self.pixel_area
            count += 1
            delta_value = new_value - mean
            # new_mean = mean + delta / new_count
            mean += delta_value / count
            return (count, mean.squeeze()), None

        init = (0, jnp.zeros(self.Grid.num_pixel))
        (_, model) = jax.lax.scan(body, init, self.deltas)[0]
        model = model.reshape(self.Grid.num_pixel_axes)
        if return_pixels_coords:
            if any(self._src_rtu_grid):
                return model, (grid_coords, grid_edges)
            else:
                return model, None
        if not unconvolved:
            model = PSF_class.convolution2d(model)
        return model
