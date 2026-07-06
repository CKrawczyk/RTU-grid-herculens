# RTU-grid-herculens

A herculens implementation of the RTU-grid presented in Wolfgang et al. 2026

## Requirements

This code was developed and tested with the following versions of Herculens and Jax:

- Herculens >= v0.2.2
- Jax >= v0.4.38 and <= v0.9.2 (Herculens does not support higher versions of Jax at the moment)

Importantly for the `PixelatedLight` class `interpolation_type='fast_bilinear'` should be used.  For this to work the `jaxinterp2d` package should be installed with

```bash
pip install git+https://github.com/adam-coogan/jaxinterp2d.git@master
```

The fallback `bilinear` interpolation produced some unexpected errors due to a bug in Herculens.  A bug-fix PR has been submitted and should be in the next version release.


## Features

This code provides the `LensImageRTUGrid` class that is a subclasses Herculens' `LensImage` class.  Two main things have been updated in this subclass:

### Low memory supersampling

One of the main drivers of the memory usage for Hercules is the supersampling value.  A uniform supersampling is applied across the entire observed image and all calculations are done with this larger grid.  For higher resolution images from telescopes like JWST and modest supersampling factors (e.g. 4x), this can quickly cause issues.

To address this, the `lens_surface_brightness` and `source_surface_brightness` methods have been updated to instead calculate a running average of subpixel grids that are all the same size as the observed image.  Using this method give a fixed-memory usage that is driven by the observed image size rather than the supersampling factor.  This is a small performance hit vs working with the full supersampled grid all at once, but the ability to run inference on GPUs with less vRAM makes up for it.

### Pytree PSF class

The original Herculens code bundles the "resize" and "convolve" functions together, these had to be separated back out for these changes as the results are already "resized" by construction of the low memory supersampling code.  A `SimpleFFTConvolve` class is provided to be used with this subclass.  This new class provides an FFT based PSF convolution that is stored as a jax pytree.  As a pytree, an instance of this class can be passed into `jax.jit`ed functions without issue.  **An instance of this class must be passed as an argument to the `model` method.  Any PSF set in the `LensImageRTUGrid` initialization is not used.**

Because an instance is passed directly into the `model` method this opens up the ability to pass in new instances without needing to recompile the `model` function.  This can be useful when creating a pipeline to fit a large number of lenses that potentially have different PSFs (e.g. Euclid) or if you want to do inference on the PSF.

###  Ray-guided Transformed Uniform grid (RTU-grid)

Finally we implement the RTU-grid as presented in Enzi et al. 2026.  Three new subclasses have been defined `LightModelRTU`, `PixelatedLightRTU`, and `LensImageRTUGrid`.

`PixelatedLightRTU` has three new keywords:
- `rtu_grid` (bool): Use an RTU-grid as the source.  `False` is the default
- `rtu_grid_type` (str): Either `'spline'` or `'linear'` and defines how the RTU's transformation is defined.  `'spline'` will fit a smoothing spline for the transform function, `linear` will use linear interpolation to define the transform.  The `'linear'` method is provided as a way to reproduce the plots in Appendix C of Enzi et al. 2026, it is not recommended for use as it is far slower.  `'spline'` is the default.
- `rtu_poly_order` (int): Should be an odd number, the degree of the polynomial used to smooth out inverse-transform function when `'spline'` is used.  `11` is the default.

`LensImageRTUGrid` has one new keyword:
- `rtu_mesh_weights` (NxN array): NxN array the same shape as the observed image providing importance weights for each pixel when defining the transformation.  Smaller pixels will be use in the RTU-grid where these weights are larger.  These weights do not need to be normalized.

`LightModelRTU` has no new associated keywords.

Note: an arc mask must be proved if an RTU-grid is being used.  This can be an array of `1` the same shape as the observed image if you don't want to use an arc mask.

If `return_source_pixels_coords=True` in the model call the pixel coordinates of the source grid will be returned in the following format:

`((grid_center_std, grid_center), (grid_edges_std, grid_edges))`

`std` stand for "standardized coordinates" and is a reference to a "mass-sheet-like transform invariant" coordinate system.  In these coordinates any mass models that are the same up to a sift and uniform scaling will resolve to the same grid.  The other returned values are the "observed" coordinates using the same units as observed image.

- `grid_center_std` (M x 2 array): An M x 2 array where M is `rtu_grid_source_size` containing the `x` and `y` standardized coordinates of the pixel centers in the RTU-grid.
- `grid_center` (M x 2 array): An M x 2 array where M is `rtu_grid_source_size` containing the `x` and `y` observed coordinates of the pixel centers in the RTU-grid.
- `grid_edges_std` (M+1 x 2 array): An M+1 x 2 array where M is `rtu_grid_source_size` containing the `x` and `y` standardized coordinates of the pixel edges in the RTU-grid.
- `grid_edges` (M+1 x 2 array): An M+1 x 2 array where M is `rtu_grid_source_size` containing the `x` and `y` observed coordinates of the pixel edges in the RTU-grid.

"Center" and "edges" are references to the pixel centers and edges of the uniform grid this method maps onto, as a result, when mapped back onto the sky the "centers" might not land half way between the "edges".  This is why both values are returned.

Note: the RTU-grid contains non-square pixels that are dependent on the mass model parameters.  If you want to average together multiple samples from something like an MCMC chain, you will first need to resolve all of the sources onto the *same* pixel grid before taking the average.  Just taking the average in the uniform grid coordinate system and transforming it back to observed space with the average mass parameter will *not* work.

## Example usage

Please see the [example notebook](example_usage.ipynb) for fully worked examples comparing an RTU-grid source to a regular grid source and showing GPU benchmarks computation times and memory usage.
