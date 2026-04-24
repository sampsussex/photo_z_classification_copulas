import numpy as np
import pyvinecopulib as pv
import matplotlib.pyplot as plt
from probability_funcs import forward_cdf, density, xy2xy_parameteric_cdf_transform, xy2xy_empirical_cdf_transform

def train_copulas_empirical(uv_all, uv_fn, make_plots=False):
    controls_np = pv.FitControlsBicop(family_set=[pv.BicopFamily.tll])
    cop_np_all = pv.Bicop.from_data(data=uv_all, controls=controls_np)
    cop_np_fn  = pv.Bicop.from_data(data=uv_fn,  controls=controls_np)

    print('Empirical TLL copula fits:')
    print(f'All: {cop_np_all}')
    print(f'FN: {cop_np_fn}')

    if make_plots:
        print('All Fits...')
        cop_np_all.plot()
        print('FN Fits...')
        cop_np_fn.plot()

    return cop_np_all, cop_np_fn


def train_copulas_parametric(uv_all, uv_fn, make_plots=False):
    controls = pv.FitControlsBicop(family_set=pv.parametric)
    cop_all = pv.Bicop.from_data(data=uv_all, controls=controls)
    cop_fn = pv.Bicop.from_data(data=uv_fn, controls=controls)

    print('Parametric copula fits:')
    print(f'All: {cop_all}')
    print(f'FN: {cop_fn}')
    if make_plots:
        print('All Fits...')
        cop_all.plot()
        print('FN Fits...')
        cop_fn.plot()

    return cop_all, cop_fn


def get_completeness_parametric(xy_input, copula_all, copula_fn, pi_fn, pdf_transformations, apply_xy2xy_transform=False):
    """Compute completeness at xy_input by comparing PDFs from copula_all vs copula_fn.
    mag all and gi are input features, copula_all and copula_fn are the fitted copulas for the full vs FN (False Negative) subsets."""

    if apply_xy2xy_transform:
        xy_input = xy2xy_parameteric_cdf_transform(xy_input, pdf_transformations)
    
    u_input_all = forward_cdf(xy_input[:,0], 'x_all', pdf_transformations, model_type='parametric')
    v_input_all = forward_cdf(xy_input[:,1], 'y_all', pdf_transformations, model_type='parametric')


    u_input_fn = forward_cdf(xy_input[:,0], 'x_fn', pdf_transformations, model_type='parametric')
    v_input_fn = forward_cdf(xy_input[:,1], 'y_fn', pdf_transformations, model_type='parametric')

    

    uv_input_all = np.column_stack((u_input_all, v_input_all))
    uv_input_fn = np.column_stack((u_input_fn, v_input_fn))
    print(uv_input_all)
    u_all = copula_all.pdf(uv_input_all)
    u_fn = copula_fn.pdf(uv_input_fn)

    den_x_all = density(xy_input[:, 0], 'x_all', pdf_transformations, model_type='parametric')
    den_x_fn = density(xy_input[:, 0], 'x_fn', pdf_transformations, model_type='parametric')

    den_y_all = density(xy_input[:, 1], 'y_all', pdf_transformations, model_type='parametric')
    den_y_fn = density(xy_input[:, 1], 'y_fn', pdf_transformations, model_type='parametric')

    den_fn = u_fn * den_x_fn * den_y_fn
    den_all = u_all * den_x_all * den_y_all


    completeness = 1 - den_fn * pi_fn  / (den_all + 1e-12)
    # Clip completeness to [0,1]
    completeness = np.clip(completeness, 0, 1)
    return completeness


def get_completeness_empirical(xy_input, copula_all, copula_fn, pi_fn, pdf_transformations, apply_xy2xy_transform=False):
    """Compute completeness at xy_input by comparing PDFs from copula_all vs copula_fn.
    mag all and gi are input features, copula_all and copula_fn are the fitted copulas for the full vs FN (False Negative) subsets."""

    xy_original = np.column_stack((pdf_transformations['x_all']['x'], pdf_transformations['y_all']['x']))

    if apply_xy2xy_transform:
        xy_input = xy2xy_empirical_cdf_transform(xy_input, xy_original)
    
    u_input_all = forward_cdf(xy_input[:,0], 'x_all', pdf_transformations, model_type='empirical')
    v_input_all = forward_cdf(xy_input[:,1], 'y_all', pdf_transformations, model_type='empirical')
    uv_input_all = np.column_stack((u_input_all, v_input_all))


    u_input_fn = forward_cdf(xy_input[:,0], 'x_fn', pdf_transformations, model_type='empirical')
    v_input_fn = forward_cdf(xy_input[:,1], 'y_fn', pdf_transformations, model_type='empirical')
    uv_input_fn = np.column_stack((u_input_fn, v_input_fn))

    u_all = copula_all.pdf(uv_input_all)
    u_fn = copula_fn.pdf(uv_input_fn)

    den_x_all = density(xy_input[:, 0], 'x_all', pdf_transformations, model_type='empirical')
    den_x_fn = density(xy_input[:, 0], 'x_fn', pdf_transformations, model_type='empirical')

    den_y_all = density(xy_input[:, 1], 'y_all', pdf_transformations, model_type='empirical')
    den_y_fn = density(xy_input[:, 1], 'y_fn', pdf_transformations, model_type='empirical')

    den_fn = u_fn * den_x_fn * den_y_fn
    den_all = u_all * den_x_all * den_y_all
    # Compute densities at the recovered points

    completeness = 1 - den_fn * pi_fn  / (den_all + 1e-12)
    # Clip completeness to [0,1]
    completeness = np.clip(completeness, 0, 1)
    return completeness