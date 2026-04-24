import numpy as np
import pickle
import pyvinecopulib as pv
import matplotlib.pyplot as plt
from probability_funcs import forward_cdf, density, xy2xy_parameteric_cdf_transform, xy2xy_empirical_cdf_transform, fit_all_marginals , fit_all_marginals_empirical
from utils import plot_marginal_fits
from copula_funcs import train_copulas_parametric, train_copulas_empirical, get_completeness_parametric, get_completeness_empirical


class CopulaCompletenessEstimator:
    def __init__(self, type='parametric'):
        self.xy_input = None
        self.uv_input = None
        self.copula_fn = None
        self.copula_all = None
        self.pdf_transformations = None
        self.pi_fn= None
        self.completeness = None
        self.selection_mask = None
        self.xy_not_training_set = True

        if type not in ['parametric', 'empirical']:
            raise ValueError("type must be 'parametric' or 'empirical'")
        self.type = type
        
    

    def train_copula_model(self, make_plots=True):
        
        self.xy_not_training_set = False
        self.xy_fn_training = self.xy_fn

        x_all, x_fn = self.xy_input[:, 0], self.xy_fn_training[:, 0]
        y_all, y_fn = self.xy_input[:, 1], self.xy_fn_training[:, 1]

        if self.type == 'parametric':
            self.pdf_transformations = fit_all_marginals(x_all, x_fn, y_all, y_fn)


        if self.type == 'empirical':
            self.pdf_transformations = fit_all_marginals_empirical(x_all, x_fn, y_all, y_fn)

        if make_plots:
            data_map    = {"x_all": x_all, "x_fn": x_fn, "y_all": y_all, "y_fn": y_fn}
            plot_marginal_fits(self.pdf_transformations, data_map)

        self.u_all = self.pdf_transformations['x_all']['u']
        self.v_all = self.pdf_transformations['y_all']['u']
        self.u_fn = self.pdf_transformations['x_fn']['u']
        self.v_fn = self.pdf_transformations['y_fn']['u']
        self.uv_all = np.column_stack((self.u_all, self.v_all))
        self.uv_fn = np.column_stack((self.u_fn, self.v_fn))
        


        if self.type == 'parametric':
            self.copula_all, self.copula_fn = train_copulas_parametric(
                self.uv_all, self.uv_fn,
                make_plots=make_plots
            )

        elif self.type == 'empirical':
            self.copula_all, self.copula_fn = train_copulas_empirical(
                self.uv_all, self.uv_fn,
                make_plots=make_plots
            )
        

    def save_copula_model(self, path):
        pickle.dump(self.copula_fn, open(path + '_fn.pkl', 'wb'))
        pickle.dump(self.copula_all, open(path + '_all.pkl', 'wb'))
        pickle.dump(self.pdf_transformations, open(path + '_pdf_transforms.pkl', 'wb'))
        pickle.dump(self.pi_fn, open(path + '_pi_fn.pkl', 'wb'))


    def load_copula_model(self, path):
        self.copula_fn = pickle.load(open(path + '_fn.pkl', 'rb'))
        self.copula_all = pickle.load(open(path + '_all.pkl', 'rb'))
        self.pdf_transformations = pickle.load(open(path + '_pdf_transforms.pkl', 'rb'))
        self.pi_fn = pickle.load(open(path + '_pi_fn.pkl', 'rb'))


    def find_completeness(self):
        if self.type == 'parametric':
            self.completeness = get_completeness_parametric(
                self.xy_input, self.copula_all, self.copula_fn, 
                self.pi_fn, self.pdf_transformations, 
                apply_xy2xy_transform=self.xy_not_training_set
            )
        elif self.type == 'empirical':
            self.completeness = get_completeness_empirical(
                self.xy_input, self.copula_all, self.copula_fn, 
                self.pi_fn, self.pdf_transformations, 
                apply_xy2xy_transform=self.xy_not_training_set
            )
        return self.completeness
    

    def apply_completeness_as_selection(self, seed=42):
        np.random.seed(seed)
        uniform_randoms = np.random.rand(len(self.completeness))
        self.selection_mask = uniform_randoms < self.completeness
        return self.selection_mask
        

    def set_xy(self, xy):
        # Check xy is a numpy array of shape (n, 2)
        if not isinstance(xy, np.ndarray):
            raise ValueError("xy must be a numpy array")
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must be of shape (n, 2)")
        self.xy_input = xy
        self.xy_not_training_set = True


    def set_xy_fn_mask(self, xy_true_positive_mask):
        self.xy_tp_mask = xy_true_positive_mask
        self.xy_fn = self.xy_input[~xy_true_positive_mask]
        self.pi_fn = len(self.xy_fn) / len(self.xy_input)


