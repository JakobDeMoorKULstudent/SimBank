from copy import copy
from src.utils.inference import Forward_pass
import numpy as np
import random
from skopt import gp_minimize

class BOZORGIModelValidator(Forward_pass):
    def __init__(self, full_dataset_params, s_learner=True):
        super().__init__()
        self.full_dataset_params = full_dataset_params
        self.s_learner = s_learner


    def threshold_tuning(self, CI_evaluator, model_list, iteration, max_th=1000, step=-100, model_params=None):
        overall_seed = copy(model_params["random_seed"]) + iteration*5
        random.seed(overall_seed)
        np.random.seed(overall_seed)

        self.CI_evaluator = CI_evaluator
        self.model_list = model_list
        self.iteration = iteration
        print("Tuning: ")

        # First do it for th = 0 and get the max y_diff
        _, _ = self.CI_evaluator.model_policy_inference(self.full_dataset_params["test_val_size"], self.model_list, 0, self.iteration, tuning=True)
        if isinstance(self.CI_evaluator.max_y_diff, int):
            max_th = (self.CI_evaluator.max_y_diff)
            if self.s_learner:
                max_th = max_th - 500
            
        else:
            max_th = (self.CI_evaluator.max_y_diff.flatten()[0])
            if self.s_learner:
                max_th = max_th - 500
        
        print("     Max y_diff: ", max_th)

        th_search_space = [(0, max_th)]

        result = gp_minimize(self.objective_function, th_search_space, n_calls=10, random_state=overall_seed, n_initial_points=3)
        opt_th = result.x[0]
        opt_obj = -result.fun

        print("     Threshold:")
        print('         - Optimal threshold: ', opt_th)
        print('         - Optimal objective: ', opt_obj)
        print('\n\n\n')

        return opt_th, opt_obj
    
    def objective_function(self, th):
        performance, _ = self.CI_evaluator.model_policy_inference(self.full_dataset_params["test_val_size"], self.model_list, th, self.iteration, tuning=True)
        return -performance