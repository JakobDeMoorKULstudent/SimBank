path = "c:\\Users\\u0166838\\OneDrive - KU Leuven\\Documents\\Doc\\Code\\SimBank from Simulation to Solution in Prescriptive Process Monitoring"
data_folder = "data"
results_folder = "res"
import sys
sys.path.append(path)
sys.path.append(path + "\\SimBank")

import pandas as pd
import numpy as np
import math
from copy import deepcopy
from src.utils.inference import Forward_pass
import random
from SimBank.activity_execution import ActivityExecutioner
from SimBank import simulation
from src.methods.BRANCHI import BRANCHI_data_preparation
from src.methods.BOZORGI import BOZORGI_data_preparation

val_multiplier = 1000

class RLModelEvaluator(Forward_pass):
    def __init__(self, model_params, int_dataset_params, full_dataset_params, prep_utils, print_cases=False, print_transitions=False, realcause_model_list=None, realcause_prep_utils=None):
        super().__init__()
        self.int_dataset_params = int_dataset_params
        self.full_dataset_params = full_dataset_params
        self.intervention_total_len = sum(self.full_dataset_params["intervention_info"]["len"])
        self.prep_utils = prep_utils
        self.print_cases = print_cases
        self.print_transitions = print_transitions
        self.realcause_model_list = realcause_model_list
        self.realcause_prep_utils = realcause_prep_utils
        random.seed(full_dataset_params["random_seed_test"])
        np.random.seed(full_dataset_params["random_seed_test"])

    def get_bank_best_action(self, prefix_list, current_int_index):
        prefix_without_int = prefix_list[0][0:-1]
        prev_event = prefix_without_int[-1]
        action_index = 0
        
        if self.full_dataset_params["intervention_info"]["name"][current_int_index] == "time_contact_HQ":
            cancel_condition = ((prev_event["unc_quality"] == 0 and prev_event["est_quality"] < self.full_dataset_params["policies_info"]["min_quality"] and prev_event["noc"] >= self.full_dataset_params["policies_info"]["max_noc"]) or (prev_event["noc"] >= self.full_dataset_params["policies_info"]["max_noc"] and prev_event["unc_quality"] > 0))
            contact_condition = (prev_event["noc"] < 2 and prev_event["unc_quality"] == 0 and prev_event["amount"] > 10000 and prev_event["est_quality"] >= self.full_dataset_params["policies_info"]["min_quality"])

            if cancel_condition:
                action_index = 0
            elif contact_condition:
                action_index = 1
        
        elif self.full_dataset_params["intervention_info"]["name"][current_int_index] == "choose_procedure":
            priority_condition = (prev_event["amount"] > self.full_dataset_params["policies_info"]["choose_procedure"]["amount"] and prev_event["est_quality"] >= self.full_dataset_params["policies_info"]["choose_procedure"]["est_quality"])

            if priority_condition:
                action_index = 1
            else:
                action_index = 0
        
        elif self.full_dataset_params["intervention_info"]["name"][current_int_index] == "set_ir_3_levels":
            activity_executioner = ActivityExecutioner()
            ir, _, _ = activity_executioner.calculate_offer(prev_event=prev_event, intervention_info=self.full_dataset_params["intervention_info"])
            action_index = self.full_dataset_params["intervention_info"]["actions"][current_int_index].index(ir)
        
        return action_index
    
    def bank_policy_inference(self, n_cases, validation=False):
        # GENERATING BANK POLICY ONLINE TO MAINTAIN SAME TEST SET AS MODEL

        #Init performance metrics
        bank_performance = 0
        full_dataset_params = deepcopy(self.full_dataset_params)
        if validation:
            full_dataset_params["random_seed_test"] = full_dataset_params["random_seed_test"]*val_multiplier

        #Init data generator
        case_gen = simulation.PresProcessGenerator(self.full_dataset_params, seed=full_dataset_params["random_seed_test"])
        
        self.bank_actions = {}

        #Run
        for case_nr in range(n_cases):
            prefix_list = []
            prefix_list = case_gen.start_simulation_inference(seed_to_add=case_nr)
            while case_gen.int_points_available:
                bank_best_action = self.get_bank_best_action(prefix_list, case_gen.current_int_index)
                # Break if intervention done or in last timing
                prefix_list = case_gen.continue_simulation_inference(bank_best_action)

                if self.print_cases:
                    print("Bank best action", bank_best_action, '\n')

                if bank_best_action not in self.bank_actions:
                    self.bank_actions[bank_best_action] = 1
                else:
                    self.bank_actions[bank_best_action] += 1

            full_case = case_gen.end_simulation_inference()
            full_case = pd.DataFrame(full_case)
            bank_performance += full_case["outcome"].iloc[-1]

            if self.print_cases:
                print("Bank full case", full_case, "\n")
        
        print("Bank policy actions: ", self.bank_actions, "\n")

        return bank_performance

    def random_policy_inference(self, n_cases, validation=False):
        #Init performance metrics
        random_performance = 0
        full_dataset_params = deepcopy(self.full_dataset_params)
        if validation:
            full_dataset_params["random_seed_test"] = full_dataset_params["random_seed_test"]*val_multiplier

        #Init data generator
        case_gen = simulation.PresProcessGenerator(self.full_dataset_params, seed=full_dataset_params["random_seed_test"])

        #Run
        for case_nr in range(n_cases):
            prefix_list = []
            prefix_list = case_gen.start_simulation_inference(seed_to_add=case_nr)
            timings = [0] * len(self.full_dataset_params["intervention_info"]["name"])
            random_best_timings = []
            for int_index in range(len(self.full_dataset_params["intervention_info"]["name"])):
                if self.full_dataset_params["intervention_info"]["name"][int_index] == "time_contact_HQ":
                    random_best_timings.append(random.choice(range(self.full_dataset_params["intervention_info"]["action_depth"][int_index])) * 2)
                else:
                    random_best_timings.append(0)
            random_best_action = 0 # control
            while case_gen.int_points_available:
                if timings[case_gen.current_int_index] == random_best_timings[case_gen.current_int_index]:
                    random_best_action = random.choice(range(self.full_dataset_params["intervention_info"]["action_width"][case_gen.current_int_index]))
                timings[case_gen.current_int_index] += 1
                # Break if intervention done or in last timing
                prefix_list = case_gen.continue_simulation_inference(random_best_action)

                if self.print_cases:
                    print("Random best action", random_best_action, '\n')

            full_case = case_gen.end_simulation_inference()
            full_case = pd.DataFrame(full_case)
            random_performance += full_case["outcome"].iloc[-1]

            if self.print_cases:
                print("Full case", full_case, "\n")
        
        return random_performance

    def model_policy_inference(self, n_cases, model_class, q_table, kmeans, validation=False, calc_realcause_performance=False, iteration=None):
        full_dataset_params = deepcopy(self.full_dataset_params)
        if validation:
            full_dataset_params["random_seed_test"] = full_dataset_params["random_seed_test"]*val_multiplier
        else:
            self.test_set_df = pd.DataFrame([])
        case_gen = simulation.PresProcessGenerator(self.full_dataset_params, seed=full_dataset_params["random_seed_test"])
        case_prep = BRANCHI_data_preparation.LoanProcessPreprocessor(dataset_params=self.int_dataset_params[0]) # Just take the first one, does not matter as we only need the scaler, 
        model_performance = 0
        model_realcause_performance = 0
        case_prep_realcause = BOZORGI_data_preparation.LoanProcessPreprocessor(dataset_params=self.int_dataset_params[0])
        self.model_actions = {}
        self.model_action_timings = {0: 0, 2: 0, 4: 0, 6: 0, 8: 0, 100: 0}

        for i_episode in range(n_cases):
            if i_episode % 500 == 0:
                print('Episode: ', i_episode)
            int_executed = False
            int_point_available = False
            realcause_outcome_list = []
            prefix_list = []
            prefix_list = case_gen.start_simulation_inference(seed_to_add=i_episode)
            if prefix_list != []:
                # Define initial state
                running_df = pd.DataFrame([prefix_list[0][0]])
                state_preproc = case_prep.preprocess_kmeans_RL(data_sample=running_df, reward_preproc = 0, prep_utils=self.prep_utils, kmeans=kmeans)

                if self.print_transitions:
                    print('\n\n\n')
                    print('NEW EPISODE: ', i_episode)
                    print('\n')
                    print('initial running_df:', running_df)
                    print("\n")
                
            self.current_timing = 0
            while case_gen.int_points_available:
                int_point_available = True
                action_taken, current_int_action_index = model_class.select_action(state_preproc, case_gen.current_int_index, int_point = True, exploit = True, current_timing=self.current_timing, q_table=q_table)

                if action_taken not in self.model_actions:
                    self.model_actions[action_taken] = 1
                else:
                    self.model_actions[action_taken] += 1

                if self.full_dataset_params["intervention_info"]["name"] == ["time_contact_HQ"]:
                    if self.current_timing % 2 != 0:
                        if current_int_action_index == 1:
                            current_int_action_index = 0
                        if self.print_transitions:
                            print("not intervention point for time_contact_HQ (validate_application)")
                    if current_int_action_index == 1:
                        int_executed = True
                        self.model_action_timings[self.current_timing] += 1
                
                if calc_realcause_performance:
                    if (self.current_timing == 6 or current_int_action_index == 1) and self.full_dataset_params["intervention_info"]["name"] == ["time_contact_HQ"]:
                        realcause_outcome_list = self.get_realcause_outcome(prefix_list[current_int_action_index], case_prep_realcause, self.realcause_prep_utils[case_gen.current_int_index], current_int_action_index, iteration)
                
                # Define next state
                prefix_list = case_gen.continue_simulation_inference(current_int_action_index)
                if self.full_dataset_params["intervention_info"]["name"] == ["time_contact_HQ"]:
                    self.current_timing += 1
                if not case_gen.int_points_available:
                    break
                running_df = pd.DataFrame(prefix_list[0][:-1]) # Just take the first prefix (both are the same)
                next_state_preproc = case_prep.preprocess_kmeans_RL(data_sample=running_df, reward_preproc = 0, prep_utils=self.prep_utils, kmeans=kmeans)
                
                # Observe reward (will always be 0 as process is not finished yet)
                reward = 0
                state_preproc = deepcopy(next_state_preproc)

                if self.print_transitions:
                    print("NEW STATE")
                    print("action taken:", action_taken)
                    print('next_running_df:', running_df)
                    print("reward:", reward)
                    print("\n")

            if (not int_executed) and int_point_available:
                self.model_action_timings[8] += 1
            if not int_point_available:
                self.model_action_timings[100] += 1

            # Terminal
            full_case = case_gen.end_simulation_inference()
            full_case = pd.DataFrame(full_case)
            model_performance += full_case["outcome"].iloc[-1]
            if calc_realcause_performance:
                if len(realcause_outcome_list) > 0:
                    model_realcause_performance += realcause_outcome_list[-1]
                else:
                    model_realcause_performance += full_case["outcome"].iloc[-1]

            if not validation:
                self.test_set_df = pd.concat([self.test_set_df, full_case], axis=0)

            if self.print_transitions:
                print("TERMINAL STATE")
                print('full case:', full_case)
                print("outcome model actions: ", full_case["outcome"].iloc[-1])
                print("\n")

        print('Evaluation complete')
        return model_performance, model_realcause_performance
    

    def get_realcause_outcome(self, prefix, case_prep, prep_utils, chosen_action, iteration=None):
        prefix = pd.DataFrame(prefix)
        prep_prefix = case_prep.preprocess_sample_bozorgi_retaining(prefix, prep_utils, self.print_cases)
        prep_prefix = prep_prefix.drop(columns=["treatment"])
        realcause_outcome_list = []
        outcome_scaler = prep_utils["scaler_dict"]["outcome"]
        if iteration is None:
            realcause_model_list = self.realcause_model_list
        else:
            realcause_model_list = [self.realcause_model_list[iteration]]
        for model in realcause_model_list:
            # w is equal to prep_prefix without the last column (treatment)
            w = prep_prefix.to_numpy()
            t = np.array([chosen_action] * len(w))
            t = t.reshape(1, 1)
            y0, y1 = model.sample_y(
                t, w, ret_counterfactuals=True, seed=self.full_dataset_params["random_seed_test"]
            )
            if chosen_action == 0:
                y_pred_norm = y0
            else:
                y_pred_norm = y1
            y_pred = outcome_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))
            realcause_outcome_list.append(y_pred)
            if self.print_cases:
                print('t', t)
                print('y0', outcome_scaler.inverse_transform(y0.reshape(-1, 1)), 'y1', outcome_scaler.inverse_transform(y1.reshape(-1, 1)), 'y_pred', y_pred)
        if self.print_cases:
            print("prefix", prefix)
            print("prep_prefix", prep_prefix)
            print("realcause_outcome_list", realcause_outcome_list)
        # flatten list
        realcause_outcome_list = [item for sublist in realcause_outcome_list for item in sublist]
        # flatten one more time
        realcause_outcome_list = [item for sublist in realcause_outcome_list for item in sublist]
        return realcause_outcome_list