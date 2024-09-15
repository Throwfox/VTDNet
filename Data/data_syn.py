import numpy as np
import scipy
import os

class AutoregressiveSimulation:
    def __init__(self, gamma, num_simulated_hidden_confounders,num_covariates,num_treatments):
        self.num_covariates = num_covariates
        self.num_confounders = num_simulated_hidden_confounders
        self.num_treatments = num_treatments  
        self.p = 5

        self.gamma_a = gamma
        self.gamma_y = gamma

        self.covariates_coefficients = dict()
        self.covariates_coefficients['treatments'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_covariates, self.num_treatments), treatment_coefficients=True)

        self.covariates_coefficients['covariates'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_covariates, self.num_covariates), variables_coefficients=True)

        self.confounders_coefficients = dict()
        self.confounders_coefficients['treatments'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_confounders, self.num_treatments))
        self.confounders_coefficients['confounders'] = self.generate_coefficients(
            self.p, matrix_shape=(self.num_confounders, self.num_confounders), variables_coefficients=True)

        self.outcome_coefficients = np.array([np.random.normal(0, 1) for _ in range(self.num_confounders + self.num_covariates)])
        self.treatment_coefficients = self.generate_treatment_coefficients()

    def generate_treatment_coefficients(self):
        treatment_coefficients = np.zeros(shape=(self.num_treatments, self.num_covariates + self.num_confounders))
        for treatment in range(self.num_treatments):
            treatment_coefficients[treatment][treatment] = 1.0 - self.gamma_a
            treatment_coefficients[treatment][self.num_covariates] = self.gamma_a
        return treatment_coefficients

    def generate_coefficients(self, p, matrix_shape, variables_coefficients=False, treatment_coefficients=False):
        coefficients = []
        for i in range(p):
            if variables_coefficients:
                diag_elements = [np.random.normal(1.0 - (i+1) * 0.2, 0.2) for _ in range(matrix_shape[0])]
                timestep_coefficients = np.diag(diag_elements)
            elif treatment_coefficients:
                timestep_coefficients = np.random.normal(0, 0.5, size=matrix_shape)
            else:
                timestep_coefficients = np.random.normal(0, 0.5, size=matrix_shape[1])
            normalized_coefficients = timestep_coefficients / p
            coefficients.append(normalized_coefficients)
        return coefficients

    def generate_outcome(self, covariates, confounders, treatment_combination):

        all_variables = np.concatenate((covariates, confounders))

        non_linear_term = np.sin(np.dot(all_variables, self.outcome_coefficients))  
        expanded_treatment = np.tile(treatment_combination, 10) # 10=30/3
        interaction_term = np.sum(np.multiply(covariates, expanded_treatment)) 
        squared_term = np.sum(confounders**2)  
        
        outcome = self.gamma_y * non_linear_term + interaction_term + squared_term
        
        return outcome
    def generate_covariates_single_timestep(self, p, history):

        treatments_history = history['treatments']
        covariates_history = history['covariates']
        past_treatment_coefficients = self.covariates_coefficients['treatments']
        past_covariates_coefficients = self.covariates_coefficients['covariates']

        history_length = len(covariates_history)
        if history_length < p:
            p = history_length

        treatments_sum = np.zeros(shape=(self.num_covariates,))
        covariates_sum = np.zeros(shape=(self.num_covariates,))
        for index in range(p):
            treatments_sum += np.matmul(past_treatment_coefficients[index],
                                        treatments_history[history_length - index - 1])

            covariates_sum += np.matmul(past_covariates_coefficients[index],
                                        covariates_history[history_length - index - 1])

        noise = np.random.normal(0, 0.01, size=(self.num_covariates))
        x_t = treatments_sum + covariates_sum + noise
        x_t = np.clip(x_t, -1, 1) 

        return x_t

    def generate_confounders_single_timestep(self, p, history):

        treatments_history = history['treatments']
        confounders_history = history['confounders']

        past_treatment_coefficients = self.confounders_coefficients['treatments']
        past_confounders_coefficients = self.confounders_coefficients['confounders']

        history_length = len(confounders_history)
        if history_length < p:
            p = history_length

        treatments_sum = np.zeros(shape=(self.num_confounders,))
        confounders_sum = np.zeros(shape=(self.num_confounders,))
        for index in range(p):
            treatments_sum += np.matmul(past_treatment_coefficients[index],
                                        treatments_history[history_length - index - 1])
            confounders_sum += np.matmul(past_confounders_coefficients[index],
                                         confounders_history[history_length - index - 1])

        noise = np.random.normal(0, 0.01, size=(self.num_confounders))
        z_t = treatments_sum + confounders_sum + noise
        z_t = np.clip(z_t, -1, 1)  

        return z_t
    def generate_treatment_assignments_single_timestep(self, p, history):

        confounders_history = history['confounders']
        covariates_history = history['covariates']

        history_length = len(covariates_history)
        if history_length < p:
            p = history_length

        average_covariates = np.zeros(shape=len(covariates_history[-1]))
        average_confounders = np.zeros(shape=len(confounders_history[-1]))

        for index in range(p):
            average_covariates += covariates_history[history_length - index - 1]
            average_confounders += confounders_history[history_length - index - 1]

        all_variables = np.concatenate((average_covariates, average_confounders)).T

        treatment_assignment = np.zeros(shape=(self.num_treatments,))

        for index in range(self.num_treatments):
            aux_normal = 30 * np.dot(all_variables, self.treatment_coefficients[index])  
            treatment_assignment[index] = np.random.binomial(1, scipy.special.expit(aux_normal))  
        return treatment_assignment

    def generate_data_single_patient(self, timesteps):
        x_0 = np.random.normal(0, 2, size=(self.num_covariates,))
        z_0 = np.random.normal(0, 2, size=(self.num_confounders,))
        a_0 = np.zeros(shape=(self.num_treatments,))

        history = dict()
        history['covariates'] = [x_0]
        history['confounders'] = [z_0]
        history['treatments'] = [a_0]

        for t in range(timesteps):
            x_t = self.generate_covariates_single_timestep(self.p, history)
            z_t = self.generate_confounders_single_timestep(self.p, history)

            history['covariates'].append(x_t)
            history['confounders'].append(z_t)

            a_t = self.generate_treatment_assignments_single_timestep(self.p, history)

            history['treatments'].append(a_t)

        return np.array(history['covariates']), np.array(history['confounders']), np.array(history['treatments'])

    def generate_dataset(self, num_patients, max_timesteps):
        dataset = dict()
        dataset['previous_covariates'] = []
        dataset['previous_treatments'] = []
        dataset['covariates'] = []
        dataset['confounders'] = []
        dataset['treatments'] = []
        dataset['sequence_length'] = []
        dataset['outcomes'] = []
        
        for i in range(self.num_treatments):
            dataset[f'cf_outcomes_t{i+1}'] = []

        for patient in range(num_patients):
            timesteps = np.random.randint(int(max_timesteps)-10, int(max_timesteps), 1)[0]
            covariates_history, confounders_history, treatments_history = self.generate_data_single_patient(timesteps + 1)

            covariates = np.vstack((np.array(covariates_history[1:timesteps]),
                                    np.zeros(shape=(max_timesteps - timesteps, self.num_covariates))))

            confounders = np.vstack((np.array(confounders_history[1:timesteps]),
                                    np.zeros(shape=(max_timesteps - timesteps, self.num_confounders))))

            treatments = np.vstack((np.array(treatments_history[1:timesteps]),
                                    np.zeros(shape=(max_timesteps-timesteps, self.num_treatments))))

            factual_outcome = np.array([self.generate_outcome(covariates_history[t], confounders_history[t], treatments_history[t])
                                        for t in range(1, timesteps)])
            factual_outcome = factual_outcome[:, np.newaxis] #[20,1]
            factual_outcome = np.vstack((factual_outcome, np.zeros(shape=(max_timesteps - timesteps, 1))))
            dataset['outcomes'].append(factual_outcome)

            for i in range(self.num_treatments):
                counterfactual_outcomes = np.array([
                    self.generate_outcome(covariates_history[t], confounders_history[t], 
                                          np.where(treatments_history[t].copy() == i, 1 - treatments_history[t][i], treatments_history[t]))
                    for t in range(1, timesteps)
                ])

                counterfactual_outcomes = counterfactual_outcomes[:, np.newaxis]

                padding = np.zeros(shape=(max_timesteps - timesteps, 1))
                counterfactual_outcomes = np.vstack((counterfactual_outcomes, padding))

                dataset[f'cf_outcomes_t{i+1}'].append(counterfactual_outcomes)

            dataset['covariates'].append(covariates)
            dataset['confounders'].append(confounders)
            dataset['treatments'].append(treatments)
            dataset['sequence_length'].append(timesteps)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        return dataset
#params
confounders=1
gamma=0.6
sample_num=5000
lenth=30
num_covariates=30
num_treatments=3

autoregressive = AutoregressiveSimulation(gamma, confounders,num_covariates,num_treatments)
dataset = autoregressive.generate_dataset(sample_num, lenth+1)

# Check if the directory exists, if not, create it
if not os.path.exists('./syn_data'):
    os.makedirs('./syn_data')
np.save('./syn_data/syn_data_gamma_'+str(int(10*gamma))+'.npy',dataset,allow_pickle=True)
