# SimBank: from Simulation to Solution in Prescriptive Process Monitoring
This repository provides the code for the paper *"SimBank: from Simulation to Solution in Prescriptive Process Monitoring"*.

The structure of the code is as follows:
```
SimBank from Simulation to Solution in Prescriptive Process Monitoring/
|_ config/                          
|_ data/                            # Data generated and used in experiments
|_ res/                             # Results of the experiments
|_ SimBank/                         # The full SimBank simulator
  |_ activity_execution.py                    # Execute an acivity in the process            
  |_ confounding level.py                     # Create a dataset with a specified confounding bias level
  |_ extra_flow_conditions.py                 # The (extra) underlying mechanism of the control-flow         
  |_ SimBank_Generator_Guide.ipynb            # **SimBank Guide**: a complete guide on how to use SimBank 
  |_ petri_net_generator.py                   # Setup initial control-flow      
  |_ simulation.py                            # Main simulator code
|_ scripts/
  |_ BOZORGI_run.py                 # Experiments of the RealCause-based S-learner (based on Bozorgi et. al)                    
  |_ BRANCHI_run.py                 # Experiments of the K-means-based Q-learning (based on Branchi et. al)
  |_ CI_run.py                      # Experiments of the S-learner  
  |_ RL_run.py                      # Experiments of the Deep Q-learning      
|_ src/        
  |_ methods/
    |_ BOZORGI/                               # Code for the RealCause-based S-learner         
      |_ BOZORGI_models/                                # RealCause models
      |_ BOZORGI_calc_atoms.py                          # Code for calculating concentration of discrete values in distribution (atoms) 
      |_ BOZORGI_data_generation.py                     # Code for data generation for RealCause model        
      |_ BOZORGI_data_preparation.py                    # Code for data preprocessing         
      |_ BOZORGI_evaluation.py                          # Code for policy evaluation 
      |_ BOZORGI_realcause.py                           # Code for RealCause training & testing
      |_ BOZORGI_utils.py                               
      |_ BOZORGI_validation.py                          # Code for policy validation (threshold tuning)  
    |_ BRANCHI/                               # Code for the K-means-based Q-learning 
      |_ BRANCHI_RL_evaluation.py                       # Code for policy evaluation
      |_ BRANCHI_RL_training.py                         # Code for training        
      |_ BRANCHI_RL_data_preparation.py                 # Code for data preprocessing (including K-means model)         
      |_ BRANCHI_utils.py    
    |_ CI/                                    # Code for the S-learner         
      |_ CI_data_preparation.py                         # Code for data preprocessing
      |_ CI_evaluation.py                               # Code for policy evaluation  
      |_ CI_training.py                                 # Code for training        
      |_ CI_validation.py                               # Code for policy validation (threshold tuning)
    |_ CI/                                    # Code for the S-learner         
      |_ RL_data_preparation.py                         # Code for data preprocessing
      |_ RL_evaluation.py                               # Code for policy evaluation  
      |_ RL_training.py                                 # Code for training
  |_ utils/
    |_ inference.py                                   
    |_ loss_functions.py                              
    |_ models.py                              # Code for model initialization
    |_ tools.py      
    |_ utils.py/
```

## Installation.
The ```requirements.txt``` provides the necessary packages.
All code was written for ```python 3.10.13```.

## SimBank
The complete SimBank simulator can be found in the ```SimBank/``` folder. The file SimBank_Generator_Guide contains a full walkthrough on how to use SimBank to create offline datasets and allow online training for each intervention, to vary the confounding level, and to evaluate method policies.

## Experiments of the paper
Download the data for the experiments from [Google Drive](https://drive.google.com/drive/folders/1CGOKpd7NU-brk9PpiO6nJcVYp3idi97E?usp=sharing). 

Put the data in the ```data/``` folder. Now, the results from the paper can be reproduced by setting the ```path``` variable in the config/config.py file to your directory and running the appropriate script.

Download the results of the experiments from [Google Drive](https://drive.google.com/drive/folders/1CGOKpd7NU-brk9PpiO6nJcVYp3idi97E?usp=sharing). 