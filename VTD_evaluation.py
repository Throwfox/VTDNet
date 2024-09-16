import argparse
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from utils.data_loader import loader_all
from VTD_modules import VTD
from utils.vtd_utils import evaluate_syn

def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Manage parameters for different datasets and tasks")
    
    # Add arguments for dataset and task
    parser.add_argument('--task', type=str, required=True, help='Task to perform')
    parser.add_argument('--model', default='VTD',type=str, required=False, help='model name')
    
    # Parse the arguments
    args = parser.parse_args()
    if args.task not in ['0','1','2', '3','4','5', '6','7','8' ,'9']:
        print(f"Error: Invalid task '{args.task}' for dataset 'syn'. Choose from ['0','1','2', '3','4','5', '6','7','8' ,'9'].")
        return
    
    
    #-->> put here to fix rmse (fix the seed to get consistent result)
    torch.manual_seed(3407)
    # ------------------------------------------------------------
    
    dataset = 'syn'
    input_dim = 30
    head=4 # 1
    hidden_dim = 256 # 128
    latent_dim = 256 # 128
    output_dim = 1
    treatment_dim = 3
    length = 30
    batch_size = 64
    epochs = 500
    
    print('-------Inference-----')
    print('dataset:{}  '.format(dataset),'task:{}  '.format(args.task), 'model:{}  '.format(args.model))
    
    
    _, _, test_loader = loader_all(dataset, args.task, batch_size) 
    
    
    model_save_path = './checkpoints/best_model_'+dataset+'_'+args.task+'_'+args.model+'_'+str(epochs)+'epoch.pth'

    model = VTD(input_dim, hidden_dim, latent_dim, output_dim, treatment_dim, head, length)
    model.load_state_dict(torch.load(model_save_path))

    all_predicted_outcomes,all_true_outcomes,all_masks,ite_samples_original=evaluate_syn(model, dataset,test_loader)
                                                                                                                          
if __name__ == '__main__':
    main()
