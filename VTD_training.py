import argparse
import torch
import torch.optim as optim
import os
from utils.data_load import loader_all
import numpy as np
from VTD_modules import VTD,trainer

def main():
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Manage parameters for different datasets and tasks")
    
    # Add arguments for dataset and task
    parser.add_argument('--model', default='VTD',type=str, required=False, help='model name')
    parser.add_argument('--dataset', type=str, required=True, choices=['syn'], help='Dataset to use')
    parser.add_argument('--task', type=str, required=True, help='Task to perform')
    parser.add_argument('--cuda', type=str, required=True, help='cuda device')
    parser.add_argument("--resume", type=int, choices=[0,1],required=True, help="resume,o:false, 1:true")

    # Parse the arguments
    args = parser.parse_args()
        
    # Check and manage dataset and task combinations
    if args.task not in ['0','1','2', '3','4','5', '6','7','8' ,'9']:
        print(f"Error: Invalid task '{args.task}' for dataset 'nacc'. Choose from ['0','1','2', '3','4','5', '6','7','8' ,'9'].")
        return

    # Proceed with the specific dataset and task
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")

    # Your logic here based on dataset and task
    # hyperparameter
    args.input_dim = 30
    args.head=4 
    args.hidden_dim = 256 
    args.latent_dim = 256
    args.output_dim = 1
    args.treatment_dim = 3
    args.length = 30
    args.batch_size = 64
    args.epochs = 500
    args.lr = 1e-4
    args.weight_decay = 1e-5
    args.alpha = 50 #adjust the outcome loss 
    # train
    train(args)

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(np.random.randint(3407))
    
    print('-------Training-----')
    print('dataset:{}  '.format(args.dataset),'task:{}  '.format(args.task))

    train_loader, val_loader, test_loader = loader_all(args.dataset, args.task, args.batch_size) 
    #build model
    '''
    wandb.init(project='VTD',
        entity="",
        sync_tensorboard=False,
        name= 'vtd_'+args.dataset+'_'+args.task+'_'+str(args.epochs)+'epoch_r'+str(r)+'_VTDv3',        
        config={
        "learning_rate": args.lr,
        "architecture": "vtd",
        "epochs": args.epochs,
        },
        reinit=True)
    '''
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    model_save_path = './checkpoints/best_model_'+args.dataset+'_'+args.task+'_'+args.model+'_'+str(args.epochs)+'epoch.pth'  
    model = VTD(args.input_dim, args.hidden_dim, args.latent_dim, args.output_dim, args.treatment_dim, args.head, args.length)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    trainer(model, train_loader, val_loader, optimizer, args.epochs, args.alpha, model_save_path, resume=args.resume)

if __name__ == '__main__':
    main()
