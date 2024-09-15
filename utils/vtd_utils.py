import torch
from torch.autograd import Variable

def rmse(predictions, targets, mask):
    return torch.sqrt(((predictions - targets) ** 2 * mask).sum() / mask.sum())
    
def mae(predictions, targets, mask):
    return (torch.abs(predictions - targets) * mask).sum() / mask.sum()

def ite_calculate(model,x, delta_t,treatment,activate,task):
    treatment_case=torch.ones_like(treatment).cuda()
    treatment_contrl=torch.zeros_like(treatment).cuda()
    if task=='mimic':
        treatment_mech_case= torch.cat((treatment_case[:,:,0].unsqueeze(-1),treatment[:,:,1].unsqueeze(-1)), dim=-1)
        treatment_vaso_case= torch.cat((treatment[:,:,0].unsqueeze(-1),treatment_case[:,:,1].unsqueeze(-1)), dim=-1)
        treatment_mech_ctrl= torch.cat((treatment_contrl[:,:,0].unsqueeze(-1),treatment[:,:,1].unsqueeze(-1)), dim=-1)
        treatment_vaso_ctrl= torch.cat((treatment[:,:,0].unsqueeze(-1),treatment_contrl[:,:,1].unsqueeze(-1)), dim=-1)
        _,_,_,_,_,outcome_pred_mech_case=model(x, delta_t,treatment_mech_case,activate)
        _,_,_,_,_,outcome_pred_mech_ctrl=model(x, delta_t,treatment_mech_ctrl,activate)
        _,_,_,_,_,outcome_pred_vaso_case=model(x, delta_t,treatment_vaso_case,activate)
        _,_,_,_,_,outcome_pred_vaso_ctrl=model(x, delta_t,treatment_vaso_ctrl,activate)
        ite_sub_orginal= torch.cat(((outcome_pred_mech_case-outcome_pred_mech_ctrl),(outcome_pred_vaso_case-outcome_pred_vaso_ctrl)),dim=-1)
    else:
        treatment_1st_case = torch.cat((treatment_case[:,:,0].unsqueeze(-1),treatment[:,:,1].unsqueeze(-1), treatment[:,:,2].unsqueeze(-1)),dim=-1)
        treatment_2nd_case = torch.cat((treatment[:,:,0].unsqueeze(-1),treatment_case[:,:,1].unsqueeze(-1),treatment[:,:,2].unsqueeze(-1)), dim=-1)
        treatment_3th_case= torch.cat((treatment[:,:,0].unsqueeze(-1),treatment[:,:,1].unsqueeze(-1),treatment_case[:,:,2].unsqueeze(-1)), dim=-1)
        treatment_1st_ctrl= torch.cat((treatment_contrl[:,:,0].unsqueeze(-1),treatment[:,:,1].unsqueeze(-1),treatment[:,:,2].unsqueeze(-1)), dim=-1)   
        treatment_2nd_ctrl= torch.cat((treatment[:,:,0].unsqueeze(-1),treatment_contrl[:,:,1].unsqueeze(-1),treatment[:,:,2].unsqueeze(-1)), dim=-1)
        treatment_3th_ctrl= torch.cat((treatment[:,:,0].unsqueeze(-1),treatment[:,:,1].unsqueeze(-1),treatment_contrl[:,:,2].unsqueeze(-1)), dim=-1)
        
        _,_,_,_,_,outcome_pred_1st_case = model(x, delta_t,treatment_1st_case,activate)
        _,_,_,_,_,outcome_pred_1st_ctrl = model(x, delta_t,treatment_1st_ctrl,activate)
        _,_,_,_,_,outcome_pred_2nd_case = model(x, delta_t,treatment_2nd_case,activate)
        _,_,_,_,_,outcome_pred_2nd_ctrl = model(x, delta_t,treatment_2nd_ctrl,activate)
        _,_,_,_,_,outcome_pred_3th_case = model(x, delta_t,treatment_3th_case,activate)
        _,_,_,_,_,outcome_pred_3th_ctrl = model(x, delta_t,treatment_3th_ctrl,activate)        
        ite_sub_orginal= torch.cat(((outcome_pred_1st_case-outcome_pred_1st_ctrl),(outcome_pred_2nd_case-outcome_pred_2nd_ctrl),(outcome_pred_3th_case-outcome_pred_3th_ctrl)),dim=-1)    
    return ite_sub_orginal   
    
def ite_mask(ite, mask):
    if torch.isnan(ite).any() or torch.isinf(ite).any():
        print("NaN or Inf found in ite")
    mask_expanded = mask.expand(-1, -1, ite.size(-1))
    ite_all = (ite * mask_expanded).sum(dim=(0, 1)) / mask_expanded.sum(dim=(0, 1))
    ite_samples = (ite * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

    return ite_all,ite_samples

def transfer_data_mimic(df,task):
    if task=='vaso':
        x=torch.cat((df[0], df[1][:,:,2:5], df[1][:,:,5:-1]), dim=2)
        treatment=df[1][:,:,:2]
        outcome=df[1][:,:,5]
        activate=df[3]
    elif task=='mechvent':
        x=torch.cat((df[0], df[1][:,:,2:-7], df[1][:,:,-7:-1]), dim=2)
        treatment=df[1][:,:,:2]
        outcome=df[1][:,:,-7]
        activate=df[3]
    else:print('No such task')
    return x,treatment,outcome.unsqueeze(2),activate
def calculate_error_ate_per_sample(factual_outcomes, cf_outcomes_t1, cf_outcomes_t2, cf_outcomes_t3, ite_pred):
    """
    Calculate errorATE (Error in Average Treatment Effect) for each sample and time step, keeping shape [5000, 30, 3].
    
    Parameters:
    factual_outcomes: torch.Tensor 
        The observed (factual) outcomes, shape [5000, 30, 1].
    cf_outcomes_t1: torch.Tensor 
        Counterfactual outcomes for treatment 1, shape [5000, 30, 1].
    cf_outcomes_t2: torch.Tensor 
        Counterfactual outcomes for treatment 2, shape [5000, 30, 1].
    cf_outcomes_t3: torch.Tensor 
        Counterfactual outcomes for treatment 3, shape [5000, 30, 1].
    ite_pred: torch.Tensor 
        The predicted ITE, shape [5000, 30, 3].
    
    Returns:
    error_ate_per_sample: torch.Tensor 
        The error in ATE for each sample, time step, and treatment, shape [5000, 30, 3].
    """
    
    # 1. Calculate the true ITE for each treatment at each sample and time step
    true_ite_t1 = cf_outcomes_t1 - factual_outcomes  # ITE for treatment 1
    true_ite_t2 = cf_outcomes_t2 - factual_outcomes  # ITE for treatment 2
    true_ite_t3 = cf_outcomes_t3 - factual_outcomes  # ITE for treatment 3

    # 2. Combine the true ITE for all treatments into a tensor of shape [5000, 30, 3]
    true_ite = torch.cat([true_ite_t1, true_ite_t2, true_ite_t3], dim=-1)  # Shape: [5000, 30, 3]

    # 3. Compute the absolute error in ATE for each sample, time step, and treatment
    error_ate_per_sample = torch.abs(true_ite - ite_pred)  # Shape: [5000, 30, 3]

    # 4. Return the per-sample, per-time-step errorATE
    return error_ate_per_sample

def calculate_pehe(factual_outcomes, cf_outcomes_t1, cf_outcomes_t2, cf_outcomes_t3, ite_pred):
    """
    Calculate PEHE (Precision in Estimation of Heterogeneous Effect) for each sample, time step, and treatment.
    
    Parameters:
    factual_outcomes: torch.Tensor 
        The observed (factual) outcomes, shape [5000, 30, 1].
    cf_outcomes_t1: torch.Tensor 
        Counterfactual outcomes for treatment 1, shape [5000, 30, 1].
    cf_outcomes_t2: torch.Tensor 
        Counterfactual outcomes for treatment 2, shape [5000, 30, 1].
    cf_outcomes_t3: torch.Tensor 
        Counterfactual outcomes for treatment 3, shape [5000, 30, 1].
    ite_pred: torch.Tensor 
        The predicted ITE, shape [5000, 30, 3].
    
    Returns:
    pehe_per_sample: torch.Tensor 
        The PEHE value per sample, time step, and treatment, shape [5000, 30, 3].
    """
    
    # 1. Calculate the true ITE for each treatment
    ite_true_t1 = cf_outcomes_t1 - factual_outcomes  # ITE for treatment 1
    ite_true_t2 = cf_outcomes_t2 - factual_outcomes  # ITE for treatment 2
    ite_true_t3 = cf_outcomes_t3 - factual_outcomes  # ITE for treatment 3
    
    # 2. Combine the true ITE for all treatments into a tensor of shape [5000, 30, 3]
    ite_true = torch.cat([ite_true_t1, ite_true_t2, ite_true_t3], dim=-1)  # Shape: [5000, 30, 3]
    
    # 3. Compute the squared difference between true ITE and predicted ITE
    pehe_per_sample = (ite_true - ite_pred) ** 2  # Shape: [5000, 30, 3]
    
    # 4. Return the per-sample, per-time-step, per-treatment PEHE
    return pehe_per_sample,ite_true

def evaluate_syn(model,dataset_name, test_loader):
    model.eval()
    model.cuda()
    all_predicted_outcomes = []
    all_true_outcomes = []
    all_masks = []
    all_ite_ori=[]
    length=[]
    all_cf1=[]
    all_cf2=[]
    all_cf3=[]
    with torch.no_grad():
        for x, delta_t, treatment, y, activate,cf1,cf2,cf3 in test_loader:
            x, delta_t, treatment, y, activate,cf1,cf2,cf3 = Variable(x).cuda(), Variable(delta_t).cuda(), Variable(treatment).cuda(), Variable(y).cuda(), Variable(activate).cuda(), Variable(cf1).cuda(), Variable(cf2).cuda(), Variable(cf3).cuda()
            
            x_recon, mu, logvar, treatment_pred,outcome_pred_wtreat = model(x, delta_t,treatment,activate)
            #print(outcome_pred_wtreat.shape)# batch length 1            
            ite_sub_orginal= ite_calculate(model,x, delta_t,treatment,activate,dataset_name) 
            
            
            #ite
            # Create mask based on activate
            batch_size = x.size(0)
            sequence_length = x.size(1)
            mask = torch.arange(sequence_length).expand(batch_size, sequence_length).cuda() < activate.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()  # Expand mask to match outcome dimensions

            all_predicted_outcomes.append(outcome_pred_wtreat)
            all_true_outcomes.append(y)
            all_masks.append(mask)
            all_ite_ori.append(ite_sub_orginal) 
            length.append(activate)
            all_cf3.append(cf3)
            all_cf2.append(cf2)
            all_cf1.append(cf1)
    # Concatenate all outcomes and masks
    all_predicted_outcomes = torch.cat(all_predicted_outcomes, dim=0)
    all_true_outcomes = torch.cat(all_true_outcomes, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_ite_ori = torch.cat(all_ite_ori, dim=0)
    all_cf3 = torch.cat(all_cf3, dim=0)
    all_cf2 = torch.cat(all_cf2, dim=0)
    all_cf1 = torch.cat(all_cf1, dim=0)
    length = torch.cat(length, dim=0)
    # Compute RMSE and MAE
    rmse_val = rmse(all_predicted_outcomes, all_true_outcomes, all_masks)
    mae_val = mae(all_predicted_outcomes, all_true_outcomes, all_masks)
    ite_val_ori,ite_samples_ori = ite_mask(all_ite_ori, all_masks)
    #pehe
    pehe_per_sample,ite_true_sample=calculate_pehe(all_true_outcomes, all_cf1, all_cf2, all_cf3, all_ite_ori)
    pehe_per,pehe_per_person= ite_mask(pehe_per_sample, all_masks)
    print(f"PEHE original: {pehe_per}")
    print(f"root of PEHE original: {torch.sqrt(abs(pehe_per))}")
    error_ate_per_sample=calculate_error_ate_per_sample(all_predicted_outcomes, all_cf1, all_cf2, all_cf3, all_ite_ori)
    eate,eate_person= ite_mask(error_ate_per_sample, all_masks)
    print(f"eATE original: {eate}")
    
    ite_true,ite_true_per_person= ite_mask(ite_true_sample, all_masks)
    print(f"true ITE original: {ite_true}")
    print(f"pred ITE original: {ite_val_ori}") 
    #
    print(f"RMSE: {rmse_val}")
    print(f"MAE: {mae_val}")

    
    return all_predicted_outcomes,all_true_outcomes,all_masks,ite_samples_ori