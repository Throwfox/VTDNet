import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import StepLR
from utils.vtd_utils import rmse,mae
import math

class AttentionMechanism(nn.Module):
    def __init__(self, d_model, d_treatment):
        super(AttentionMechanism, self).__init__()
        # Define the linear layers for query, key, and value projections
        self.query_layer = nn.Linear(d_model, d_model)      # Projects z into query space
        self.key_layer = nn.Linear(d_treatment, d_model)    # Projects t into key space
        self.value_layer = nn.Linear(d_treatment, d_model)  # Projects t into value space
        
        self.scale_factor = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
    
    def forward(self, z, t):
        # z is [sequence, batch, 256], t is [sequence, batch, 3]
        # Project z (query), t (key and value) using their respective layers
        query = self.query_layer(z)              # Shape: [sequence, batch, 256]
        key = self.key_layer(t)                  # Shape: [sequence, batch, 256]
        value = self.value_layer(t)              # Shape: [sequence, batch, 256]
        
        # Compute attention scores: query (z) dot product with key (t), scaled by the dimension
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale_factor  # Shape: [sequence, batch, batch]
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # Shape: [sequence, batch, batch]
        
        # Compute the final weighted value
        attention_output = torch.matmul(attention_weights, value)  # Shape: [sequence, batch, 256]
        
        # Add the attention output back to z
        z_with_attention = z + attention_output  # Shape: [sequence, batch, 256]
        
        return z_with_attention, attention_weights
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class VTD(nn.Module):
    '''
    if samples size: 5000; max length:30; num of covariances:30; num of treatments:3
    x:covariates (5000, 30, 30)
    t:treatments (5000, 30, 3)
    activate:sequence_length (5000,)
    y:factual_outcomes (5000, 30, 1)
    cf_outcomes_t1 (5000, 30, 1)
    cf_outcomes_t2 (5000, 30, 1)
    cf_outcomes_t3 (5000, 30, 1)
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim, treatment_dim,head,length):
        super(VTD, self).__init__()
        # Embedding layer to transform input_dim to hidden_dim
        self.latent_dim=latent_dim
        self.embedding = nn.Sequential(
                        nn.Linear(input_dim+treatment_dim,hidden_dim),
                        #nn.Embedding(num_embeddings=input_dim, embedding_dim=hidden_dim),
                        PositionalEncoding(d_model=hidden_dim, dropout=0.1, max_len=length)
                        )
        # Encoder network
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=head)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.treatment_dim=treatment_dim
        self.fc_enc = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.bn_enc = nn.BatchNorm1d(hidden_dim * 2)##for rapid converge
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) ##avoid over fitting
        self.Softplus = nn.Softplus()
        
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)  # Mapping hidden states (h_t and h_t-1) to μ
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)  # Mapping hidden states (h_t and h_t-1) to log(Σ)
        self.expand_t=nn.Linear(treatment_dim, latent_dim)
        # Decoder network
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=4)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.fc_de = nn.Linear(hidden_dim, latent_dim)
        self.fc_reconstruct = nn.Linear(latent_dim, input_dim)
        
        # ITE estimation block
        self.fc_treatment = nn.Linear(latent_dim+treatment_dim, treatment_dim)
        self.fc_outcome = nn.Sequential(
                        nn.Linear(latent_dim, latent_dim//2),
                        #nn.BatchNorm1d(latent_dim + treatment_dim//2),  #n c l
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(latent_dim //2, output_dim),   
                                    )
        #self.fc_outcome_simple=nn.Linear(latent_dim, output_dim)
    def encode(self, x, delta_t,t_true,varing_length):
        
        zero_state_t = torch.zeros(t_true.size(0), 1, t_true.size(2), device=t_true.device)
        t_true_minus_1 = torch.cat((zero_state_t, t_true[:,:-1,:]), dim=1) 
        
        x=torch.cat((x,t_true_minus_1),dim=2)
        
        x = x.transpose(0, 1)
        
        x = self.embedding(x)
        
          # Transpose to [sequence_length, batch_size, input_dim]
        #mask for attentsion
        batch_size = x.size(1)
        sequence_length = x.size(0)

    # Create masks based on the activate tensor
        padding_mask = torch.arange(sequence_length).expand(batch_size, sequence_length).cuda() >= varing_length.unsqueeze(1)
    
        delta_t=delta_t.transpose(0, 1) # Transpose to [sequence_length, batch_size, input_dim]

        hidden_states = self.encoder(x,src_key_padding_mask=padding_mask)  # Encoder output shape: [sequence_length, batch_size, hidden_dim]
        
        # Prepare h_t and h_t-1 for each time step
        zero_state = torch.zeros(1, hidden_states.size(1), hidden_states.size(2), device=hidden_states.device)
        h_t_minus_1 = torch.cat([zero_state, hidden_states[:-1]], dim=0)  # Add zero state at the beginning
        delta_t_expanded=delta_t.expand_as(h_t_minus_1)
        #print(delta_t_expanded.shape,h_t_minus_1.shape)
        assert delta_t_expanded.shape == h_t_minus_1.shape, "Shape mismatch between delta_t_expanded and h_t_minus_1"
        h_t_minus_1 = h_t_minus_1 * torch.exp(-delta_t_expanded)
        
        # Concatenate h_t and h_t-1
        h_concat = torch.cat([hidden_states, h_t_minus_1], dim=-1)  # Shape: [sequence_length, batch_size, hidden_dim*2]

        # Flatten the concatenated hidden states for each timestep
        h_concat_flat = h_concat.reshape(-1, h_concat.size(-1))  # Shape: [sequence_length * batch_size, hidden_dim*2]
        
        # Fusion
        h_concat_flat = self.fc_enc(h_concat_flat)
        h_concat_flat = self.bn_enc(h_concat_flat) 
        h_concat_flat = self.relu(h_concat_flat)
        h_concat_flat = self.dropout(h_concat_flat) ## add dropbox
        
        # Map to latent parameters μ and log(Σ)
        mu = self.fc_mu(h_concat_flat)  # Shape: [sequence_length * batch_size, latent_dim]
        logvar = self.fc_logvar(h_concat_flat)  # Shape: [sequence_length * batch_size, latent_dim]
        logvar = self.Softplus(logvar)
        
        # Reshape back to [sequence_length, batch_size, latent_dim]
        mu = mu.reshape(hidden_states.size(0), hidden_states.size(1), -1)  # Shape: [sequence_length, batch_size, latent_dim]
        logvar = logvar.reshape(hidden_states.size(0), hidden_states.size(1), -1)  # Shape: [sequence_length, batch_size, latent_dim]
        
        return mu, logvar, hidden_states
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, memory):
        hidden_states = self.decoder(z, memory)  # Decoder output shape: [sequence_length, batch_size, latent_dim]
        hidden_states = self.fc_de(hidden_states)
        hidden_states = self.relu(hidden_states)
        
        x_recon = self.fc_reconstruct(hidden_states)  # Shape: [sequence_length, batch_size, input_dim]
        return x_recon.transpose(0, 1)  # Transpose back to [batch_size, sequence_length, input_dim]

    def forward(self, x, delta_t,t_true,varing_length):
        mu, logvar, memory = self.encode(x, delta_t,t_true,varing_length)
        z = self.reparameterize(mu, logvar) 
        x_recon = self.decode(z, memory)
        
        
        zero_state_t = torch.zeros(t_true.size(0), 1, t_true.size(2), device=t_true.device)
        t_true_minus_1 = torch.cat((zero_state_t, t_true[:,:-1,:]), dim=1)
        t_true_minus_1 = t_true_minus_1.transpose(0, 1)
        z_t=torch.cat((z,t_true_minus_1),dim=2)
        treatment_pred = torch.sigmoid(self.fc_treatment(z_t))
        
        #t_pred_expand=self.expand_t(treatment_pred)
        #t_true_expand=self.expand_t(t_true.transpose(0, 1))
        
        #if =='atten'
        attention = AttentionMechanism(d_model=self.latent_dim, d_treatment=self.treatment_dim)
        attention.cuda()
        
        z_with_truet, attn_weights = attention(z, t_true.transpose(0, 1))
        z_with_predt, attn_weights = attention(z, treatment_pred)
        
        #if mode_train:
        #    if np.random.rand()>0.5:
        #        outcome_pred_wtreat = self.fc_outcome(z+t_true_expand)
        #    else:
        #        outcome_pred_wtreat = self.fc_outcome(z+t_pred_expand)
        #elif not mode_train:         
            #outcome_pred_wtreat = self.fc_outcome(z+t_true_expand)

        
        #print(torch.cat((z, treatment_pred), dim=-1).shape) #[21, 64, 130] #ncl
        #------------------------------------------------------
        outcome_pred = self.fc_outcome(z_with_predt)
        outcome_pred_wtreat = self.fc_outcome(z_with_truet)
        #------------------------------------------------------
        treatment_pred = treatment_pred.transpose(0, 1) # [batch, seq, treatment_dim]
        outcome_pred = outcome_pred.transpose(0, 1) # [batch, seq, output_dim]
        outcome_pred_wtreat = outcome_pred_wtreat.transpose(0, 1) # [batch, seq, output_dim]
        
        return x_recon, mu, logvar, treatment_pred, outcome_pred, outcome_pred_wtreat
    
def loss_function(x, x_recon, mu, logvar, treatment, treatment_pred, outcome, outcome_pred, outcome_pred_wtreat,varing_length,alpha):
    batch_size = x.size(0)
    sequence_length = x.size(1)
    num_treat=treatment.size(2)
    # Create masks based on the activate tensor
    mask = torch.arange(sequence_length).expand(batch_size, sequence_length).cuda() < varing_length.unsqueeze(1)
    
    # Expand mask to match dimensions of x and x_recon
    mask_expanded = mask.unsqueeze(-1).expand_as(x).float()

    # Compute recon_loss only for activated parts
    recon_loss = nn.MSELoss(reduction='none')(x_recon, x)
    recon_loss = (recon_loss * mask_expanded).sum() / mask_expanded.sum()

    # Compute kl_loss only for activated parts
    mu = mu.transpose(0, 1)  # [batch, sequence_length, latent_dim]
    logvar = logvar.transpose(0, 1)  # [batch, sequence_length, latent_dim]
    mask_kl = mask.unsqueeze(-1).expand_as(mu).float()
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = (kl_loss * mask_kl).sum() / mask_kl.sum()

    # Compute treatment_loss only for activated parts
    mask_treatment = mask.unsqueeze(-1).expand_as(treatment).float()
    treatment_loss = nn.BCELoss(reduction='none')(treatment_pred, treatment) #BCELoss
    treatment_loss = (treatment_loss * mask_treatment).sum() / mask_treatment.sum()
    #print(treatment.shape) [64,21,2]
    
    # Compute outcome_loss only for activated parts
    mask_outcome = mask.unsqueeze(-1).expand_as(outcome).float()
    outcome_loss = nn.MSELoss(reduction='none')(outcome_pred_wtreat, outcome) #!!!!!!!!!!!!!
    outcome_loss_ori = (outcome_loss * mask_outcome).sum() / mask_outcome.sum()
    
    #iptw-loss
    iptw_weights = torch.ones_like(treatment)

    for i in range(treatment.shape[2]):
        iptw_weights[:, :, i] = torch.where(treatment[:, :, i] == 1, 
                                            1.0 / treatment_pred[:, :, i], 
                                            1.0 / (1.0 - treatment_pred[:, :, i]*(1-1e-5)))
    iptw_losses=0
    for i in range(treatment.shape[2]):
        iptw_losses += outcome_loss * iptw_weights[:, :, i].unsqueeze(-1)
    iptw_losses = (iptw_losses * mask_outcome).sum() / mask_outcome.sum()
    # Sum all losses
    total_loss = recon_loss + kl_loss + treatment_loss + (0.1*alpha*iptw_losses )/num_treat+alpha*outcome_loss_ori
    #check if nan
    if torch.isnan(total_loss).any():
        print("Loss contains NaN values")
    return total_loss, recon_loss, kl_loss, treatment_loss, outcome_loss_ori,iptw_losses

def trainer(model, train_loader, val_loader,test_loader, optimizer, num_epochs,alpha, model_save_path,resume):
    print('model training....',resume)
    if resume==1:
        model.load_state_dict(torch.load(model_save_path))
        print('load model and continous to train..')
        # Validate
        model.eval()
        val_loss = 0
        all_predicted_outcomes = []
        all_true_outcomes = []
        all_masks = []
        with torch.no_grad():
            for x, delta_t, treatment, outcome,varing_length in tqdm(val_loader):
                x, delta_t, treatment, outcome,varing_length = Variable(x).cuda(), Variable(delta_t).cuda(), Variable(treatment).cuda(), Variable(outcome).cuda(),Variable(varing_length).cuda()
                x_recon, mu, logvar, treatment_pred, outcome_pred,outcome_pred_wtreat = model(x, delta_t,treatment,varing_length)
                loss, recon_loss ,kl_loss , treatment_loss , outcome_loss,iptw_losses= loss_function(x, x_recon, mu, logvar, treatment, treatment_pred, outcome, outcome_pred,outcome_pred_wtreat,varing_length,alpha)
                val_loss += loss.item()
                # Create mask based on activate
                batch_size = x.size(0)
                sequence_length = x.size(1)
                mask = torch.arange(sequence_length).expand(batch_size, sequence_length).cuda() < varing_length.unsqueeze(1)
                mask = mask.unsqueeze(-1).float()  # Expand mask to match outcome dimensions
                all_predicted_outcomes.append(outcome_pred_wtreat)
                all_true_outcomes.append(outcome)
                all_masks.append(mask)  
        # Concatenate all outcomes and masks
        all_predicted_outcomes = torch.cat(all_predicted_outcomes, dim=0)
        all_true_outcomes = torch.cat(all_true_outcomes, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        # Compute RMSE and MAE
        rmse_val = rmse(all_predicted_outcomes, all_true_outcomes, all_masks)
        mae_val = mae(all_predicted_outcomes, all_true_outcomes, all_masks)  
        print(f"Val rmse: {rmse_val}, Val mae: {mae_val}")   
        best_val_mae = mae_val 
    elif resume==0:
        best_val_mae = float('inf')
        
    opt_scheduler = StepLR(optimizer, step_size=100, gamma=0.8)     
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        
        for x, delta_t, treatment, outcome, varing_length in tqdm(train_loader):
            x, delta_t, treatment, outcome,varing_length = Variable(x).cuda(), Variable(delta_t).cuda(), Variable(treatment).cuda(), Variable(outcome).cuda(),Variable(varing_length).cuda()
            
            optimizer.zero_grad()
            x_recon, mu, logvar, treatment_pred, outcome_pred,outcome_pred_wtreat = model(x, delta_t,treatment,varing_length)
            loss, recon_loss ,kl_loss , treatment_loss , outcome_loss,iptw_losses= loss_function(x, x_recon, mu, logvar, treatment, treatment_pred, outcome, outcome_pred,outcome_pred_wtreat,varing_length,alpha)
            loss.backward()
            #---------------------------------
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            #---------------------------------
            optimizer.step()
            total_loss += loss.item()
            
            #wandb log
            wandb.log({'training/loss': loss.cpu().detach().numpy(),
          'training/recon_loss': recon_loss.cpu().detach().numpy(),
          'training/kl_loss': kl_loss.cpu().detach().numpy(),
          'training/treatment_loss': treatment_loss.cpu().detach().numpy(),
          'training/outcome_loss': outcome_loss.cpu().detach().numpy(),
          'training/iptw_losses': iptw_losses.cpu().detach().numpy()})       
        avg_train_loss = total_loss / len(train_loader)
        opt_scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0
        all_predicted_outcomes = []
        all_true_outcomes = []
        all_masks = []
        with torch.no_grad():
            for x, delta_t, treatment, outcome,varing_length in tqdm(val_loader):
                x, delta_t, treatment, outcome,varing_length = Variable(x).cuda(), Variable(delta_t).cuda(), Variable(treatment).cuda(), Variable(outcome).cuda(),Variable(varing_length).cuda()
                x_recon, mu, logvar, treatment_pred, outcome_pred,outcome_pred_wtreat = model(x, delta_t,treatment,varing_length)
                loss, recon_loss ,kl_loss , treatment_loss , outcome_loss,iptw_losses= loss_function(x, x_recon, mu, logvar, treatment, treatment_pred, outcome, outcome_pred,outcome_pred_wtreat,varing_length,alpha)
                val_loss += loss.item()
                
                #wandb
                wandb.log({'val/loss': loss.cpu().detach().numpy(),
          'val/recon_loss': recon_loss.cpu().detach().numpy(),
          'val/kl_loss': kl_loss.cpu().detach().numpy(),
          'val/treatment_loss': treatment_loss.cpu().detach().numpy(),
          'val/outcome_loss': outcome_loss.cpu().detach().numpy(),
          'val/iptw_losses': iptw_losses.cpu().detach().numpy()})
                
                # Create mask based on activate
                batch_size = x.size(0)
                sequence_length = x.size(1)
                mask = torch.arange(sequence_length).expand(batch_size, sequence_length).cuda() < varing_length.unsqueeze(1)
                mask = mask.unsqueeze(-1).float()  # Expand mask to match outcome dimensions
                all_predicted_outcomes.append(outcome_pred_wtreat)
                all_true_outcomes.append(outcome)
                all_masks.append(mask)  
        # Concatenate all outcomes and masks
        all_predicted_outcomes = torch.cat(all_predicted_outcomes, dim=0)
        all_true_outcomes = torch.cat(all_true_outcomes, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        # Compute RMSE and MAE
        rmse_val = rmse(all_predicted_outcomes, all_true_outcomes, all_masks)
        mae_val = mae(all_predicted_outcomes, all_true_outcomes, all_masks)
        
        wandb.log({'val/rmse': rmse_val.cpu().detach().numpy(),
          'val/mae': mae_val.cpu().detach().numpy()})       
          
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")
        print(f"Epoch {epoch + 1}, Val rmse: {rmse_val}, Val mae: {mae_val}")   
        
        # Save the model if the validation loss is the smallest
        if mae_val < best_val_mae:
            best_val_mae = mae_val
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch + 1} with validation mae {mae_val}")
            
        # testingset
        model.eval()
        all_predicted_outcomes = []
        all_true_outcomes = []
        all_masks = []
        with torch.no_grad():
            for x, delta_t, treatment, outcome,varing_length in tqdm(test_loader):
                x, delta_t, treatment, outcome,varing_length = Variable(x).cuda(), Variable(delta_t).cuda(), Variable(treatment).cuda(), Variable(outcome).cuda(),Variable(varing_length).cuda()
                x_recon, mu, logvar, treatment_pred, outcome_pred,outcome_pred_wtreat = model(x, delta_t,treatment,varing_length)
                
                # Create mask based on activate
                batch_size = x.size(0)
                sequence_length = x.size(1)
                mask = torch.arange(sequence_length).expand(batch_size, sequence_length).cuda() < varing_length.unsqueeze(1)
                mask = mask.unsqueeze(-1).float()  # Expand mask to match outcome dimensions
                all_predicted_outcomes.append(outcome_pred_wtreat)
                all_true_outcomes.append(outcome)
                all_masks.append(mask)  
        # Concatenate all outcomes and masks
        all_predicted_outcomes = torch.cat(all_predicted_outcomes, dim=0)
        all_true_outcomes = torch.cat(all_true_outcomes, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        # Compute RMSE and MAE
        rmse_test = rmse(all_predicted_outcomes, all_true_outcomes, all_masks)
        mae_test = mae(all_predicted_outcomes, all_true_outcomes, all_masks)
        
        wandb.log({'test/rmse': rmse_test.cpu().detach().numpy(),
          'test/mae': mae_test.cpu().detach().numpy()})
          
        print(f"Epoch {epoch + 1}, Test rmse: {rmse_test}, Test mae: {mae_test}")  
