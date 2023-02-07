import torch 
import torch.nn as nn
import ipdb
import torch.autograd.functional as F_autograd

torch.manual_seed(42)

class EncoderObservation(nn.Module):
    def __init__(self, device, B, in_channels, height, width, state_dim=256) -> None:
        super().__init__()
        '''
        encoder maps last UNet observation to Z_t and L_t
        '''
        self.device = device
        self.B = B
        self.state_dim = state_dim
        mid_channels = in_channels//2

        # mapping last layer UNet to small latent state
        self.embedding_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3,padding=1),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(mid_channels),
                                            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1),
                                            nn.ReLU()
                                            )
        self.embedding_z = nn.Sequential(
                                        nn.Linear(int(height*width), state_dim),
                                        nn.ReLU()
                                        )
        # self.embedding_L = nn.Sequential(nn.ReLU(),
        #                                nn.Linear(int(height*width), state_dim*state_dim))

    def forward(self, x):
        '''
        x -> last encoded state of UNet
        '''              
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x_conv = self.embedding_conv(x)        
        x_conv = x_conv.view(B*T, -1)
        x_emb_z = self.embedding_z(x_conv)
        # x_emb_L = self.embedding_L(x_conv)

        x_emb_z = x_emb_z.view(B, T, self.state_dim)
        # x_emb_L = x_emb_L.view(B, T, int(self.state_dim*self.state_dim))
        return x_emb_z #, x_emb_L

class DecoderObservation(nn.Module):
    def __init__(self, device, B, out_channels, height, width, state_dim=256) -> None:
        super().__init__()
        '''
        '''
        self.device = device
        self.B = B 
        self.state_dim = state_dim
        self.out_channels = out_channels
        self.height = height 
        self.width = width
        mid_channels = out_channels//2

        self.embedding = nn.Sequential(nn.Linear(state_dim, int(height*width)),
                                       nn.ReLU())
        self.embedding_conv_trans = nn.Sequential(nn.ConvTranspose2d(1, mid_channels, kernel_size=3, stride=1, padding=1),
                                                  nn.BatchNorm2d(mid_channels),
                                                  nn.ReLU(),
                                                  nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                                  nn.ReLU()
                                                )                                    

    def forward(self, z):
        '''
        z -> latent state
        '''
        T = z.shape[1]
        z = z.view(-1, self.state_dim)
        z_emb = self.embedding(z)
        z_emb = z_emb.view(self.B*T, 1, self.height, self.width)        
        z_conv = self.embedding_conv_trans(z_emb)
        z_conv = z_conv.view(self.B, T, self.out_channels, self.height, self.width)
        return z_conv

class KalmanModel(nn.Module):

    def __init__(self, device, B, in_channels, height, width, state_dim=256):
        '''
        Kalman Filter model class
        Implementation of backprop KF paper: https://arxiv.org/pdf/1605.07148.pdf

        @param device
        @param B: Batch Size
        @param in_channels
        @param state_dim: Size of Latent state (flattened)
        '''

        super().__init__()

        self.device = device
        self.B = B
        self.in_channels = in_channels        
        self.state_dim = state_dim

        # mapping last layer UNet to state_dim
        self.encoder = EncoderObservation(device, B, in_channels, height, width, state_dim)

        # mapping state_dim to last_layer UNet
        self.decoder = DecoderObservation(device, B, in_channels, height, width, state_dim)
        
        # Initialize the model parameters
        self.Cz = torch.eye(self.state_dim).to(device)

        self.Cy = torch.eye(self.state_dim).to(device)

        self.A = nn.Parameter(torch.eye(self.state_dim), requires_grad=True).to(device)
        # self.A = torch.eye(self.state_dim).to(device)

        # self.diag_Q = nn.Parameter(torch.rand(self.state_dim), requires_grad=True, device = device)
        self.diag_Q = torch.eye(self.state_dim).to(device)
        
        # declare initial Mean: (B, state_dim), Covar (B, state_dim, state_dim) 
        self.state_mean = torch.zeros(B, state_dim, device=device) 
        self.state_covar = 10.0*torch.eye(state_dim, device=device).repeat(self.B, 1, 1)

        #push to cuda
        # self.encoder.to('cuda')
        # self.decoder.to('cuda')
        
    def forward(self, x):
        '''
        Forward pass of the Kalman Filter model
        @init_mean: Initial flattened Unet last layer output
        @param l_prev: covariance Cholesky input at all times 
        @param: z_t: measurements array for the whole trajectory
        @returns: y_outs: mean outputs for all times 
        '''
        traj_len = x.size(1)
        z, L = self.encoder(x)  

        # declare initial Mean: (B, state_dim), Covar (B, state_dim, state_dim)
        self.state_mean = torch.zeros(self.B, self.state_dim, device=self.device) 
        self.state_covar = 10.0*torch.eye(self.state_dim, device=self.device).repeat(self.B, 1, 1)

        mean_list = [self.state_mean]
        covar_list = [self.state_covar]

        z_estimate_list = []
        for t in range(traj_len):
            # print("t=" , t)
            z_t = z[:,t] # B x state_dim 
            L_t = L[:,t] # B x state_dim x state_dim 
            
            # update state_mean and state_covar using dynamics 
            state_mean_est = mean_list[-1] @ self.A.T # (B x state_dim) @ (state_dim x state_dim).T
            state_covar_est = self.A @ covar_list[-1] @ self.A.T + self.diag_Q.repeat(self.B, 1, 1)

            # find innovation error and covariance                         
            z_tilde_t = z_t - state_mean_est @ self.Cz.T # batch x state_dim
            R_t = self.cholesky_decomposition(L_t)
            # z_covar_t = self.Cz @ state_covar_est @ self.Cz.T + R_t # batch x state_dim x state_dim
            z_covar_t = state_covar_est  + R_t # batch x state_dim x state_dim
            
            # find Kalman gain 
            K_t = state_covar_est @ self.Cz.T @ torch.inverse(z_covar_t) # batch x state_dim x state_dim 

            # update state and covar estimate 
            state_mean_updated = state_mean_est + (K_t @ z_tilde_t.unsqueeze(-1)).squeeze(-1) #             
            I_kh = ((torch.eye(self.state_dim)).repeat(self.B,1,1).to(self.device) - K_t @ self.Cz)
            state_covar_updated = I_kh @ state_covar_est @ I_kh.transpose(-1,-2)
            
            # update state and covar so that for next time step most updated state and covar used 
            self.state_mean = state_mean_updated
            self.state_covar = state_covar_updated
            # Bookkeeping
            mean_list.append(self.state_mean)
            covar_list.append(self.state_covar)            

        # pass state_mean through decoder to get updated UNet states
        mean_list.pop(0)
        covar_list.pop(0)

        mean_list = torch.stack(mean_list, dim = 1) # batch x T x state_dim
        x_estimate = self.decoder(mean_list)
        
        return x_estimate

    def cholesky_decomposition(self, l_t):
        '''
        @param: l_t matrix (B*state_dim**2) consists of elements
        @return: Noise mat R: (B, state_dim, state_dim)
        '''
        l_mat = l_t.view(self.B, self.state_dim, self.state_dim)
        lower_triangular = torch.tril(l_mat)
        R = torch.matmul(lower_triangular.transpose(-2,-1), lower_triangular)
        return R


class DynamicsModel(nn.Module):
    def __init__(self, state_dim) -> None:
        super().__init__()

        self.dynamics_model = nn.Sequential(nn.Linear(state_dim, state_dim),
                                            nn.ReLU(),
                                            nn.Linear(state_dim, state_dim))

    def forward(self, x):

        return self.dynamics_model(x)

    def jacobian(self, x):

        return F_autograd.jacobian(self.dynamics_model, x) #reate_graph=True

    def grad(self, x):
        return torch.autograd.grad(self.dynamics_model, x, is_grads_batched=True)

class ExtendedKalmanModel(KalmanModel):

    def __init__(self, device, B, in_channels, height, width, state_dim=256) -> None:
        super().__init__(device, B, in_channels, height, width, state_dim)

        self.dynamics_model = DynamicsModel(state_dim)
        self.A = None

    def forward(self, x):
        '''
        Forward pass of the Kalman Filter model
        @init_mean: Initial flattened Unet last layer output
        @param l_prev: covariance Cholesky input at all times 
        @param: z_t: measurements array for the whole trajectory
        @returns: y_outs: mean outputs for all times 
        '''
        traj_len = x.size(1)
        z, L = self.encoder(x)  

        # declare initial Mean: (B, state_dim), Covar (B, state_dim, state_dim) 
        self.state_mean = torch.zeros(self.B, self.state_dim, device=self.device) 
        self.state_covar = 10.0*torch.eye(self.state_dim, device=self.device).repeat(self.B, 1, 1)

        mean_list = [self.state_mean]
        covar_list = [self.state_covar]

        z_estimate_list = []
        for t in range(traj_len):
            # print("t=" , t)
            z_t = z[:,t] # B x state_dim 
            L_t = L[:,t] # B x state_dim x state_dim 
            
            # update state_mean and state_covar using dynamics 
            state_mean_est = self.dynamics_model(mean_list[-1]) # (B x state_dim) @ (state_dim x state_dim).T
            # find jacobian of dynamics
            self.A = self.dynamics_model.jacobian(mean_list[-1]).squeeze(-2).squeeze(0) 
            
            state_covar_est = self.A @ covar_list[-1] @ self.A.T + self.diag_Q.repeat(self.B, 1, 1)

            # find innovation error and covariance                         
            z_tilde_t = z_t - state_mean_est @ self.Cz.T # batch x state_dim
            R_t = self.cholesky_decomposition(L_t)
            # z_covar_t = self.Cz @ state_covar_est @ self.Cz.T + R_t # batch x state_dim x state_dim
            z_covar_t = state_covar_est  + R_t # batch x state_dim x state_dim
            
            # find Kalman gain 
            K_t = state_covar_est @ self.Cz.T @ torch.inverse(z_covar_t) # batch x state_dim x state_dim 

            # update state and covar estimate 
            state_mean_updated = state_mean_est + (K_t @ z_tilde_t.unsqueeze(-1)).squeeze(-1) #             
            I_kh = ((torch.eye(self.state_dim)).repeat(self.B,1,1).to(self.device) - K_t @ self.Cz)
            state_covar_updated = I_kh @ state_covar_est @ I_kh.transpose(-1,-2)
            
            # update state and covar so that for next time step most updated state and covar used 
            self.state_mean = state_mean_updated
            self.state_covar = state_covar_updated
            # Bookkeeping
            mean_list.append(self.state_mean)
            covar_list.append(self.state_covar)            

        # pass state_mean through decoder to get updated UNet states
        mean_list.pop(0)
        covar_list.pop(0)

        mean_list = torch.stack(mean_list, dim = 1) # batch x T x state_dim
        x_estimate = self.decoder(mean_list)
        
        return x_estimate    

class KalmanNet(KalmanModel):

    def __init__(self, device, B, in_channels, height, width, state_dim=256):
        super().__init__(device, B, in_channels, height, width, state_dim)

        self.dynamics_model = nn.Sequential(nn.Linear(state_dim, state_dim),
                                            nn.ReLU(),
                                            nn.Linear(state_dim, state_dim),
                                            nn.ReLU()
                                            )

        self.observation_model = nn.Sequential(nn.Linear(state_dim, state_dim),
                                                nn.ReLU(),
                                                nn.Linear(state_dim, state_dim),
                                                nn.ReLU()
                                                )

        # concatenate state error and observation error and project to RNN state size
        self.RNN_state = nn.Sequential(nn.Linear(2*state_dim, state_dim), 
                                       nn.ReLU())

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=state_dim, batch_first=True)        

        # project RNN state to kalman_gain matrix size 
        self.kalman_gain = nn.Sequential(nn.Linear(state_dim, int(state_dim*state_dim)),
                                         nn.ReLU()
                                        )


    def forward(self, x):
        '''
        Forward pass of the Kalman Filter model
        @init_mean: Initial flattened Unet last layer output
        @param l_prev: covariance Cholesky input at all times 
        @param: z_t: measurements array for the whole trajectory
        @returns: y_outs: mean outputs for all times 
        '''
        traj_len = x.size(1)
        z = self.encoder(x)  

        # declare initial Mean: (B, state_dim), Covar (B, state_dim, state_dim) 
        self.state_mean = z[:,0] #torch.zeros(self.B, self.state_dim, device=self.device) 
        # self.state_covar = 10.0*torch.eye(self.state_dim, device=self.device).repeat(self.B, 1, 1)

        mean_list = [self.state_mean]
        # covar_list = [self.state_covar]

        state_error_prev = self.state_mean #torch.zeros(self.B, self.state_dim, device=self.device)        
        
        # RNN h, c init 0
        h = torch.zeros(self.B, self.state_dim, device=self.device)
        c = torch.zeros(self.B, self.state_dim, device=self.device)

        for t in range(1,traj_len):
            # print("t=" , t)
            z_t = z[:,t] # B x state_dim 
            
            # update state_mean and state_covar using dynamics 
            state_mean_est = self.dynamics_model(mean_list[-1]) # (B x state_dim) @ (state_dim x state_dim).T                            

            # find innovation error and covariance                         
            z_tilde_t = z_t - self.observation_model(state_mean_est) # batch x state_dim
                        
            # find Kalman gain 
            rnn_input = torch.cat([z_tilde_t, state_error_prev], dim=-1)
            rnn_state = self.RNN_state(rnn_input)
            out, (h,c) = self.lstm(rnn_state, (h, c)) # batch x state_dim x state_dim 
            K_t = self.kalman_gain(out)
            K_t = K_t.view(self.B, self.state_dim, self.state_dim)
            
            # update state and covar estimate 
            state_mean_updated = state_mean_est + (K_t @ z_tilde_t.unsqueeze(-1)).squeeze(-1) #             
            
            # find state error
            state_error_prev = state_mean_updated - state_mean_est
            # update state and covar so that for next time step most updated state and covar used 
            self.state_mean = state_mean_updated
            
            # Bookkeeping
            mean_list.append(self.state_mean)
            # covar_list.append(self.state_covar)            

        # pass state_mean through decoder to get updated UNet states
        # mean_list.pop(0)
        # covar_list.pop(0)

        mean_list = torch.stack(mean_list, dim = 1) # batch x T x state_dim
        x_estimate = self.decoder(mean_list)
        
        return x_estimate    
