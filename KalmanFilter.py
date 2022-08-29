import torch 
import torch.nn as nn

class KalmanModel(nn.Module):

    def __init__(self, device, B, state_dim):
        '''
        Kalman Filter model class
        Implementation of backprop KF paper: https://arxiv.org/pdf/1605.07148.pdf

        @param device
        @param B: Batch Size
        @param state_dim: Size of Latent state (flattened)
        '''

        super().__init__()

        self.device = device
        self.B = B
        self.state_dim = state_dim

        # Initialize the model parameters
        self.Cz = torch.eye(self.state_dim).to(device)

        self.Cy = torch.eye(self.state_dim).to(device)

        self.A = nn.Parameter(torch.eye(self.state_dim), requires_grad=True, device = device)

        # self.diag_Q = nn.Parameter(torch.rand(self.state_dim), requires_grad=True, device = device)
        self.diag_Q = torch.ones(self.state_dim).to(device)

    def forward(self, mean_0,l_t, z_t):
        '''
        Forward pass of the Kalman Filter model
        @init_mean: Initial flattened Unet last layer output
        @param l_prev: covariance Cholesky input at all times 
        @param: z_t: measurements array for the whole trajectory
        @returns: y_outs: mean outputs for all times 
        '''
        traj_len = z_t.size(0)

        #  Mean: (B, state_dim), Covar (B, state_dim, state_dim)
        mean_0.to(self.device)
        covar_0 = 10*torch.eye(self.state_dim).repeat(self.B, 1, 1)
        means = [mean_0]
        covars = [covar_0]

        y_outs = []

        for t in range(traj_len):
            z = z_t[t]
            l = l_t[t]
            mean_out, covar_out = self.kalman_update(means[-1], covars[-1], z, l)
            y_out = (self.Cy @ mean_out.unsqueeze(-1)).squeeze(-1)

            # Bookkeeping
            means.append(mean_out)
            covars.append(covar_out)
            y_outs.append(y_out) 

        
        return torch.stack(y_outs, 0)

    def mean_dynamics_update(self, mu_prev):
        '''
        Mean dynamics mu update function
        @param mu_prev: prev latent state mu at time t-1
        @return mu_t: predicted latent state mu at time t
        '''
        # mu_{x_t} = A mu_{x_prev}
        return self.A @ mu_prev.unsqueeze(-1)
    
    def covar_dynamics_update(self, covar_prev):
        '''
        Covariance dynamics covar update function
        @param covar_prev: prev latent state covar at time t-1
        @return covar_t: predicted latent state covar at time t
        '''
        # covar_{x_t} = A covar_{x_prev} A^T + Q
        return self.A @ covar_prev @ self.A.T + torch.diag(self.diag_Q)
    
    def kalman_gain(self, covar_pred, R, Cz):
        '''
        Kalman gain function
        @param covar_pred: predicted covariance matrix
        @param R: measurement covariance matrix
        @param Cz: measurement matrix
        @return K: Kalman gain matrix
        '''
        # K =  covar_{x_t} Cz^T (Cz covar_{x_t} Cz^T + R)^-1
        K = covar_pred @ Cz.transpose(-2,-1) @ torch.inverse(Cz @ covar_pred @ Cz.transpose(-2,-1) + R)
        return K

    def mean_observation_update(self, K, z_t, mu_pred, Cz):
        '''
        Mean observation update function
        @param K: Kalman gain matrix
        @param z_t: measurement for time t
        @param mu_pred: predicted latent state mu at time t
        @param Cz: measurement matrix
        @return mu_t: updated latent state mu at time t
        '''
        # mu_{x_t} = mu_{x_pred} + K (z_t - Cz mu_{x_pred})
        return (mu_pred + K @ (z_t.unsqueeze(-1) - Cz @ mu_pred)).squeeze(-1)

    def covar_observation_update(self, K, covar_pred, Cz):
        '''
        Covariance observation update function
        @param K: Kalman gain matrix
        @param covar_pred: predicted latent state covar at time t
        @param Cz: measurement matrix
        @return covar_t: updated latent state covar at time t
        '''
        # covar_{x_t} = (I - K Cz) covar_{x_pred}
        return (torch.eye(self.state_dim).to(self.device) - K @ Cz) @ covar_pred
    
    def cholesky_decomposition(self, l_t):
        '''
        @param: l_t matrix (B*state_dim**2) consists of elements
        @return: Noise mat R: (B, state_dim, state_dim)
        '''
        l_mat = l_t.view(self.B, self.state_dim, self.state_dim)
        lower_triangular = torch.tril(l_mat)
        R = torch.matmul(lower_triangular.transpose(-2,-1), lower_triangular)
        return R
    
    def kalman_update(self, mean_prev, covar_prev, z_t, l_t):
        '''
        Kalman update function
        @param mean_prev: prev latent state mu at time t-1
        @param covar_prev: prev latent state covar at time t-1
        @param z_t: measurement for time t
        @param l_t: covariance Cholesky input at time t
        @return mean_t: updated latent state mu at time t
        @return covar_t: updated latent state covar at time t
        '''
        # call kf update function for the next output value
        Cz = self.Cz.unsqueeze(0).repeat(self.B,1,1)

        mean_predicted = self.mean_dynamics_update(mean_prev)

        covar_predicted = self.covar_dynamics_update(covar_prev)

        R = self.cholesky_decomposition(l_t)

        K = self.kalman_gain(covar_predicted, R, Cz)

        mean_t = self.mean_observation_update(K, z_t, mean_prev, Cz)

        covar_t = self.covar_observation_update(K, covar_prev, Cz)

        return mean_t, covar_t