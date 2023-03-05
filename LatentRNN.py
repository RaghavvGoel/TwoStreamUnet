import torch 
import torch.nn as nn
import ipdb

torch.manual_seed(42)
# torch.manual_seed(101)

class EncoderObservation(nn.Module):
    def __init__(self, device, B, in_channels, height, width, state_dim=256) -> None:
        super().__init__()
        '''
        encoder maps last UNet observation to Z_t and L_t
        '''
        self.device = device
        self.B = B
        self.state_dim = state_dim

        # mapping last layer UNet to small latent state
        self.embedding_conv = nn.Conv2d(in_channels, 1, kernel_size=3,padding=1)
        self.embedding_z = nn.Sequential(nn.ReLU(),
                                       nn.Linear(int(height*width), state_dim))


    def forward(self, x):
        '''
        x -> size is B,T,C,H,W 
            last encoded state of UNet
        '''              
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x_conv = self.embedding_conv(x)
        x_conv = x_conv.view(B*T, -1) #flatten
        x_emb_z = self.embedding_z(x_conv)
        x_emb_z = x_emb_z.view(B, T, self.state_dim)
        return x_emb_z

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

        self.embedding = nn.Sequential(nn.Linear(state_dim, int(height*width)),
                                       nn.ReLU())
        self.embedding_conv_trans = nn.ConvTranspose2d(1, out_channels, kernel_size=3, stride=1, padding=1)                                    

    def forward(self, z):
        '''
        z -> latent state
        '''
        B, T, _ = z.shape
        z = z.reshape(-1, self.state_dim) #z.view(-1, self.state_dim)
        z_emb = self.embedding(z)
        z_emb = z_emb.view(B*T, 1, self.height, self.width)        
        z_conv = self.embedding_conv_trans(z_emb)
        z_conv = z_conv.view(B, T, self.out_channels, self.height, self.width)
        return z_conv

class LSTMModel(nn.Module):

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
        
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=state_dim, batch_first=True)

    def forward(self, x, new_video_flag=False):
        '''
        Forward pass of the Kalman Filter model
        @init_mean: Initial flattened Unet last layer output
        @param l_prev: covariance Cholesky input at all times 
        @param: z_t: measurements array for the whole trajectory
        @returns: y_outs: mean outputs for all times 
        '''
        _,_,C,H,W = x.shape

        z = self.encoder(x)  
        
        out, (_,_) = self.lstm(z)

        x_estimate = self.decoder(out) # size of x_estimate is [B,T,C,H,W] change to [BT,C,H,W]
        x_estimate = x_estimate.view(-1, C,H,W)
        return x_estimate

class LSTMModelEval(LSTMModel):

    def __init__(self, device, B, in_channels, height, width, state_dim=256):
        '''
        Kalman Filter model class
        Implementation of backprop KF paper: https://arxiv.org/pdf/1605.07148.pdf

        @param device
        @param B: Batch Size
        @param in_channels
        @param state_dim: Size of Latent state (flattened)
        '''

        super(LSTMModelEval, self).__init__(device, B, in_channels, height, width, state_dim)

        self.device = device
        self.B = B
        self.in_channels = in_channels        
        self.state_dim = state_dim

        # mapping last layer UNet to state_dim
        # self.encoder = EncoderObservation(device, B, in_channels, height, width, state_dim)

        # mapping state_dim to last_layer UNet
        # self.decoder = DecoderObservation(device, B, in_channels, height, width, state_dim)
        
        # self.lstm = nn.LSTM(input_size=state_dim, hidden_size=state_dim, batch_first=True)

    def forward(self, x, new_video_flag=False):
        '''
        Forward pass of the Kalman Filter model
        @init_mean: Initial flattened Unet last layer output
        @param l_prev: covariance Cholesky input at all times 
        @param: z_t: measurements array for the whole trajectory
        @returns: y_outs: mean outputs for all times 
        '''
        _,_,C,H,W = x.shape

        if new_video_flag:
            self.h = torch.zeros(1, 1, self.state_dim, device=self.device)
            self.c = torch.zeros(1, 1, self.state_dim, device=self.device)
            return x.view(-1, C, H, W)

        z = self.encoder(x)  
        
        out, (self.h,self.c) = self.lstm(z, (self.h,self.c))

        x_estimate = self.decoder(out)
        x_estimate = x_estimate.reshape(-1, C, H, W)

        return x_estimate

