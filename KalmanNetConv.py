from random import gauss
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from Transformer import Transformer

# torch.manual_seed(42)
# torch.manual_seed(10)
torch.manual_seed(101)

class ViT(nn.Module):
    '''
    ViT based multi-head attention
    1) feature maps broken down to tokens to reduce dimensionality
    2) embedding: learned and hard-coded
    '''
    def __init__(self, Cin, Hin, Win, patch_size=4, dim=128, dim_head=64, depth=2, heads=4) -> None:
        super(ViT, self).__init__()
        """
        patch-size: int | p x p
        hidden_dim: int 
        """
        self.Hin, self.Win = Hin, Win
        self.patch_size = patch_size
        patch_dim = Cin* patch_size* patch_size
        num_patches = (Hin//patch_size)*(Win//patch_size)

        #convert images to patches to latent representation
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
        #convert latent representation to images
        self.to_image_embedding = nn.Sequential(
            nn.Linear(dim, patch_dim)
        )

        # use same position mebedding for all three: z_tilde, x_tilde, K_prev
        self.pos_embbeding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, dim)

        self.to_latent = nn.Identity()

    def forward(self, z_tilde, x_tilde, K_prev):
        # encode: tensor to vectors
        B, C, H, W = z_tilde.shape
        # divide into patches
        assert H % self.patch_size == 0 and W % self.patch_size == 0, 'feature map height width should be divisible by patch_size'
        z_x_K_combined = torch.cat([z_tilde, x_tilde, K_prev], dim=0)
        z_x_K_combined = self.to_patch_embedding(z_x_K_combined)
        N = z_x_K_combined.shape[1]
        z_x_K_combined += self.pos_embbeding[:,:N]
        x_K_combined = torch.cat([z_x_K_combined[B:2*B],z_x_K_combined[2*B:]], dim=1)
        
        K_emb = self.transformer(z_x_K_combined[:B],x_K_combined)
        K_emb = self.to_image_embedding(K_emb)
        
        K_emb = rearrange(K_emb, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',p1 = self.patch_size, p2 = self.patch_size, h=self.Hin//self.patch_size, w=self.Win//self.patch_size)
        
        return K_emb

class ConvLSTMCell(nn.Module):
    '''
    Used the implementation of ConvLSTM https://arxiv.org/pdf/1506.04214.pdf
    Repo: https://github.com/ndrplz/ConvLSTM_pytorch
    '''
    def __init__(self, input_dim, hidden_dim, feature_sz, kernel_size=(3,3), bias=False):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.Wci = nn.Parameter(torch.zeros(self.hidden_dim, feature_sz[0], feature_sz[1]))
        self.Wcf = nn.Parameter(torch.zeros(self.hidden_dim, feature_sz[0], feature_sz[1]))
        self.Wco = nn.Parameter(torch.zeros(self.hidden_dim, feature_sz[0], feature_sz[1]))

    def forward(self, input_tensor, cur_state):
        batch = input_tensor.shape[0]

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        # include batches
        Wci_batch = self.Wci.unsqueeze(0).repeat(batch,1,1,1)
        Wcf_batch = self.Wcf.unsqueeze(0).repeat(batch,1,1,1)
        Wco_batch = self.Wco.unsqueeze(0).repeat(batch,1,1,1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i + Wci_batch*c_cur)
        f = torch.sigmoid(cc_f + Wcf_batch*c_cur)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        o = torch.sigmoid(cc_o + Wco_batch*c_next)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, gauss_flag=False):
        height, width = image_size
        if gauss_flag:
            return (3*torch.ones(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                    3*torch.ones(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))                
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class KalmanNetConv(nn.Module):
    def __init__(self, device, B, in_channels, height, width, state_dim=256, embed_channels=16, transformer_flag=False, high_res_flag=False):
        super(KalmanNetConv, self).__init__()
        self.device = device
        self.B = B
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.state_dim = state_dim
        # Input channels = C, EmbedChannels = EC
        self.embed_channels = embed_channels
        self.transformer_flag = transformer_flag

        # H_out = (H_in + 2*p - (k-1) - 1)/stride + 1 | Conv2D
        # 32x32 -> 32x32
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
                                    nn.LayerNorm([self.embed_channels, self.height, self.width]),
                                    nn.ReLU())

        # H_out = (H_in - 1)*stride - 2*p + (k -1) + 1 | ConvTranspose2D
        # 32x32 -> 32x32
        if high_res_flag:
            self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=1, stride=1, padding=0),
                                        nn.LayerNorm([self.embed_channels, self.height, self.width]),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels//2, kernel_size=4, stride=2, padding=1),
                                        nn.LayerNorm([self.in_channels//2, 2*self.height, 2*self.width]),
                                        nn.ReLU()
                                        )
        else:
            self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                                        nn.LayerNorm([self.in_channels, self.height, self.width]),
                                        nn.ReLU(),
                                        # nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels, kernel_size=3, stride=2)
                                        )

        # 32x32 -> 32x32
        self.dynamics_model = nn.Sequential(nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
                                            nn.LayerNorm([self.embed_channels, height, width]),
                                            nn.ReLU(),
                                            # nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=7, stride=1, padding=0,)
                                            )
        # 32x32 -> 32x32
        self.observation_model = nn.Sequential(nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
                                    nn.LayerNorm([self.embed_channels, height, width]),
                                    nn.ReLU(),
                                    # nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=7, stride=1, padding=0,)
                                    )
        # 
        self.RNN_state = nn.Sequential(            
            nn.Conv2d(in_channels=self.embed_channels*2, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1,),
            nn.LayerNorm([self.embed_channels, height, width]),
            nn.ReLU()
        )

        if transformer_flag:
            self.ViT = ViT(self.embed_channels, width, height)
        else:
            # 32x32 -> 32x32
            self.conv_lstm_cell = ConvLSTMCell(input_dim=self.embed_channels*2, hidden_dim=self.embed_channels, feature_sz=(height, width), kernel_size=(3,3), bias=False)

    def forward(self, x, new_video_flag=False):
        '''
        # x: (B, T, C, H, W)
        in_channels = 512
        embed_channels = 128
        '''
        traj_len = x.shape[1]
        # pass through encoder 
        B, T, C, H, W = x.shape
        x_ = x.view(B*T, C, H, W)
        z = self.encoder(x_)

        z = z.view(B, T, self.embed_channels, H, W)

        # self.state_mean = torch.zeros(self.B, self.embed_channels, self.height, self.width, device = self.device)
        self.state_mean = z[:,0]
        mean_list = [self.state_mean]
        state_error_prev = self.state_mean

        if self.transformer_flag:
            h_t = torch.zeros_like(self.state_mean, device=self.device)
        else:
            h_t, c_t =self.conv_lstm_cell.init_hidden(B, image_size=(self.height, self.width))

        for t in range(1,traj_len):
            z_t = z[:,t] #self.encoder(x[:,t])
            # State dynamics: B, EC, H, W
            state_mean_est = self.dynamics_model(mean_list[-1])

            # Observation model: B, EC, H, W
            z_tilde_t = z_t - self.observation_model(state_mean_est)

            if self.transformer_flag:
                h_t = self.ViT(z_tilde_t, state_error_prev, h_t)
            else:
                # Concatenate along the channels
                rnn_input = torch.cat([z_tilde_t, state_error_prev], dim=1)
                # rnn_state = rnn_input #self.RNN_state(rnn_input)
                # ConvLSTM
                h_t, c_t = self.conv_lstm_cell(input_tensor =rnn_input, cur_state=(h_t, c_t))

            K_t = h_t # B, EC, H, W

            # Update state
            state_mean_updated = state_mean_est + K_t * z_tilde_t
            # state_mean_updated = state_mean_est + K_t

            # state error
            state_error_prev = state_mean_updated - state_mean_est

            self.state_mean = state_mean_updated

            # Bookkeping
            mean_list.append(self.state_mean)
        

        # mean_list.pop(0)
        mean_list = torch.stack(mean_list, dim=1)
        mean_list = mean_list.view(B*T, self.embed_channels, H, W)
        x_estimate = self.decoder(mean_list)
        # x_estimate = x_estimate.view(B, T, C//2, 2*H, 2*W)

        return x_estimate



class KalmanNetConvVal(KalmanNetConv):
    def __init__(self, device, B, in_channels, height, width, state_dim=256, embed_channels=16):
        super(KalmanNetConvVal, self).__init__(device, B, in_channels, height, width, state_dim, embed_channels)
        self.device = device
        self.B = B
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.state_dim = state_dim
        self.embed_channels = embed_channels

        #* Note: Every initialization will swap out any Kalman Net Conv history
        self.state_mean = None
        self.h_t = None
        self.c_t = None

        # H_out = (H_in + 2*p - (k-1) - 1)/stride + 1 | Conv2D
        # 32x32 -> 32x32
        # self.encoder = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
        #                             nn.LayerNorm([self.embed_channels, self.height, self.width]),
        #                             nn.ReLU())
        # # H_out = (H_in - 1)*stride - 2*p + (k -1) + 1 | ConvTranspose2D
        # # 32x32 -> 32x32
        # self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
        #                             nn.LayerNorm([self.in_channels, self.height, self.width]),
        #                             nn.ReLU(),
        #                             # nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels, kernel_size=3, stride=2)
        #                             )

        # # 32x32 -> 32x32
        # self.dynamics_model = nn.Sequential(nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
        #                                     nn.LayerNorm([self.embed_channels, height, width]),
        #                                     nn.ReLU(),
        #                                     # nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=7, stride=1, padding=0,)
        #                                     )
        # # 32x32 -> 32x32
        # self.observation_model = nn.Sequential(nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
        #                             nn.LayerNorm([self.embed_channels, height, width]),
        #                             nn.ReLU(),
        #                             # nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=7, stride=1, padding=0,)
        #                             )
        # # 
        # self.RNN_state = nn.Sequential(            
        #     nn.Conv2d(in_channels=self.embed_channels*2, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1,),
        #     nn.LayerNorm([self.embed_channels, height, width]),
        #     nn.ReLU()
        # )
        # # 32x32 -> 32x32
        # self.conv_lstm_cell = ConvLSTMCell(input_dim=self.embed_channels*2, hidden_dim=self.embed_channels, feature_sz=(height, width) ,kernel_size=(3,3), bias=False)

    def forward(self, x, new_video_flag=False):
        '''
        # x: (B, C, H, W)
        in_channels = 512
        embed_channels = 128
        '''

        # pass through encoder 
        B, T, C, H, W = x.shape
        x_ = x.view(B*T, C, H, W)
        
        z = self.encoder(x_)

        # re-initialise the network
        if new_video_flag:
            # print("new video")
            self.state_mean = z
            self.mean_list = [self.state_mean]
            self.state_error_prev = self.state_mean
            self.h_t, self.c_t =self.conv_lstm_cell.init_hidden(B, image_size=(self.height, self.width))
            new_video_flag = False            

        z_t = z #self.encoder(x[:,t])  
        # State dynamics: B, EC, H, W
        state_mean_est = self.dynamics_model(self.mean_list[-1])

        # Observation model: B, EC, H, W
        z_tilde_t = z_t - self.observation_model(state_mean_est)

        # Concatenate along the channels
        rnn_state = torch.cat([z_tilde_t, self.state_error_prev], dim=1) 
                
        # ConvLSTM
        self.h_t, self.c_t = self.conv_lstm_cell(input_tensor =rnn_state, cur_state=(self.h_t, self.c_t))

        K_t = self.h_t # B, EC, H, W

        # Update state
        state_mean_updated = state_mean_est + K_t * z_tilde_t
        # state_mean_updated = state_mean_est + K_t

        # state error
        self.state_error_prev = state_mean_updated - state_mean_est

        self.state_mean = state_mean_updated

        # Bookkeping | no need of this 
        self.mean_list.append(self.state_mean)


        x_estimate = self.decoder(self.state_mean)
        x_estimate = x_estimate.view(B, T, C, H, W)

        return x_estimate

    def process_model(self, x, new_video_flag=False):
        
        # pass through encoder 
        B, C, H, W = x.shape
        x_ = x.view(B, C, H, W)
        
        z = self.encoder(x_)

        seq_len = 7
        states_list = [z]
        # generate next x purely using process_model
        for i in range(seq_len):
            state_mean_est = self.dynamics_model(states_list[-1])
            states_list.append(state_mean_est)

        states_list = torch.cat(states_list, dim=0) # along batch
        states_list = self.decoder(states_list)
        
        return states_list

class KalmanNetConvGauss(nn.Module):
    def __init__(self, device, B, in_channels, height, width, state_dim=256, embed_channels=16):
        super(KalmanNetConvGauss, self).__init__()

        self.device = device
        self.B = B
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.state_dim = state_dim
        # Input channels = C, EmbedChannels = EC
        self.embed_channels = embed_channels

        # H_out = (H_in + 2*p - (k-1) - 1)/stride + 1 | Conv2D
        # 32x32 -> 32x32
        self.encoder = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
                                    nn.LayerNorm([self.embed_channels, self.height, self.width]),
                                    nn.ReLU())
        # H_out = (H_in - 1)*stride - 2*p + (k -1) + 1 | ConvTranspose2D
        # 32x32 -> 32x32
        self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                                    nn.LayerNorm([self.in_channels, self.height, self.width]),
                                    nn.ReLU(),
                                    # nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels, kernel_size=3, stride=2)
                                    )

        # 32x32 -> 32x32
        self.dynamics_model = nn.Sequential(nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
                                            nn.LayerNorm([self.embed_channels, height, width]),
                                            nn.ReLU(),
                                            # nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=7, stride=1, padding=0,)
                                            )
        # 32x32 -> 32x32
        self.conv_lstm_cell_Sigma = ConvLSTMCell(input_dim=self.embed_channels, hidden_dim=self.embed_channels,feature_sz=(height, width), kernel_size=(3,3), bias=False)

        # 32x32 -> 32x32
        self.observation_model = nn.Sequential(nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
                                    nn.LayerNorm([self.embed_channels, height, width]),
                                    nn.ReLU(),
                                    # nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=7, stride=1, padding=0,)
                                    )
        #
        self.RNN_state = nn.Sequential(
            nn.Conv2d(in_channels=self.embed_channels*2, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1,),
            nn.LayerNorm([self.embed_channels, height, width]),
            nn.ReLU()
        )
        # 32x32 -> 32x32 | 3 channels = [z_tilde, state_error_prev, Sigma]
        self.conv_lstm_cell_KG = ConvLSTMCell(input_dim=self.embed_channels*3, hidden_dim=self.embed_channels, feature_sz=(height, width), kernel_size=(3,3), bias=False)


    def forward(self, x, new_video_flag=False):
        '''
        # x: (B, T, C, H, W)
        in_channels = 512
        embed_channels = 128
        '''
        traj_len = x.shape[1]

        # pass through encoder
        B, T, C, H, W = x.shape
        x_ = x.view(B*T, C, H, W)
        z = self.encoder(x_)
        # reshape measurement to have time dimension
        z = z.view(B, T, self.embed_channels, H, W)

        self.state_mean = z[:,0]
        mean_list = [self.state_mean]
        state_error_prev = self.state_mean
        # state_error_process = self.state_mean


        h_KG_t, c_KG_t = self.conv_lstm_cell_KG.init_hidden(B, image_size=(self.height, self.width))
        h_Sigma_t, c_Sigma_t = self.conv_lstm_cell_Sigma.init_hidden(B, image_size=(self.height, self.width), gauss_flag=True)
        Sigma_list = [h_Sigma_t]

        for t in range(1,traj_len):
            z_t = z[:,t] #self.encoder(x[:,t])  
            # State dynamics: B, EC, H, W
            state_mean_est = self.dynamics_model(mean_list[-1])
            ## find Sigma | can include KG_prev appended with state_error_process?
            state_error_process = state_mean_est - mean_list[-1]
            h_Sigma_t, c_Sigma_t = self.conv_lstm_cell_Sigma(input_tensor = state_error_process, cur_state=(h_Sigma_t, c_Sigma_t))
            Sigma_t = h_Sigma_t

            # Observation model: B, EC, H, W
            z_tilde_t = z_t - self.observation_model(state_mean_est)

            # Concatenate along the channels
            rnn_state = torch.cat([z_tilde_t, state_error_prev, Sigma_t], dim=1) 
            # rnn_state = rnn_input #self.RNN_state(rnn_input)            
            
            # ConvLSTM
            h_KG_t, c_KG_t = self.conv_lstm_cell_KG(input_tensor =rnn_state, cur_state=(h_KG_t, c_KG_t))

            K_t = h_KG_t # B, EC, H, W

            # Update state
            state_mean_updated = state_mean_est + K_t * z_tilde_t
            # state_mean_updated = state_mean_est + K_t

            # state error
            state_error_prev = state_mean_updated - state_mean_est

            self.state_mean = state_mean_updated

            # Bookkeping
            mean_list.append(self.state_mean)
            Sigma_list.append(Sigma_t)

        # mean_list.pop(0)
        mean_list = torch.stack(mean_list, dim=1)
        mean_list = mean_list.view(B*T, self.embed_channels, H, W)
        x_estimate = self.decoder(mean_list)
        x_estimate = x_estimate.view(B, T, C, H, W)

        #Sigma list 
        Sigma_list = torch.stack(Sigma_list, dim=1)
        Sigma_list = Sigma_list.view(B*T, self.embed_channels, H, W)
        Sigma_list = self.decoder(Sigma_list)
        Sigma_list = Sigma_list.view(B, T, C, H, W) 

        return x_estimate, Sigma_list


class KalmanNetConvGaussVal(KalmanNetConvGauss):
    def __init__(self, device, B, in_channels, height, width, state_dim=256, embed_channels=16):
        super(KalmanNetConvGaussVal, self).__init__(device, B, in_channels, height, width, state_dim, embed_channels)
        # self.device = device
        # self.B = B
        # self.in_channels = in_channels
        # self.height = height
        # self.width = width
        # self.state_dim = state_dim
        # self.embed_channels = embed_channels

        #* Note: Every initialization will swap out any Kalman Net Conv history
        self.state_mean = None
        self.h_KG_t = None
        self.c_KG_t = None
        self.h_Sigma_t = None
        self.c_Sigma_t = None

        # # H_out = (H_in + 2*p - (k-1) - 1)/stride + 1 | Conv2D
        # # 32x32 -> 32x32
        # self.encoder = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
        #                             nn.LayerNorm([self.embed_channels, self.height, self.width]),
        #                             nn.ReLU())
        # # H_out = (H_in - 1)*stride - 2*p + (k -1) + 1 | ConvTranspose2D
        # # 32x32 -> 32x32
        # self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
        #                             nn.LayerNorm([self.in_channels, self.height, self.width]),
        #                             nn.ReLU(),
        #                             # nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels, kernel_size=3, stride=2)
        #                             )

        # # 32x32 -> 32x32
        # self.dynamics_model = nn.Sequential(nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
        #                                     nn.LayerNorm([self.embed_channels, height, width]),
        #                                     nn.ReLU(),
        #                                     # nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=7, stride=1, padding=0,)
        #                                     )
        # # 32x32 -> 32x32
        # self.observation_model = nn.Sequential(nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
        #                             nn.LayerNorm([self.embed_channels, height, width]),
        #                             nn.ReLU(),
        #                             # nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=7, stride=1, padding=0,)
        #                             )
        # # 
        # self.RNN_state = nn.Sequential(            
        #     nn.Conv2d(in_channels=self.embed_channels*2, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1,),
        #     nn.LayerNorm([self.embed_channels, height, width]),
        #     nn.ReLU()
        # )

        # # 32x32 -> 32x32 | 3 channels = [z_tilde, state_error_prev, Sigma]
        # self.conv_lstm_cell_KG = ConvLSTMCell(input_dim=self.embed_channels*3, hidden_dim=self.embed_channels, kernel_size=(3,3), bias=False)

        # # 32x32 -> 32x32
        # self.conv_lstm_cell_Sigma = ConvLSTMCell(input_dim=self.embed_channels, hidden_dim=self.embed_channels, kernel_size=(3,3), bias=False)

    def forward(self, x, new_video_flag=False):
        '''
        # x: (B, C, H, W)
        in_channels = 512
        embed_channels = 128
        '''

        # pass through encoder 
        B, T, C, H, W = x.shape
        x_ = x.view(B*T, C, H, W)
        
        z = self.encoder(x_)

        # re-initialise the network
        if new_video_flag:
            self.state_mean = z            
            self.h_KG_t, self.c_KG_t = self.conv_lstm_cell_KG.init_hidden(B, image_size=(self.height, self.width))
            self.h_Sigma_t, self.c_Sigma_t = self.conv_lstm_cell_Sigma.init_hidden(B, image_size=(self.height, self.width), gauss_flag=True)
            self.Sigma_t = self.h_Sigma_t
            x_estimate = self.decoder(self.state_mean).unsqueeze(0)
            self.Sigma_t = self.decoder(self.Sigma_t).unsqueeze(0)
            new_video_flag = False     
            return x_estimate, self.Sigma_t

        state_error_prev = self.state_mean

        z_t = z #self.encoder(x[:,t])  
        # State dynamics: B, EC, H, W
        state_mean_est = self.dynamics_model(self.state_mean)
        # process model uncertainty
        state_error_process = state_mean_est - self.state_mean
        self.h_Sigma_t, self.c_Sigma_t = self.conv_lstm_cell_Sigma(input_tensor = state_error_process, cur_state=(self.h_Sigma_t, self.c_Sigma_t))
        self.Sigma_t = self.h_Sigma_t

        # Observation model: B, EC, H, W
        z_tilde_t = z_t - self.observation_model(state_mean_est)

        # Concatenate along the channels
        rnn_state = torch.cat([z_tilde_t, state_error_prev, self.Sigma_t], dim=1) 
                
        # ConvLSTM
        self.h_KG_t, self.c_KG_t = self.conv_lstm_cell_KG(input_tensor =rnn_state, cur_state=(self.h_KG_t, self.c_KG_t))

        K_t = self.h_KG_t # B, EC, H, W

        # Update state
        state_mean_updated = state_mean_est + K_t * z_tilde_t
        # state_mean_updated = state_mean_est + K_t

        # state error
        state_error_prev = state_mean_updated - state_mean_est

        self.state_mean = state_mean_updated                

        x_estimate = self.decoder(self.state_mean).unsqueeze(0)
        self.Sigma_t = self.decoder(self.Sigma_t).unsqueeze(0)

        # x_estimate = x_estimate.view(T, C, H, W)
        # Sigma_t = Sigma_t.view(T, C, H, W)

        return x_estimate, self.Sigma_t 


class KalmanNetConvMoCA(nn.Module):
    def __init__(self, device, B, in_channels, height, width, state_dim=256, embed_channels=16, transformer_flag=False, high_res_flag=False, ViT_flag=False):
        super(KalmanNetConvMoCA, self).__init__()
        self.device = device
        self.B = B
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.state_dim = state_dim
        self.embed_channels = embed_channels
        self.transformer_flag = transformer_flag
        self.ViT_flag = ViT_flag

        # H_out = (H_in + 2*p - (k-1) - 1)/stride + 1 | Conv2D
        # 32x32 -> 32x32
        if self.transformer_flag:
            # self.encoder = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
            #                             nn.LayerNorm([self.embed_channels, self.height, self.width]),
            #                             nn.ReLU())
            self.encoder = nn.AvgPool2d(self.height)
        else:
            self.encoder = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
                                        nn.LayerNorm([self.embed_channels, self.height, self.width]),
                                        nn.ReLU())

        # H_out = (H_in - 1)*stride - 2*p + (k -1) + 1 | ConvTranspose2D
        # 32x32 -> 32x32
        if self.transformer_flag:
            # C x 1 x 1 -> C x 28 x 28
            img_sz_flat = int(self.in_channels*self.height*self.width)
            self.decoder = nn.Sequential(nn.Linear(self.in_channels, 2*self.in_channels),
                                                nn.BatchNorm1d(2*self.in_channels),
                                                nn.ReLU(),
                                                nn.Linear(2*self.in_channels, img_sz_flat),
                                                nn.BatchNorm1d(img_sz_flat),
                                                nn.ReLU()
                                                )

            # self.decoder = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        
            #                             )
        else:
            if high_res_flag:
                # double H,W and half C
                self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=1, stride=1, padding=0),
                                            nn.LayerNorm([self.embed_channels, self.height, self.width]),
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels//2, kernel_size=4, stride=2, padding=1),
                                            nn.LayerNorm([self.in_channels//2, 2*self.height, 2*self.width]),
                                            nn.ReLU()
                                            )
            else:
                self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                                            nn.LayerNorm([self.in_channels, self.height, self.width]),
                                            nn.ReLU(),
                                            # nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels, kernel_size=3, stride=2)
                                            )

        # 32x32 -> 32x32
        self.dynamics_model = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1),
                                            nn.LayerNorm([self.in_channels, height, width]),
                                            nn.ReLU(),
                                            # nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=7, stride=1, padding=0,)
                                            )
        # 32x32 -> 32x32
        self.observation_model = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1),
                                    nn.LayerNorm([self.in_channels, height, width]),
                                    nn.ReLU(),
                                    # nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=7, stride=1, padding=0,)
                                    )
        # 
        # self.RNN_state = nn.Sequential(            
        #     nn.Conv2d(in_channels=self.embed_channels*2, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1,),
        #     nn.LayerNorm([self.embed_channels, height, width]),
        #     nn.ReLU()
        # )

        if ViT_flag:
            self.ViT = ViT(self.embed_channels, width, height)
        elif transformer_flag:
            self.transformer = Transformer(dim=self.in_channels, depth=1, heads=4, dim_head=128)
        else:
            # 32x32 -> 32x32
            self.conv_lstm_cell = ConvLSTMCell(input_dim=self.embed_channels*2, hidden_dim=self.embed_channels, feature_sz=(height, width), kernel_size=(3,3), bias=False)

    def forward(self, z, new_video_flag=False):
        '''
        # x: (B, T, C, H, W)
        in_channels = 512
        embed_channels = 128
        using transformers on past (x_tilde, z_tilde, h_{t-1}) using current (x_tilde, z_tilde) as query
        '''
        traj_len = z.shape[1]
        # pass through encoder 
        B, T, C, H, W = z.shape
        if not self.transformer_flag:
            z_ = z.view(B*T, C, H, W)
            z = self.encoder(z_)
            z = z.view(B, T, self.embed_channels, H, W)
        # else:
        #     z = z.view(B, T, self.in_channels, H, W)


        # self.state_mean = torch.zeros(self.B, self.embed_channels, self.height, self.width, device = self.device)
        self.state_mean = z[:,0]
        mean_list = [self.state_mean]
        state_error_prev = self.state_mean

        if self.ViT_flag:
            h_t = torch.zeros_like(self.state_mean, device=self.device)
        
        elif self.transformer_flag:
            h_t = torch.zeros(B, 1, self.in_channels, device=self.device)
            transformer_keys_values_list = []
        
        else:
            h_t, c_t =self.conv_lstm_cell.init_hidden(B, image_size=(self.height, self.width))

        for t in range(1,traj_len):
            z_t = z[:,t] #self.encoder(x[:,t])
            # State dynamics: B, EC, H, W
            state_mean_est = self.dynamics_model(mean_list[-1])

            # Observation model: B, EC, H, W
            z_tilde_t = z_t - self.observation_model(state_mean_est)

            if self.ViT_flag:
                h_t = self.ViT(z_tilde_t, state_error_prev, h_t)
            
            elif self.transformer_flag:
                # C x 28 x 28 -> C x 1 x 1 map state_prev_err and z_tilde to vector                
                state_prev_err_vec = self.encoder(state_error_prev).view(B, 1, self.in_channels)
                z_tilde_t_vec = self.encoder(z_tilde_t).view(B, 1, self.in_channels)
                trans_query = (state_prev_err_vec + z_tilde_t_vec)/2
                if len(transformer_keys_values_list) == 0:
                    h_t_vec = trans_query
                else:
                    transformer_keys_values_arr = torch.cat(transformer_keys_values_list, dim=1)
                    trans_feat = torch.cat([h_t_vec, transformer_keys_values_arr], dim = 1)
                    h_t_vec = self.transformer(trans_query, trans_feat)
                transformer_keys_values_list.append(trans_query)
                # decode h_t to same size as x_t z_t
                h_t = self.decoder(h_t_vec.squeeze(1))
                h_t = rearrange(h_t, 'b (c h w) -> b c h w', c=self.in_channels, h = self.height, w=self.width)
            
            else:
                # Concatenate along the channels
                rnn_input = torch.cat([z_tilde_t, state_error_prev], dim=1)
                # rnn_state = rnn_input #self.RNN_state(rnn_input)
                # ConvLSTM
                h_t, c_t = self.conv_lstm_cell(input_tensor =rnn_input, cur_state=(h_t, c_t))

            K_t = h_t # B, EC, H, W

            # Update state
            state_mean_updated = state_mean_est + K_t * z_tilde_t
            # state_mean_updated = state_mean_est + K_t

            # state error
            state_error_prev = state_mean_updated - state_mean_est

            self.state_mean = state_mean_updated

            # Bookkeping
            mean_list.append(self.state_mean)
        

        x_estimate = torch.stack(mean_list, dim=1)
        if self.transformer_flag:
            x_estimate = x_estimate.view(B*T, self.in_channels, H, W)

        else:
            x_estimate = x_estimate.view(B*T, self.embed_channels, H, W)
            x_estimate = self.decoder(mean_list)
        # x_estimate = x_estimate.view(B, T, C//2, 2*H, 2*W)

        return x_estimate

class KalmanNetConvSpatialTemporal(nn.Module):
    def __init__(self, device, B, in_channels, height, width, state_dim=256, embed_channels=16, high_res_flag=False):
        super(KalmanNetConvSpatialTemporal, self).__init__()
        self.device = device
        self.B = B
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.state_dim = state_dim
        self.embed_channels = embed_channels

        # H_out = (H_in + 2*p - (k-1) - 1)/stride + 1 | Conv2D
        # 32x32 -> 32x32
        if self.transformer_flag:
            # self.encoder = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
            #                             nn.LayerNorm([self.embed_channels, self.height, self.width]),
            #                             nn.ReLU())
            self.encoder = nn.AvgPool2d(self.height)
        else:
            self.encoder = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1),
                                        nn.LayerNorm([self.embed_channels, self.height, self.width]),
                                        nn.ReLU())

        # H_out = (H_in - 1)*stride - 2*p + (k -1) + 1 | ConvTranspose2D
        # 32x32 -> 32x32
        if self.transformer_flag:
            # C x 1 x 1 -> C x 28 x 28
            img_sz_flat = int(self.in_channels*self.height*self.width)
            self.decoder = nn.Sequential(nn.Linear(self.in_channels, 2*self.in_channels),
                                                nn.BatchNorm1d(2*self.in_channels),
                                                nn.ReLU(),
                                                nn.Linear(2*self.in_channels, img_sz_flat),
                                                nn.BatchNorm1d(img_sz_flat),
                                                nn.ReLU()
                                                )

            # self.decoder = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                        
            #                             )
        else:
            if high_res_flag:
                # double H,W and half C
                self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=1, stride=1, padding=0),
                                            nn.LayerNorm([self.embed_channels, self.height, self.width]),
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels//2, kernel_size=4, stride=2, padding=1),
                                            nn.LayerNorm([self.in_channels//2, 2*self.height, 2*self.width]),
                                            nn.ReLU()
                                            )
            else:
                self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                                            nn.LayerNorm([self.in_channels, self.height, self.width]),
                                            nn.ReLU(),
                                            # nn.ConvTranspose2d(in_channels=self.embed_channels, out_channels=self.in_channels, kernel_size=3, stride=2)
                                            )

        # 32x32 -> 32x32
        self.dynamics_model = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1),
                                            nn.LayerNorm([self.in_channels, height, width]),
                                            nn.ReLU(),
                                            # nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=7, stride=1, padding=0,)
                                            )
        # 32x32 -> 32x32
        self.observation_model = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1),
                                    nn.LayerNorm([self.in_channels, height, width]),
                                    nn.ReLU(),
                                    # nn.Conv2d(in_channels=self.embed_channels, out_channels=self.embed_channels, kernel_size=7, stride=1, padding=0,)
                                    )
        # 
        # self.RNN_state = nn.Sequential(            
        #     nn.Conv2d(in_channels=self.embed_channels*2, out_channels=self.embed_channels, kernel_size=3, stride=1, padding=1,),
        #     nn.LayerNorm([self.embed_channels, height, width]),
        #     nn.ReLU()
        # )

        if ViT_flag:
            self.ViT = ViT(self.embed_channels, width, height)
        elif transformer_flag:
            self.transformer = Transformer(dim=self.in_channels, depth=1, heads=4, dim_head=128)
        else:
            # 32x32 -> 32x32
            self.conv_lstm_cell = ConvLSTMCell(input_dim=self.embed_channels*2, hidden_dim=self.embed_channels, feature_sz=(height, width), kernel_size=(3,3), bias=False)

    def forward(self, z, new_video_flag=False):
        '''
        # x: (B, T, C, H, W)
        in_channels = 512
        embed_channels = 128
        using transformers on past (x_tilde, z_tilde, h_{t-1}) using current (x_tilde, z_tilde) as query
        '''
        traj_len = z.shape[1]
        # pass through encoder 
        B, T, C, H, W = z.shape
        if not self.transformer_flag:
            z_ = z.view(B*T, C, H, W)
            z = self.encoder(z_)
            z = z.view(B, T, self.embed_channels, H, W)
        # else:
        #     z = z.view(B, T, self.in_channels, H, W)


        # self.state_mean = torch.zeros(self.B, self.embed_channels, self.height, self.width, device = self.device)
        self.state_mean = z[:,0]
        mean_list = [self.state_mean]
        state_error_prev = self.state_mean

        if self.ViT_flag:
            h_t = torch.zeros_like(self.state_mean, device=self.device)
        
        elif self.transformer_flag:
            h_t = torch.zeros(B, 1, self.in_channels, device=self.device)
            transformer_keys_values_list = []
        
        else:
            h_t, c_t =self.conv_lstm_cell.init_hidden(B, image_size=(self.height, self.width))

        for t in range(1,traj_len):
            z_t = z[:,t] #self.encoder(x[:,t])
            # State dynamics: B, EC, H, W
            state_mean_est = self.dynamics_model(mean_list[-1])

            # Observation model: B, EC, H, W
            z_tilde_t = z_t - self.observation_model(state_mean_est)

            if self.ViT_flag:
                h_t = self.ViT(z_tilde_t, state_error_prev, h_t)
            
            elif self.transformer_flag:
                # C x 28 x 28 -> C x 1 x 1 map state_prev_err and z_tilde to vector                
                state_prev_err_vec = self.encoder(state_error_prev).view(B, 1, self.in_channels)
                z_tilde_t_vec = self.encoder(z_tilde_t).view(B, 1, self.in_channels)
                trans_query = (state_prev_err_vec + z_tilde_t_vec)/2
                if len(transformer_keys_values_list) == 0:
                    h_t_vec = trans_query
                else:
                    transformer_keys_values_arr = torch.cat(transformer_keys_values_list, dim=1)
                    trans_feat = torch.cat([h_t_vec, transformer_keys_values_arr], dim = 1)
                    h_t_vec = self.transformer(trans_query, trans_feat)
                transformer_keys_values_list.append(trans_query)
                # decode h_t to same size as x_t z_t
                h_t = self.decoder(h_t_vec.squeeze(1))
                h_t = rearrange(h_t, 'b (c h w) -> b c h w', c=self.in_channels, h = self.height, w=self.width)
            
            else:
                # Concatenate along the channels
                rnn_input = torch.cat([z_tilde_t, state_error_prev], dim=1)
                # rnn_state = rnn_input #self.RNN_state(rnn_input)
                # ConvLSTM
                h_t, c_t = self.conv_lstm_cell(input_tensor =rnn_input, cur_state=(h_t, c_t))

            K_t = h_t # B, EC, H, W

            # Update state
            state_mean_updated = state_mean_est + K_t * z_tilde_t
            # state_mean_updated = state_mean_est + K_t

            # state error
            state_error_prev = state_mean_updated - state_mean_est

            self.state_mean = state_mean_updated

            # Bookkeping
            mean_list.append(self.state_mean)
        

        x_estimate = torch.stack(mean_list, dim=1)
        if self.transformer_flag:
            x_estimate = x_estimate.view(B*T, self.in_channels, H, W)

        else:
            x_estimate = x_estimate.view(B*T, self.embed_channels, H, W)
            x_estimate = self.decoder(mean_list)
        # x_estimate = x_estimate.view(B, T, C//2, 2*H, 2*W)

        return x_estimate        