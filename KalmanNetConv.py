import torch 
import torch.nn as nn

# torch.manual_seed(42)
torch.manual_seed(42)

class ConvLSTMCell(nn.Module):
    '''
    Used the implementation of ConvLSTM https://arxiv.org/pdf/1506.04214.pdf
    Repo: https://github.com/ndrplz/ConvLSTM_pytorch
    '''
    def __init__(self, input_dim, hidden_dim, kernel_size=(3,3), bias=False):
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

        self.Wci = nn.Parameter(torch.zeros(self.hidden_dim, 32, 32))
        self.Wcf = nn.Parameter(torch.zeros(self.hidden_dim, 32, 32))
        self.Wco = nn.Parameter(torch.zeros(self.hidden_dim, 32, 32))

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

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class KalmanNetConv(nn.Module):
    def __init__(self, device, B, in_channels, height, width, state_dim=256, embed_channels=16):
        super(KalmanNetConv, self).__init__()
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
        # 32x32 -> 32x32
        self.conv_lstm_cell = ConvLSTMCell(input_dim=self.embed_channels*2, hidden_dim=self.embed_channels, kernel_size=(3,3), bias=False)

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

        h_t, c_t =self.conv_lstm_cell.init_hidden(B, image_size=(self.height, self.width))
        for t in range(1,traj_len):
            z_t = z[:,t] #self.encoder(x[:,t])  
            # State dynamics: B, EC, H, W
            state_mean_est = self.dynamics_model(mean_list[-1])

            # Observation model: B, EC, H, W
            z_tilde_t = z_t - self.observation_model(state_mean_est)

            # Concatenate along the channels
            rnn_input = torch.cat([z_tilde_t, state_error_prev], dim=1) 
            
            rnn_state = rnn_input #self.RNN_state(rnn_input)            
            # ConvLSTM
            h_t, c_t = self.conv_lstm_cell(input_tensor =rnn_state, cur_state=(h_t, c_t))

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
        x_estimate = x_estimate.view(B, T, C, H, W)

        return x_estimate



class KalmanNetConvVal(nn.Module):
    def __init__(self, device, B, in_channels, height, width, state_dim=256, embed_channels=16):
        super(KalmanNetConvVal, self).__init__()
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
        # 32x32 -> 32x32
        self.conv_lstm_cell = ConvLSTMCell(input_dim=self.embed_channels*2, hidden_dim=self.embed_channels, kernel_size=(3,3), bias=False)

    def forward(self, x, new_video_flag=False):
        '''
        # x: (B, C, H, W)
        in_channels = 512
        embed_channels = 128
        '''
        # re-initialise the network
        if new_video_flag:
            # print("new video")
            self.state_mean = None
            self.h_t = None 
            self.c_t = None
            new_video_flag = False            

        # pass through encoder 
        B, T, C, H, W = x.shape
        x_ = x.view(B*T, C, H, W)
        
        z = self.encoder(x_)

        # self.state_mean = torch.zeros(self.B, self.embed_channels, self.height, self.width, device = self.device)
        if self.state_mean is None:
            self.state_mean = z
            self.mean_list = [self.state_mean]       
        state_error_prev = self.state_mean
        if self.h_t is None and self.c_t is None:
            self.h_t, self.c_t =self.conv_lstm_cell.init_hidden(B, image_size=(self.height, self.width))

        z_t = z #self.encoder(x[:,t])  
        # State dynamics: B, EC, H, W
        state_mean_est = self.dynamics_model(self.mean_list[-1])

        # Observation model: B, EC, H, W
        z_tilde_t = z_t - self.observation_model(state_mean_est)

        # Concatenate along the channels
        rnn_input = torch.cat([z_tilde_t, state_error_prev], dim=1) 
        
        rnn_state = rnn_input #self.RNN_state(rnn_input)            
        # ConvLSTM
        self.h_t, self.c_t = self.conv_lstm_cell(input_tensor =rnn_state, cur_state=(self.h_t, self.c_t))

        K_t = self.h_t # B, EC, H, W

        # Update state
        state_mean_updated = state_mean_est + K_t * z_tilde_t
        # state_mean_updated = state_mean_est + K_t

        # state error
        state_error_prev = state_mean_updated - state_mean_est

        self.state_mean = state_mean_updated

        # Bookkeping | no need of this 
        self.mean_list.append(self.state_mean)


        x_estimate = self.decoder(self.state_mean)
        x_estimate = x_estimate.view(B, T, C, H, W)

        return x_estimate


        