import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from Autocorrelation import *

# ODE function
class ODEfunc(nn.Module):
    def __init__(self, nhidden):
        super(ODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(nhidden, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, nhidden)

    def forward(self, t, x):
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class ODELSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, method, ode_func=None):
        super(ODELSTMCell, self).__init__()
        self.method = method
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.ode_func = ODEfunc(hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, input_, hx, ts):
        new_h, new_c = self.lstm(input_, hx)
        new_h = odeint(self.ode_func, new_h, ts, method=self.method)[-1]
        return new_h, new_c


class ODELSTM(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_size,
            out_feature,
            solver_type,
            device
    ):
        super(ODELSTM, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.device = device
        self.rnn = ODELSTMCell(in_features, hidden_size, solver_type)
        self.fc = nn.Linear(self.hidden_size, self.out_feature)

    def forward(self, x, ts):
        batch_size = x.size(0)
        hidden_state = [
            torch.zeros((batch_size, self.hidden_size), device=self.device),
            torch.zeros((batch_size, self.hidden_size), device=self.device),
        ]
        last_output = torch.zeros((batch_size, self.out_feature), device=self.device)
        for t in range(x.shape[1]):
            inp = x[:, t].unsqueeze(1)
            time = ts[t].unsqueeze(0)
            hidden_state = self.rnn(inp, hidden_state, time)
            last_output = self.fc(hidden_state[0])
        return last_output


class ODEnet(nn.Module):
    def __init__(self, hid_size):
        super(ODEnet, self).__init__()

        self.hidden_dim = hid_size
        self.lin_hi = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.lin_hf = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.lin_hg = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.lin_ho = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out = nn.Linear(self.hidden_dim, hid_size)

    def forward(self, t, dz):
        x = torch.zeros_like(dz)
        i = F.sigmoid(x + self.lin_hi(dz))
        f = F.sigmoid(x + self.lin_hf(dz))
        g = F.tanh(x + self.lin_hg(dz))
        o = F.sigmoid(x + self.lin_ho(dz))
        dh = o * F.tanh(i * g)
        dh = F.softmax(self.out(dh), dim=1)
        return dh


class Predictor(nn.Module):
    def __init__(self, hid_size, output_size, method):
        super(Predictor, self).__init__()
        self.hid_size = hid_size
        self.output_size = output_size
        self.method = method
        self.odefunc = ODEnet(hid_size)
        self.fc = nn.Linear(hid_size, output_size)

    def forward(self, y, ts):
        zt = odeint(self.odefunc, y, ts, method=self.method)
        zt = F.softmax(zt, dim=2).transpose(0, 1)
        return self.fc(zt).squeeze(2)

class ODETransformer(nn.Module):
    def __init__(self, n_heads: int,
                 driving_size: int,
                 att_dim: int,
                 hidden_rnn: int,
                 hid_size: int,
                 output_size: int,
                 method: str,
                 device):
        super(ODETransformer, self).__init__()
        self.autocor = AutoCorrelation().to(device)
        self.attention = AutoCorrelationLayer(self.autocor, n_heads, driving_size, att_dim * 2).to(device)
        self.encoder = ODELSTM(1, hidden_rnn, hid_size, method, device).to(device)
        self.decoder = Predictor(hid_size+driving_size, output_size, method).to(device)

    def forward(self, tar_y, dri_x, ts, t):
        zx = self.attention(dri_x)[:, -1, :]
        zy = self.encoder(tar_y, ts)
        z = torch.cat((zx, zy), 1)
        zt = self.decoder(z, t)
        return zt

