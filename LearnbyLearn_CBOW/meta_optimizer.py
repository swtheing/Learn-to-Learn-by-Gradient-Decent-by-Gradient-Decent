from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
from utils import preprocess_gradients
from layer_norm_lstm import LayerNormLSTMCell
from layer_norm import LayerNorm1D

class MetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(MetaOptimizer, self).__init__()
        self.meta_model = model

        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(3, hidden_size)
        self.ln1 = LayerNorm1D(hidden_size)

        self.lstms = []
        for i in range(num_layers):
            self.lstms.append(LayerNormLSTMCell(hidden_size, hidden_size))

        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2.weight.data.mul_(0.1)
        self.linear2.bias.data.fill_(0.0)

    def cuda(self, cuda_id = 0):
        super(MetaOptimizer, self).cuda(cuda_id)
        for i in range(len(self.lstms)):
            self.lstms[i].cuda(cuda_id)

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False, cuda_id = 0):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            for i in range(len(self.lstms)):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
        else:
            self.hx = []
            self.cx = []
            for i in range(len(self.lstms)):
                self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
                if use_cuda:
                    self.hx[i], self.cx[i] = self.hx[i].cuda(cuda_id), self.cx[i].cuda(cuda_id)

    def forward(self, x):
        # Gradients preprocessing
        x = F.tanh(self.ln1(self.linear1(x)))

        for i in range(len(self.lstms)):
            if x.size(0) > self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x.size(0), self.cx[i].size(1))
            if x.size(0) < self.hx[i].size(0):
                x, _ = self.lstms[i](x, (self.hx[i][:x.size(0),:], self.cx[i][:x.size(0),:]))
                x = self.linear2(x)
                return x.squeeze()
            self.hx[i], self.cx[i] = self.lstms[i](x, (self.hx[i], self.cx[i]))
            x = self.hx[i]

        x = self.linear2(x)
        #print x
        return x.squeeze()

    def batch_forward(self, x, batch_size = 1):
        #for large parameter size and low GPU memory
        x_batch = []
        step = np.int64(x.size(0) / batch_size)
        for i in range(step):
            ans = self.forward(Variable(x[i*batch_size:(i+1)*batch_size,:]))
            x_batch.append(ans)
        #in = torch.zeros((batch_size, x.size(1)))
        #in[:, :] = x[step * batch_size:,:]
        ans = self.forward(Variable(x[step * batch_size:, :]))
        x_batch.append(ans)
        return torch.cat(x_batch)
       
        
    def meta_update(self, model_with_grads, loss, skip = [], lr = 0.01):
        # First we need to create a flat version of parameters and gradients
        grads = []
        skip_params = []

        tag = 0
        for module in model_with_grads.children():
            if tag in skip:
                skip_params.append(-lr * module._parameters['weight'].grad.data)
                if 'bias' in module._parameters:
                    skip_params.append(-lr * module._parameters['bias'].grad.data)
                tag += 1
                continue
            grads.append(module._parameters['weight'].grad.data.view(-1))
            if 'bias' in module._parameters:
                grads.append(module._parameters['bias'].grad.data.view(-1))
            tag += 1

        flat_params, skip_params = self.meta_model.get_flat_params(skip, skip_params)
        flat_grads = preprocess_gradients(torch.cat(grads))
        inputs = Variable(torch.cat((flat_grads.view(-1,2), flat_params.data.view(-1,1)), 1))
        #print inputs.size()

        # Meta update itself
        flat_params = flat_params + self.forward(inputs)

        self.meta_model.set_flat_params(flat_params, skip, lr, skip_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model

class FastMetaOptimizer(nn.Module):

    def __init__(self, model, num_layers, hidden_size):
        super(FastMetaOptimizer, self).__init__()
        self.meta_model = model

        self.linear1 = nn.Linear(6, 2)
        self.linear1.bias.data[0] = 1

    def forward(self, x):
        # Gradients preprocessing
        x = F.sigmoid(self.linear1(x))
        return x.split(1, 1)

    def reset_lstm(self, keep_states=False, model=None, use_cuda=False, cuda_id = 0):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)

        if keep_states:
            self.f = Variable(self.f.data)
            self.i = Variable(self.i.data)
        else:
            self.f = Variable(torch.zeros(1, 1))
            self.i = Variable(torch.zeros(1, 1))
            if use_cuda:
                self.f = self.f.cuda(cuda_id)
                self.i = self.i.cuda(cuda_id)

    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []

        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.data.view(-1))
            if 'bias' in module._parameters:
                grads.append(module._parameters['bias'].grad.data.view(-1))

        flat_params = self.meta_model.get_flat_params()
        flat_grads = torch.cat(grads)

        self.i = self.i.expand(flat_params.size(0), 1)
        self.f = self.f.expand(flat_params.size(0), 1)

        loss = loss.expand_as(flat_grads)
#        print preprocess_gradients(flat_grads).size(), flat_params.data.view((-1,1)).size(), loss.view((-1,1)).size()
        inputs = Variable(torch.cat((preprocess_gradients(flat_grads.view((-1,1))), flat_params.data, loss), 1))
        inputs = torch.cat((inputs, self.f, self.i), 1)
        self.f, self.i = self(inputs)

        #print self.i
        # Meta update itself
        flat_params = self.f * flat_params - self.i * Variable(flat_grads)
        #print self.f.size(), flat_params.size()

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model

# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and applying updates to
# them.


class MetaModel:

    def __init__(self, model):
        self.model = model

    def reset(self):
        for module in self.model.children():
            module._parameters['weight'] = Variable(
                module._parameters['weight'].data)
            if 'bias' in module._parameters:
                module._parameters['bias'] = Variable(
                    module._parameters['bias'].data)

    def get_flat_params(self, skip = [], skip_params = []):
        params = []
        tag = 0
        seq = 0
        for module in self.model.children():
            if tag in skip:
                skip_params[seq] += module._parameters['weight'].data
                seq += 1
                if 'bias' in module._parameters:
                    skip_params[seq] += module._parameters['bias'].data
                    seq += 1
                tag += 1
                continue
            params.append(module._parameters['weight'].view(-1))
            if 'bias' in module._parameters:
                params.append(module._parameters['bias'].view(-1))
            tag += 1
        return torch.cat(params), skip_params

    def set_flat_params(self, flat_params, skip = [], lr = 0.01, skip_params = []):
        # Restore original shapes
        offset = 0
        seq = 0
        for i, module in enumerate(self.model.children()):
            if i in skip:
                module._parameters['weight'].data = skip_params[seq]
                seq += 1
                if 'bias' in module._parameters:
                    module._parameters['bias'].data = skip_params[seq]
                    seq += 1
                continue

            weight_shape = module._parameters['weight'].size()
            weight_flat_size = reduce(mul, weight_shape, 1)

            #print weight_flat_size, weight_shape
            #print flat_params[offset:offset + weight_flat_size].size()

            module._parameters['weight'] = flat_params[
                offset:offset + weight_flat_size].view(weight_shape)

            bias_flat_size = 0
            if 'bias' in module._parameters:
                bias_shape = module._parameters['bias'].size()
                bias_flat_size = reduce(mul, bias_shape, 1)
                module._parameters['bias'] = flat_params[
                    offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(bias_shape)

            offset += weight_flat_size + bias_flat_size

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)
