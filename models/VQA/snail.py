import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttOutput(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttOutput, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.2)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Conv1dWrap(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True, causal=False):
        super(Conv1dWrap, self).__init__()
        self.causal = causal
        self.dilation = dilation
        self.padding = dilation * round((kernel_size - 1)/2)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                                padding=self.padding, dilation=self.dilation, groups=groups, bias=bias)

    def forward(self, input):
        # Takes something of shape (N, in_channels, T),
        # returns (N, out_channels, T)
        out = self.conv1d(input)    # weight_norm(
        if not self.causal:
            return out
        return out[:, :, :-self.padding] # causal

class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation, filters, kernel_size=3, mode=False):
        super(DenseBlock, self).__init__()
        self.causalconv1 = Conv1dWrap(in_channels, filters, kernel_size, dilation=dilation, causal=False)
        self.causalconv2 = Conv1dWrap(in_channels, filters, kernel_size, dilation=dilation, causal=False)
        self.LayerNorm = LayerNorm(in_channels)
        self.mode = mode

    def forward(self, input):
        # input is dimensions (N, in_channels, T)
        xf = self.causalconv1(input)
        xg = self.causalconv2(input)
        activations = torch.tanh(xf) * torch.sigmoid(xg) # shape: (N, filters, T)
        if self.mode:
            x = input + activations
            x = torch.transpose(x, 1, 2)
            x = self.LayerNorm(x)
            x = torch.transpose(x, 1, 2)
        else:
            return torch.cat((input, activations), dim=1)

class TCBlock(nn.Module):
    def __init__(self, in_channels, seq_length, filters, mode=False):
        super(TCBlock, self).__init__()
        self.mode = mode
        if mode:
            if False:
                layer_count = int(math.ceil(math.log(seq_length, 2)))
                blocks = []
                for layer in range(layer_count):
                    block = DenseBlock(in_channels, 2**layer, filters, mode=mode)
                    blocks.append(block)
                self.blocks = nn.Sequential(*blocks)
            else:
                self.dense_blocks = nn.ModuleList([DenseBlock(in_channels, 2 ** (i+1), in_channels, mode=mode)
                                                   for i in range(int(math.ceil(math.log(seq_length+1, 2))))])
        else:
            self.dense_blocks = nn.ModuleList([DenseBlock(in_channels + i * filters, 2 ** i, filters, mode=mode)
                                               for i in range(int(math.ceil(math.log(seq_length*2, 2))))])
        self.out_channels=in_channels + int(math.ceil(math.log(seq_length*2, 2))) * filters;
    
    def forward(self, input):
        #print(('TC input',input.shape))
        # input is dimensions (N, T, in_channels)
        input = torch.transpose(input, 1, 2)
        if False:
            input = self.blocks(input)
        else:
            for block in self.dense_blocks:
                input = block(input)
                #print(('TC block',input.shape))
        #print(('TC final',input.shape))
        return torch.transpose(input, 1, 2)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size, output_attentions=False):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)
        self.output = SelfAttOutput(in_channels, value_size)
        self.output_attentions = output_attentions

    def forward(self, input, attention_mask):
        #print(('input',input.shape))
        #print(('attention_mask',attention_mask.shape))
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # the sequence length
        # mask = np.array([[1 if i>j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        # mask = torch.ByteTensor(mask).cuda()
        extended_attention_mask = attention_mask.unsqueeze(1)   # N, 1, T
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0 #-float('inf') #-10000.0
        # extended_attention_mask[torch.isnan(extended_attention_mask)] = 0

        #import pdb; pdb.set_trace()
        keys = self.linear_keys(input) # shape: (N, T, key_size)
        query = self.linear_query(input) # shape: (N, T, key_size)
        values = self.linear_values(input) # shape: (N, T, value_size)
        #print(('keys',keys.shape))
        #print(('query',query.shape))
        #print(('values',values.shape))
        attention_scores = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        #print(('attention_scores',attention_scores.shape))
        #print(('extended_attention_mask',extended_attention_mask.shape))

        attention_scores = attention_scores + extended_attention_mask
        # temp.data.masked_fill_(mask, -float('inf'))

        attention_probs = F.softmax(attention_scores / self.sqrt_key_size, dim=1) # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        #print(('attention_probs',attention_probs.shape))
        self_output = torch.bmm(attention_probs, values) # shape: (N, T, value_size)
        #print(('self_output',self_output.shape))

        if False:
            attention_output = self.output(self_output, input)
        else:
            attention_output = torch.cat((input, self_output), dim=2) # shape: (N, T, in_channels + value_size)

        #print(('attention_output',attention_output.shape))
        if self.output_attentions:
                return attention_probs, attention_output
        return attention_output


class SnailLayer(nn.Module):
    def __init__(self, hidden_size, seq_length, output_attentions=False):
        super(SnailLayer, self).__init__()
        self.output_attentions = output_attentions
        self.tc = TCBlock(hidden_size, seq_length, hidden_size, mode=True)
        self.attention = AttentionBlock(hidden_size, hidden_size, hidden_size) # (N,T,H)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.tc(hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        if self.output_attentions:
            attentions, attention_output = attention_output
        if self.output_attentions:
            return attentions, attention_output
        return attention_output


class SnailEncoder(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, seq_length, output_attentions=False):
        super(SnailEncoder, self).__init__()
        self.output_attentions = output_attentions
        # layer = SnailLayer(hidden_size, seq_length, output_attentions=output_attentions)
        # self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_hidden_layers)])
        # self.attention1 = AttentionBlock(hidden_size, hidden_size, hidden_size)

        num_channels = hidden_size
        num_filters = int(math.ceil(math.log(seq_length + 1, 2)))
        self.attention1 = AttentionBlock(num_channels, 192, 96, output_attentions=output_attentions) # num_channels, 64 32
        num_channels += 96
        self.tc1 = TCBlock(num_channels, seq_length + 1, 64) # 128
        num_channels += num_filters * 64
        self.attention2 = AttentionBlock(num_channels, 384, 192, output_attentions=output_attentions) # num_channels, 256, 128
        num_channels += 192
        self.tc2 = TCBlock(num_channels, seq_length + 1, 128)
        num_channels += num_filters * 128
        self.attention3 = AttentionBlock(num_channels, 768, 384, output_attentions=output_attentions) # num_channels, 512, 256
        num_channels += 384

        self.dense = nn.Linear(num_channels, hidden_size)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, input, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_attentions = []
        if True:
            x = self.attention1(input, attention_mask)
            if self.output_attentions:
                attentions, x = x
                all_attentions.append(attentions)
            if output_all_encoded_layers:
                all_encoder_layers.append(x)
            x = self.tc1(x)
            x = self.attention2(x, attention_mask)
            if self.output_attentions:
                attentions, x = x
                all_attentions.append(attentions)
            if output_all_encoded_layers:
                all_encoder_layers.append(x)
            x = self.tc2(x)
            x = self.attention3(x, attention_mask)
            if self.output_attentions:
                attentions, x = x
                all_attentions.append(attentions)
            x = self.dense(x)
            # x = self.dropout(x)
            all_encoder_layers.append(x)

            if self.output_attentions:
                return all_attentions, all_encoder_layers
            return all_encoder_layers
        else:
            # attention first layer
            hidden_states = self.attention1(input, attention_mask)
            if self.output_attentions:
                attentions, hidden_states = hidden_states
                all_attentions.append(attentions)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
            # repeating attention layers
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask)
                if self.output_attentions:
                    attentions, hidden_states = hidden_states
                    all_attentions.append(attentions)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            if not output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
            if self.output_attentions:
                return all_attentions, all_encoder_layers
            return all_encoder_layers
        

class SnailEncoder_new(nn.Module):
    def __init__(self, num_hidden_layers, hidden_size, seq_length, output_attentions=False, reduce_input=False):
        super(SnailEncoder_new, self).__init__()
        self.output_attentions = output_attentions
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size_ori = hidden_size
        self.hidden_size = hidden_size
        self.reduce_input = reduce_input
        # layer = SnailLayer(hidden_size, seq_length, output_attentions=output_attentions)
        # self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_hidden_layers)])
        # self.attention1 = AttentionBlock(hidden_size, hidden_size, hidden_size)
        
        self.reduce_input_dim = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size//2), nn.Tanh())
        
        if self.reduce_input:
            self.hidden_size = self.hidden_size//2
        
        filter_size = (torch.Tensor([1/64, 1/32, 1/16, 1/8, 1/4, 1/2]) * self.hidden_size ).long().tolist();
        num_tc_layers = int(math.ceil(math.log(seq_length*2, 2)))
        num_channels = self.hidden_size
        
        if self.num_hidden_layers == 6:
            num_channels = self.hidden_size
        #print(('attention1',num_channels));
        self.attention1 = AttentionBlock(num_channels, filter_size[0], filter_size[0], output_attentions=output_attentions)
        num_channels += filter_size[0]
        self.tc1 = TCBlock(num_channels, seq_length, filter_size[0])        
        num_channels += num_tc_layers * filter_size[0]
        
        if self.num_hidden_layers == 5:
            num_channels = self.hidden_size
        #print(('attention2',num_channels));
        self.attention2 = AttentionBlock(num_channels, filter_size[1], filter_size[1], output_attentions=output_attentions)
        num_channels += filter_size[1]
        self.tc2 = TCBlock(num_channels, seq_length, filter_size[1])
        num_channels += num_tc_layers * filter_size[1]
        
        if self.num_hidden_layers == 4:
            num_channels = self.hidden_size
        #print(('attention3',num_channels));
        self.attention3 = AttentionBlock(num_channels, filter_size[2], filter_size[2], output_attentions=output_attentions)
        num_channels += filter_size[2]
        self.tc3 = TCBlock(num_channels, seq_length, filter_size[2])
        num_channels += num_tc_layers * filter_size[2]
        
        if self.num_hidden_layers == 3:
            num_channels = self.hidden_size    
        #print(('attention4',num_channels));        
        self.attention4 = AttentionBlock(num_channels, filter_size[3], filter_size[3], output_attentions=output_attentions)
        num_channels += filter_size[3]
        self.tc4 = TCBlock(num_channels, seq_length, filter_size[3])
        num_channels += num_tc_layers * filter_size[3]
        
        #print(('attention5',num_channels));     
        self.attention5 = AttentionBlock(num_channels, filter_size[4], filter_size[4], output_attentions=output_attentions)
        num_channels += filter_size[4]
        self.tc5 = TCBlock(num_channels, seq_length, filter_size[4])
        num_channels += num_tc_layers * filter_size[4]               
        
        #print(('attention6',num_channels));  
        self.attention6 = AttentionBlock(num_channels, filter_size[5], filter_size[5], output_attentions=output_attentions)
        num_channels += filter_size[5]        
        
        self.output = SelfAttOutput(num_channels, self.hidden_size_ori)
        self.reduce_output_dim = nn.Sequential(nn.Linear(num_channels, self.hidden_size_ori), nn.Tanh())
        self.dropout = nn.Dropout(0.2)

    def forward(self, input, attention_mask, output_all_encoded_layers=False):
        # output_all_encoded_layers: get all outputs of attention layers
        all_encoder_layers = []
        all_attentions = []
        
        if self.reduce_input:
            x = self.reduce_input_dim(input)
        else:
            x = input
        
        if self.num_hidden_layers >= 6:
            x = self.attention1(x, attention_mask)
            if self.output_attentions:
                attentions, x = x
                all_attentions.append(attentions)
            if output_all_encoded_layers:
                all_encoder_layers.append(x)
            x = self.tc1(x)
            
        if self.num_hidden_layers >= 5:
            #print('layer5')
            #print(x.shape)
            x = self.attention2(x, attention_mask)
            #print(x.shape)
            if self.output_attentions:
                attentions, x = x
                all_attentions.append(attentions)
            if output_all_encoded_layers:
                all_encoder_layers.append(x)
            #print(x.shape)
            x = self.tc2(x)
            #print(x.shape)
            
        if self.num_hidden_layers >= 4:
            #print('layer4')
            #print(x.shape)
            x = self.attention3(x, attention_mask)
            #print(x.shape)
            if self.output_attentions:
                attentions, x = x
                all_attentions.append(attentions)
            if output_all_encoded_layers:
                all_encoder_layers.append(x)
            #print(x.shape)
            x = self.tc3(x)
            #print(x.shape)
        
        if self.num_hidden_layers >= 3:
            x = self.attention4(x, attention_mask)
            if self.output_attentions:
                attentions, x = x
                all_attentions.append(attentions)
            if output_all_encoded_layers:
                all_encoder_layers.append(x)
            x = self.tc4(x)
        
        x = self.attention5(x, attention_mask)
        if self.output_attentions:
            attentions, x = x
            all_attentions.append(attentions)
        if output_all_encoded_layers:
            all_encoder_layers.append(x)
        x = self.tc5(x)
        
        x = self.attention6(x, attention_mask)
        if self.output_attentions:
            attentions, x = x
            all_attentions.append(attentions)
        
        x = self.reduce_output_dim(x)        
        x = self.dropout(x)
        all_encoder_layers.append(x)
        
        #XIAO: I don't know what this is doing but it breaks the code, disabled
        #x = self.output(x)
        #all_encoder_layers.append(x)

        if self.output_attentions:
            return all_attentions, all_encoder_layers
        return all_encoder_layers
