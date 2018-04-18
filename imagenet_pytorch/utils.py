import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn import init

class attention(nn.Module):
    def __init__(self, input_channels, map_size):
        super(attention, self).__init__()

        self.pool = nn.AvgPool2d(kernel_size = map_size)
        self.fc1 = nn.Linear(in_features = input_channels,out_features = input_channels // 2)
        self.fc2 = nn.Linear(in_features = input_channels // 2, out_features = input_channels)


    def forward(self, x):
        output = self.pool(x)
        output = output.view(output.size()[0], output.size()[1])
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.sigmoid(output)
        output = output.view(output.size()[0],output.size()[1],1,1)
        output = torch.mul(x, output)
        return output


class transition(nn.Module):
    def __init__(self, if_att, current_size, input_channels, keep_prob):
        super(transition, self).__init__()
        self.input_channels = input_channels
        self.keep_prob = keep_prob
        self.bn = nn.BatchNorm2d(self.input_channels)
        self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size = 1, bias = False)
        # self.dropout = nn.Dropout2d(1 - self.keep_prob)
        self.pool = nn.AvgPool2d(kernel_size = 2)
        self.if_att = if_att
        if self.if_att == True:
            self.attention = attention(input_channels = self.input_channels, map_size = current_size)

    def forward(self, x):
        output = self.bn(x)
        output = F.relu(output)
        output = self.conv(output)
        if self.if_att==True:
            output = self.attention(output)
        # output = self.dropout(output)
        output = self.pool(output)
        return output

class global_pool(nn.Module):
    def __init__(self, input_size, input_channels):
        super(global_pool, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.bn = nn.BatchNorm2d(self.input_channels)
        self.pool = nn.AvgPool2d(kernel_size = self.input_size)

    def forward(self, x):
        output = self.bn(x)
        output = F.relu(output)
        output = self.pool(output)
        return output

class compress(nn.Module):
    def __init__(self, input_channels, keep_prob):
        super(compress, self).__init__()
        self.keep_prob = keep_prob
        self.bn = nn.BatchNorm2d(input_channels)
        self.conv = nn.Conv2d(input_channels, input_channels//2, kernel_size = 1, padding = 0, bias = False)


    def forward(self, x):
        output = self.bn(x)
        output = F.relu(output)
        output = self.conv(output)
        # output = F.dropout2d(output, 1 - self.keep_prob)
        return output

class clique_block(nn.Module):
    def __init__(self, input_channels, channels_per_layer, layer_num, loop_num, keep_prob):
        super(clique_block, self).__init__()
        self.input_channels = input_channels
        self.channels_per_layer = channels_per_layer
        self.layer_num = layer_num
        self.loop_num = loop_num
        self.keep_prob = keep_prob

        # conv 1 x 1
        self.conv_param = nn.ModuleList([nn.Conv2d(self.channels_per_layer, self.channels_per_layer, kernel_size = 1, padding = 0, bias = False)
                                   for i in range((self.layer_num + 1) ** 2)])

        for i in range(1, self.layer_num + 1):
            self.conv_param[i] = nn.Conv2d(self.input_channels, self.channels_per_layer, kernel_size = 1, padding = 0, bias = False)
        for i in range(1, self.layer_num + 1):
            self.conv_param[i * (self.layer_num + 2)] = None
        for i in range(0, self.layer_num + 1):
            self.conv_param[i * (self.layer_num + 1)] = None

        self.forward_bn = nn.ModuleList([nn.BatchNorm2d(self.input_channels + i * self.channels_per_layer) for i in range(self.layer_num)])
        self.forward_bn_b = nn.ModuleList([nn.BatchNorm2d(self.channels_per_layer) for i in range(self.layer_num)])
        self.loop_bn = nn.ModuleList([nn.BatchNorm2d(self.channels_per_layer * (self.layer_num - 1)) for i in range(self.layer_num)])
        self.loop_bn_b = nn.ModuleList([nn.BatchNorm2d(self.channels_per_layer) for i in range(self.layer_num)])

        # conv 3 x 3
        self.conv_param_bottle = nn.ModuleList([nn.Conv2d(self.channels_per_layer, self.channels_per_layer, kernel_size = 3, padding = 1, bias = False)
                                   for i in range(self.layer_num)])


    def forward(self, x):
        # key: 1, 2, 3, 4, 5, update every loop
        self.blob_dict={}
        # save every loops results
        self.blob_dict_list=[]

        # first forward
        for layer_id in range(1, self.layer_num + 1):
            bottom_blob = x
            # bottom_param = self.param_dict['0_' + str(layer_id)]

            bottom_param = self.conv_param[layer_id].weight
            for layer_id_id in range(1, layer_id):
                # pdb.set_trace()
                bottom_blob = torch.cat((bottom_blob, self.blob_dict[str(layer_id_id)]), 1)
                # bottom_param = torch.cat((bottom_param, self.param_dict[str(layer_id_id) + '_' + str(layer_id)]), 1)
                bottom_param = torch.cat((bottom_param, self.conv_param[layer_id_id * (self.layer_num + 1) + layer_id].weight), 1)
            next_layer = self.forward_bn[layer_id - 1](bottom_blob)
            next_layer = F.relu(next_layer)
            # conv 1 x 1
            next_layer = F.conv2d(next_layer, bottom_param, stride = 1, padding = 0)
            # conv 3 x 3
            next_layer = self.forward_bn_b[layer_id - 1](next_layer)
            next_layer = F.relu(next_layer)
            next_layer = F.conv2d(next_layer, self.conv_param_bottle[layer_id - 1].weight, stride = 1, padding = 1)
            # next_layer = F.dropout2d(next_layer, 1 - self.keep_prob)
            self.blob_dict[str(layer_id)] = next_layer
        self.blob_dict_list.append(self.blob_dict)

        # loop
        for loop_id in range(self.loop_num):
            for layer_id in range(1, self.layer_num + 1):

                layer_list = [l_id for l_id in range(1, self.layer_num + 1)]
                layer_list.remove(layer_id)

                bottom_blobs = self.blob_dict[str(layer_list[0])]
                # bottom_param = self.param_dict[layer_list[0] + '_' + str(layer_id)]
                bottom_param = self.conv_param[layer_list[0] * (self.layer_num + 1) + layer_id].weight
                for bottom_id in range(len(layer_list) - 1):
                    bottom_blobs = torch.cat((bottom_blobs, self.blob_dict[str(layer_list[bottom_id + 1])]), 1)
                    # bottom_param = torch.cat((bottom_param, self.param_dict[layer_list[bottom_id+1]+'_'+str(layer_id)]), 1)
                    bottom_param = torch.cat((bottom_param, self.conv_param[layer_list[bottom_id + 1] * (self.layer_num + 1) + layer_id].weight), 1)
                bottom_blobs = self.loop_bn[layer_id - 1](bottom_blobs)
                bottom_blobs = F.relu(bottom_blobs)
                # conv 1 x 1
                mid_blobs = F.conv2d(bottom_blobs, bottom_param, stride = 1, padding = 0)
                # conv 3 x 3
                top_blob = self.loop_bn_b[layer_id - 1](mid_blobs)
                top_blob = F.relu(top_blob)
                top_blob = F.conv2d(top_blob, self.conv_param_bottle[layer_id - 1].weight, stride = 1, padding = 1)
                self.blob_dict[str(layer_id)] = top_blob
            self.blob_dict_list.append(self.blob_dict)

        assert len(self.blob_dict_list) == 1 + self.loop_num

        # output
        block_feature_I = self.blob_dict_list[0]['1']
        for layer_id in range(2, self.layer_num + 1):
            block_feature_I = torch.cat((block_feature_I, self.blob_dict_list[0][str(layer_id)]), 1)
        block_feature_I = torch.cat((x, block_feature_I), 1)

        block_feature_II = self.blob_dict_list[self.loop_num]['1']
        for layer_id in range(2, self.layer_num + 1):
            block_feature_II = torch.cat((block_feature_II, self.blob_dict_list[self.loop_num][str(layer_id)]), 1)
        return block_feature_I, block_feature_II
