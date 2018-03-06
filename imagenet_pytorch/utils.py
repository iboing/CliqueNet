import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn import init


class transition(nn.Module):
    def __init__(self, input_channels, keep_prob):
        super(transition, self).__init__()
        self.input_channels = input_channels
        self.keep_prob = keep_prob
        self.bn = nn.BatchNorm2d(self.input_channels)
        self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size = 1, bias = False)
        # self.dropout = nn.Dropout2d(1 - self.keep_prob)
        self.pool = nn.AvgPool2d(kernel_size = 2)

    def forward(self, x):
        output = self.bn(x)
        output = F.relu(output)
        output = self.conv(output)
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
                # self.blob_dict[str(layer_id)] = F.dropout2d(top_blob, self.keep_prob)
            self.blob_dict_list.append(self.blob_dict)
        
        assert len(self.blob_dict_list) == 1 + self.loop_num

        # output
        block_feature_I = self.blob_dict_list[0]['1']
        for layer_id in range(2, self.layer_num + 1):
            block_feature_I = torch.cat((block_feature_I, self.blob_dict_list[0][str(layer_id)]), 1)
        block_feature_I = torch.cat((x, block_feature_I), 1)
        
        block_feature_II = self.blob_dict_list[self.loop_num]['1']
        for layer_id in range(2, self.layer_num + 1):
            block_feature_II = torch.cat((block_feature_II, self.blob_dict_list[1][str(layer_id)]), 1)    
        return block_feature_I, block_feature_II

class CliqueNet(nn.Module):
    def __init__(self, input_channels, list_channels, list_layer_num):
        super(CliqueNet, self).__init__()
        self.fir_trans = nn.Conv2d(3, input_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.fir_bn = nn.BatchNorm2d(input_channels)
        self.fir_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.block_num = len(list_channels)

        self.list_block = []
        self.list_trans = []   
        self.list_gb = []
        self.list_gb_channel = []
        self.list_compress = []
        input_size_init = 56
        for i in xrange(len(block_num)):
            if i == 0:
                self.list_block.append(Clique_block(input_channels = input_channels, channels_per_layer = list_channels[0], layer_num = list_layer_num[0], loop_num = 1, keep_prob = 0.8))
                self.list_gb_channel.append(input_channels + list_channels[0] * list_layer_num[0])
            else :
                self.list_block.append(Clique_block(input_channels = list_channels[i-1] * list_layer_num[i-1], channels_per_layer = list_channels[i], layer_num = list_layer_num[i], loop_num = 1, keep_prob = 0.8))
                self.list_gb_channel.append(list_channels[i-1] * list_layer_num[i-1] + list_channels[i] * list_layer_num[i])
            self.list_trans.append(transition(input_channels = list_channels[i] * list_layer_num[i], keep_prob = 0.8))
            self.list_gb.append(global_pool(input_size = input_size_init, input_channels = list_gb_channel[i] // 2))
            self.list_compress.append(compress(input_channels = list_gb_channel[i], keep_prob = 0.8))
            input_size_init = input_size_init // 2


        # self.block1 = Clique_block(input_channels = input_channels, channels_per_layer = list_channels[0], layer_num = list_layer_num[0], loop_num = 1, keep_prob = 0.8)
        # self.block2 = Clique_block(input_channels = list_channels[0] * list_layer_num[0], channels_per_layer = list_channels[1], layer_num = list_layer_num[1], loop_num = 1, keep_prob = 0.8)
        # self.block3 = Clique_block(input_channels = list_channels[1] * list_layer_num[1], channels_per_layer = list_channels[2], layer_num = list_layer_num[2], loop_num = 1, keep_prob = 0.8)
        # self.block4 = Clique_block(input_channels = list_channels[2] * list_layer_num[2], channels_per_layer = list_channels[3], layer_num = list_layer_num[3], loop_num = 1, keep_prob = 0.8)
        
        # self.trans1 = transition(input_channels = list_channels[0] * list_layer_num[0], keep_prob = 0.8)
        # self.trans2 = transition(input_channels = list_channels[1] * list_layer_num[1], keep_prob = 0.8)
        # self.trans3 = transition(input_channels = list_channels[2] * list_layer_num[2], keep_prob = 0.8)

        # gl_channels1 = input_channels + list_channels[0] * list_layer_num[0]
        # self.gp1 = global_pool(input_size = 56, input_channels = gl_channels1 // 2)
        # gl_channels2 = list_channels[0] * list_layer_num[0] + list_channels[1] * list_layer_num[1]
        # self.gp2 = global_pool(input_size = 28, input_channels = gl_channels2 // 2)
        # gl_channels3 = list_channels[1] * list_layer_num[1] + list_channels[2] * list_layer_num[2]
        # self.gp3 = global_pool(input_size = 14, input_channels = gl_channels3 // 2)
        # gl_channels4 = list_channels[2] * list_layer_num[2] + list_channels[3] * list_layer_num[3]
        # self.gp4 = global_pool(input_size = 7, input_channels = gl_channels4 // 2)

        # self.comp1 = compress(input_channels = gl_channels1, keep_prob = 0.8)
        # self.comp2 = compress(input_channels = gl_channels2, keep_prob = 0.8)
        # self.comp3 = compress(input_channels = gl_channels3, keep_prob = 0.8)
        # self.comp4 = compress(input_channels = gl_channels4, keep_prob = 0.8)

        self.fc = nn.Linear(in_features = (gl_channels1 + gl_channels2 + gl_channels3 + gl_channels4) // 2, out_features = 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        output = self.fir_trans(x)
        output = self.fir_bn(output)
        output = F.relu(output)
        output = self.fir_pool(output)

        feature_I_list = []


        # use stage II + stage II mode 
        for i in xrange(self.block_num):
            block_feature_I, block_feature_II = self.list_block[i](output)
            block_feature_I = self.list_compress[i](block_feature_I)
            feature_I_list.append(self.list_gb[i](block_feature_I))
            output = self.list_trans[i](block_feature_II)


        # block_feature_I, block_feature_II = self.block1(output)
        # block_feature_I = self.comp1(block_feature_I)
        # feature_I_list.append(self.gp1(block_feature_I))
        # trans = self.trans1(block_feature_II)

        # block_feature_I, block_feature_II = self.block2(trans)
        # block_feature_I = self.comp2(block_feature_I)
        # feature_I_list.append(self.gp2(block_feature_I))
        # trans = self.trans2(block_feature_II)

        # block_feature_I, block_feature_II = self.block3(trans)
        # block_feature_I = self.comp3(block_feature_I)
        # feature_I_list.append(self.gp3(block_feature_I))
        # trans = self.trans3(block_feature_II)

        # block_feature_I, block_feature_II = self.block4(trans)
        # block_feature_I = self.comp4(block_feature_I)
        # feature_I_list.append(self.gp4(block_feature_I))

        final_feature = feature_I_list[0]
        for block_id in range(1, len(feature_I_list)):
            final_feature=torch.cat((final_feature, feature_I_list[block_id]), 1)
        
        final_feature = final_feature.view(final_feature.size()[0], final_feature.size()[1])
        output = self.fc(final_feature)
        return output