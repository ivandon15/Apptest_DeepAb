import torch
import torch.nn as nn
import torch.nn.functional as F

from apptest.util.utils import get_matrix_S, pad_data_to_same_shape
# https://github.com/linzehui/Gated-Convolutional-Networks
# https://github.com/jojonki/Gated-Convolutional-Networks 有图片
class GatedCNN(nn.Module):
    def __init__(self, embed_dim,
                 kernel_width, out_channel, n_layers,
                 res_block_cnt, dropout=0.5):
        super(GatedCNN, self).__init__()
        self.res_block_cnt = res_block_cnt
        self.padding_left = nn.ConstantPad1d((kernel_width - 1, 0), 0)
        self.conv_0 = nn.Conv1d(in_channels=embed_dim, out_channels=out_channel,
                                kernel_size=kernel_width,dtype=torch.float)
        self.b_0 = nn.Parameter(torch.zeros(out_channel, 1))  # same as paper
        self.conv_gate_0 = nn.Conv1d(in_channels=embed_dim, out_channels=out_channel,
                                     kernel_size=kernel_width)
        self.c_0 = nn.Parameter(torch.zeros(out_channel, 1))

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                                              kernel_size=kernel_width)
                                    for _ in range(n_layers)])

        self.bs = nn.ParameterList([nn.Parameter(torch.zeros(out_channel, 1))  # collections of b
                                    for _ in range(n_layers)])

        self.conv_gates = nn.ModuleList([nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                                                   kernel_size=kernel_width)
                                         for _ in range(n_layers)])

        self.cs = nn.ParameterList([nn.Parameter(torch.zeros(out_channel, 1))
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(p=dropout)  # todo use dropout

    # conv1d Input: (N, Cin, Lin)
    # constantpad1d Input: (N,C,Win)  Output: (N,C,Wout)

    def forward(self, x):
        x.transpose_(1, 2)  # x:(batch,embed_dim,seq_len) , embed_dim equals to in_channel
        x = self.padding_left(x)
        A = self.conv_0(x.type(torch.float))  # A: (batch,out_channel,seq_len)   seq_len because of padding (kernel-1)
        print(f"A shape: {A.shape}")

        A += self.b_0  # b_0 broadcast
        B = self.conv_gate_0(x.type(torch.float))  # B: (batch,out_channel,seq_len)
        print(f"B shape: {B.shape}")

        B += self.c_0

        h = A * F.sigmoid(B)  # h: (batch,out_channel,seq_len)
        # todo: add resnet
        res_input = h

        for i, (conv, conv_gate) in enumerate(zip(self.convs, self.conv_gates)):
            h = self.padding_left(h)
            A = conv(h) + self.bs[i]
            B = conv_gate(h) + self.cs[i]
            h = A * F.sigmoid(B)  # h: (batch,out_channel,seq_len+kernel-1)
            if i % self.res_block_cnt == 0:  # todo Is this correct?
                h += res_input
                res_input = h

        h.transpose_(1, 2)  # h:(batch,seq_len,out_channel)

        # logic = self.fc(h)  # logic:(batch,seq_len,vocab_size)
        # logic.transpose_(1,2)  # logic:(batch,vocab_size,seq_len) cross_entropy input:(N,C,d1,d2,..) C is num of class
        return h


# if __name__ == '__main__':
#     model = GatedCNN(embed_dim=50,
#                      kernel_width=3, out_channel=32,
#                      n_layers=12, res_block_cnt=5)
#
#     # input = torch.LongTensor([[1, 2, 3],
#     #                           [4, 5, 6]])
#     Xset = [get_matrix_S('AYWGC'), get_matrix_S('VGWG')]
#     Xset = pad_data_to_same_shape(Xset)
#     print("Xset[0].shape")
#     print(Xset.shape)
#     print(model(Xset).shape)
#     print(model)