import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from apptest.dataset.SeqDataSet import SeqDataset
from apptest.util.utils import get_matrix_S, pad_data_to_same_shape, get_masked_mat, pad_data_to_max_shape
import numpy as np
import warnings

torch.set_printoptions(threshold=10_000)
warnings.filterwarnings("ignore")
# TODO：phi psi的output是cos 和 sin！！，所以应该输出的 50*2个

# 先将xy都pad，然后直接mse_loss即可
# Xset = [get_matrix_S('AYWGC'), get_matrix_S('VGWG')]
# yset = [torch.randn((5, 5)), torch.LongTensor([[6, 7, 10, 11], [12, 14, 13, 1], [1, 7, 10, 11], [14, 13, 15, 1]])]
# Xset = pad_data_to_same_shape(Xset)
# yset = pad_data_to_same_shape(yset)
# # print(Xset[0])
# data = [[Xset[0], yset[0]], [Xset[1], yset[1]]]
# trainloader = torch.utils.data.DataLoader(data, batch_size=1,
#                                           shuffle=True)


class OuterProduct2D(nn.Module):
    """Transforms sequential data to pairwise data using an outer product. L * 16 -> L * L * 16"""
    """
    sample
    i = torch.randn((2,3,4))
    print(torch.Tensor(i))
    print(torch.Tensor(i)[1,::])
    all_ordered_idx_pairs = torch.cartesian_prod(torch.tensor(range(i.shape[1])),torch.tensor(range(i.shape[1])))
    print(all_ordered_idx_pairs)
    print(i[0][all_ordered_idx_pairs])
    aa = [i[j][all_ordered_idx_pairs] for j in range(i.shape[0])]
    print(torch.mul(aa[0][:,0,:],aa[0][:,1,:]))
    print(torch.stack([torch.mul(aa[j][:,0,:],aa[j][:,1,:]).view(i.shape[1],i.shape[1],-1) for j in range(i.shape[0])]).shape)
    """

    def __init__(self):
        super(OuterProduct2D, self).__init__()

    def forward(self, x: torch.FloatTensor):
        # batch * L * dimension
        # get all the index pairs
        all_ordered_idx_pairs = torch.cartesian_prod(torch.tensor(range(x.shape[1])), torch.tensor(range(x.shape[1])))

        # batch * (L*L) * dimension combination
        expanded = [x[i][all_ordered_idx_pairs] for i in range(x.shape[0])]

        # calculate each batch separately
        # element wise for first line to second line
        expaned_multiply = [torch.mul(expanded[j][:, 0, :], expanded[j][:, 1, :]).view(x.shape[1], x.shape[1], -1) for j
                            in range(x.shape[0])]

        return torch.stack(expaned_multiply)


class GatedCNN(nn.Module):
    def __init__(self, embed_dim,
                 kernel_width, out_channel, n_layers,
                 res_block_cnt, dropout=0.5):
        super(GatedCNN, self).__init__()

        # resnet连接
        self.res_block_cnt = res_block_cnt
        # 防止未来数据泄露
        # self.padding_left = nn.ConstantPad1d((kernel_width - 1, 0), 0)
        # 最初的conv 和 gate，以及可学习的参数（线性变换的常数）
        self.conv_0 = nn.Conv2d(in_channels=embed_dim, out_channels=out_channel,
                                kernel_size=kernel_width, padding=(2, 0), dtype=torch.float)
        self.b_0 = nn.Parameter(torch.randn(1, out_channel, 1, 1))  # same as paper
        self.conv_gate_0 = nn.Conv2d(in_channels=embed_dim, out_channels=out_channel, padding=(2, 0),
                                     kernel_size=kernel_width)
        self.c_0 = nn.Parameter(torch.randn(1, out_channel, 1, 1))

        # 再来n_layers层一模一样的
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                              kernel_size=(kernel_width[0], 1), padding=(2, 0))
                                    for _ in range(n_layers)])
        self.bs = nn.ParameterList([nn.Parameter(torch.randn(1, out_channel, 1, 1))  # collections of b
                                    for _ in range(n_layers)])
        self.conv_gates = nn.ModuleList([nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                                   kernel_size=(kernel_width[0], 1), padding=(2, 0))
                                         for _ in range(n_layers)])
        self.cs = nn.ParameterList([nn.Parameter(torch.randn(1, out_channel, 1, 1))
                                    for _ in range(n_layers)])

        # 一层dropout
        self.dropout = nn.Dropout(p=dropout)  # todo use dropout

    # conv1d Input: (N, Cin, Lin)
    # constantpad1d Input: (N,C,Win)  Output: (N,C,Wout)

    def forward(self, x):
        # x.transpose_(1, 2)  # x:(batch,embed_dim,seq_len) , embed_dim equals to in_channel
        # x = self.padding_left(x)
        A = self.conv_0(x.type(torch.float))  # A: (batch,out_channel,seq_len)   seq_len because of padding (kernel-1)
        A += self.b_0.repeat(1, 1, x.size(1), 1)  # b_0 broadcast
        B = self.conv_gate_0(x.type(torch.float))  # B: (batch,out_channel,seq_len)
        B += self.c_0.repeat(1, 1, x.size(1), 1)

        h = A * torch.sigmoid(B)  # h: (batch,out_channel,seq_len)
        # todo: add resnet
        res_input = h

        for i, (conv, conv_gate) in enumerate(zip(self.convs, self.conv_gates)):
            A = conv(h) + self.bs[i].repeat(1, 1, x.size(1), 1)
            B = conv_gate(h) + self.cs[i].repeat(1, 1, x.size(1), 1)
            h = A * torch.sigmoid(B)  # h: (batch,out_channel,seq_len+kernel-1)
            if i % self.res_block_cnt == 0:  # todo Is this correct?
                h += res_input
                res_input = self.dropout(h)

        # logic = self.fc(h)  # logic:(batch,seq_len,vocab_size)
        # logic.transpose_(1,2)  # logic:(batch,vocab_size,seq_len) cross_entropy input:(N,C,d1,d2,..) C is num of class
        return h


# 默认都pad到50，
class APPTESTModel(nn.Module):
    def __init__(self):
        super(APPTESTModel, self).__init__()

        self.pairwise = OuterProduct2D()
        self.gated_cnn = GatedCNN(embed_dim=50,
                                  kernel_width=(5, 5), out_channel=50, n_layers=12,
                                  res_block_cnt=5, dropout=0.1)
        self.cnn1 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(7, 7), padding="same")
        self.cnn2 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(7, 7), padding="same")
        self.cnn3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(7, 7), padding="same")
        self.cnn4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(7, 7), padding="same")

        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()

        self.fc1 = nn.Linear(50 * 50, 50 * 50)
        self.fc2 = nn.Linear(50 * 50, 50 * 50)

        self.fc3 = nn.Linear(49 * 73, 100)
        self.fc4 = nn.Linear(49 * 73, 100)

    def forward(self, x):
        y = self.pairwise(x.float())
        y = self.gated_cnn(y)
        ca = self.cnn1(y)
        # batch * L * L * dim -> batch * L * L mean
        # TODO：可以直接变成1*2500，然后distance mask也可以变成1*2500，这样有空缺的地方就还是被mask了
        ca = torch.sum(ca, dim=-1).view(ca.size(0), ca.size(1), ca.size(2), 1).squeeze(-1)
        ca = self.flatten1(ca)
        ca = self.fc1(ca)
        ca = F.relu(ca)

        cb = self.cnn2(y)
        cb = torch.mean(cb, dim=-1).view(cb.size(0), cb.size(1), cb.size(2), 1).squeeze(-1)
        cb = self.flatten1(cb)
        cb = self.fc2(cb)
        cb = F.relu(cb)

        # 按理说相邻扭角直接取对角线上移，但不知道咋快速地取
        # x_indices = torch.LongTensor(np.arange(0, 50 - 1)).unsqueeze(-1)
        # y_indices = torch.LongTensor(np.arange(1, 50)).unsqueeze(-1)
        phi = self.cnn3(y)
        # phi = phi[:, x_indices, y_indices, :].squeeze().view(ca.size(0), -1).squeeze()  # (b, seq_len - 1, dim) => (b, (seq_len-1)*dim)
        psi = self.cnn4(y)
        # psi = psi[:, x_indices, y_indices, :].squeeze().view(ca.size(0), -1).squeeze()  # (b, seq_len - 1, dim)

        phi = self.fc3(phi).squeeze(0)
        psi = self.fc3(psi).squeeze(0)

        phi = F.tanh(phi)
        psi = F.tanh(psi)
        return [ca, cb, phi, psi]


def mse_loss(input, target, ignored_index, reduction="mean"):
    mask = (target == ignored_index).squeeze()
    # print(mask)
    # print(input)
    target = target.squeeze()  # for batch size 1
    out = (input[~mask] - target[~mask]) ** 2
    if reduction == "mean":
        return torch.mean(out)

        # return out.mean()
    elif reduction == "None":
        return out

# print(mse_loss(torch.Tensor([[1,2,3,4,5],[1,2,3,4,5]]),torch.Tensor([[2,2,3,-999,-999],[2,2,3,-999,-999]]),-999))


model = APPTESTModel()
trainloader = torch.utils.data.DataLoader(SeqDataset("../../train_data_info.pkl"), batch_size=8)  # shuffle=True

# a = torch.Tensor([[1, 1], [2, 2]])
# b = torch.Tensor([[1, 2, 3], [2, 3, 4], [4, 5, 6]])
# print(a, b)
#
# dataset = pad_data_to_same_shape([a, b])
# a, b = dataset[0], dataset[1]
# print(a, b)
# b = get_masked_mat(b, a)
# print(a)
# print(mse_loss(a, b, ignored_index=-999))

# optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=1e-7)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader)):
        # get the inputs; data is a list of [inputs, labels]
        input_tensor, labels, distance_mask, angle_mask = data
        # label_ca, label_cb, label_phi, label_psi
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(input_tensor)

        ca_output = outputs[0].squeeze()
        ca_label = labels[0].view(input_tensor.size(0),2500,-1).squeeze()
        masked_ca_label = get_masked_mat(ca_label,distance_mask.view(input_tensor.size(0),2500,-1).squeeze(),-999)

        cb_output = outputs[1].squeeze()
        cb_label = labels[1].view(input_tensor.size(0),2500,-1).squeeze()
        masked_cb_label = get_masked_mat(cb_label,distance_mask.view(input_tensor.size(0),2500,-1).squeeze(),-999)

        l1 = mse_loss(ca_output,masked_ca_label,-999)
        l2 = mse_loss(cb_output,masked_cb_label,-999)

        l3 = mse_loss(get_masked_mat(outputs[2].squeeze(), angle_mask), labels[2], -999)
        l4 = mse_loss(get_masked_mat(outputs[3].squeeze(), angle_mask), labels[3], -999)
        loss = l1 + l2 + l3 + l4
        loss = l1 + l2
        loss.backward()
        optimizer.step()

        # print statistics
        if i % 10 == 0:  # print every 10 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}')
            running_loss = 0.0

print('Finished Training')
