import os
import pathlib
import pickle
import torch.utils.data as data
import torch.nn as nn

from apptest.util.utils import one_hot_seq_1D, get_matrix_S, pad_data_to_max_shape, one_hot_seq_2D, \
    pad_data_to_same_shape, get_masked_mat
import torch
import torch.nn.functional as F
import numpy as np

vocab_size = 20
embed_size = 12


class SeqDataset(data.Dataset):
    def __init__(self, pkl_file):
        with open(pkl_file, "rb") as f:
            self.data_map = pickle.load(f)

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, index):
        # each data in the pickle file
        sample = self.data_map[index]
        seq, cyclic_matrix, ca, cb, phi, psi = sample["seq"], sample["cyclic_mat"], sample["ca_dis"], sample["cb_dis"], \
                                               sample["phi"], sample["psi"]
        # mask the first angle of phi and the last angle of psi
        phi[0] = 0
        psi[-1] = 0
        # get matrix E,S,C
        onehot_1d_pad = pad_data_to_max_shape(one_hot_seq_1D(seq))
        embed = nn.Embedding(vocab_size, embedding_dim=embed_size)
        matrix_E = embed(onehot_1d_pad)

        # normalize一下，不然matrix_S本身太大了
        matrix_S = get_matrix_S(seq)
        matrix_S = F.normalize(matrix_S.type(torch.float))
        matrix_C = torch.Tensor(cyclic_matrix)
        matrix_C = pad_data_to_max_shape(matrix_C)

        # concat matrix
        input_tensor = torch.cat([matrix_E, matrix_C, matrix_S], 1)

        label_ca = torch.Tensor(ca)
        label_cb = torch.Tensor(cb)
        label_phi = torch.Tensor(phi)
        label_psi = torch.Tensor(psi)

        distance_mask = pad_data_to_max_shape(torch.ones((len(seq), len(seq))))
        angle_mask = (pad_data_to_max_shape(label_phi) != 0).long()

        label_ca = pad_data_to_max_shape(label_ca)
        label_cb = pad_data_to_max_shape(label_cb)
        # print(label_ca.view(-1,1))
        label_phi = pad_data_to_max_shape(label_phi)
        label_psi = pad_data_to_max_shape(label_psi)

        return input_tensor, [label_ca, label_cb, label_phi, label_psi], distance_mask, angle_mask





# print(SeqDataset("../../val_data_info.pkl").__getitem__(0)[1][0])

#
# a = torch.randn((2,3))
# o = []
# print(a)
# print(np.multiply(a,a))
# print(torch.mul(a,a))
# for i in range(len(a)):
#     t = []
#     for j in range(len(a)):
#         t.append(np.multiply(a[i].tolist(),a[j].tolist()).tolist())
#         print(t)
#     o.append(t)
# print(o)
# print(torch.Tensor(o))
# print("==============================")
# i = torch.randn((2,3,4))
# print(torch.Tensor(i))
# print(torch.Tensor(i)[1,::])
# all_ordered_idx_pairs = torch.cartesian_prod(torch.tensor(range(i.shape[1])),torch.tensor(range(i.shape[1])))
# print(all_ordered_idx_pairs)
# print(i[0][all_ordered_idx_pairs])
# aa = [i[j][all_ordered_idx_pairs] for j in range(i.shape[0])]
# print(torch.mul(aa[0][:,0,:],aa[0][:,1,:]))
# print(torch.stack([torch.mul(aa[j][:,0,:],aa[j][:,1,:]).view(i.shape[1],i.shape[1],-1) for j in range(i.shape[0])]).shape)
# # print(torch.Tensor(np.array(o)).shape)
# # print(torch.Tensor(o).shape)
#
# test = torch.randn(2,2,2,2)
# print(test)
# print(torch.mean(test,dim=-1).view(test.size(0),test.size(1),test.size(2),1))
# print(torch.mean(test,dim=-1).view(test.size(0),test.size(1),test.size(2),1).squeeze(-1))
