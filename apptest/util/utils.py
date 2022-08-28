import math
import re
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

MASK_VALUE = -999
# follow the order in aaindex
_aa_dict = {
    'A': '0',
    'R': '1',
    'N': '2',
    'D': '3',
    'C': '4',
    'Q': '5',
    'E': '6',
    'G': '7',
    'H': '8',
    'I': '9',
    'L': '10',
    'K': '11',
    'M': '12',
    'F': '13',
    'P': '14',
    'S': '15',
    'T': '16',
    'W': '17',
    'Y': '18',
    'V': '19'
}


def get_masked_mat(input_mat, mask, mask_fill_value=MASK_VALUE, device=None):
    out_mat = torch.ones(input_mat.shape)
    if device is not None:
        mask = mask.to(device)
        out_mat = out_mat.to(device)
    out_mat[(mask == 0).squeeze()] = mask_fill_value
    out_mat[(mask != 0).squeeze()] = input_mat[(mask != 0).squeeze()]
    return out_mat


def max_shape(data):
    """Gets the maximum length along all dimensions in a list of Tensors"""
    shapes = torch.Tensor([_.shape for _ in data])
    # 转置 0 维和 1 维，转置之后就可以把行放在一个list里，列放在一个list里，然后选取没有一个list里面最大
    # 返回value和indice，然后选取value转换成int
    return torch.max(shapes.transpose(0, 1), dim=1)[0].int()


def pad_data_to_max_shape(tensor_data, pad_value=0, type="n"):
    target_shape = torch.Tensor([50]).int()
    padding = reversed(target_shape - torch.Tensor(list(tensor_data.shape)).int())
    padding = F.pad(padding.unsqueeze(0).t(), (1, 0, 0, 0)).view(1, -1)[0].tolist()

    if type == "s":
        # 只pad下方内容，不pad右边
        padding[1] = 0
    padded_data = F.pad(tensor_data, padding, value=pad_value)

    return padded_data


def pad_data_to_same_shape(tensor_list, pad_value=0):
    target_shape = max_shape(tensor_list)
    padded_dataset_shape = [len(tensor_list)] + list(target_shape)

    # 生成data行，max列的垃圾值？相当于初始化这个batch的值！！
    padded_dataset = torch.Tensor(*padded_dataset_shape).type_as(
        tensor_list[0])

    # 遍历每一条数据
    for i, data in enumerate(tensor_list):
        # Get how much padding is needed per dimension
        # 数据输入可能是多行多列，所以需要行列都pad，reverse之后变成[列差多少，行差多少]
        padding = reversed(target_shape - torch.Tensor(list(data.shape)).int())

        # Add 0 every other index to indicate only right padding
        # padding.unsqueeze(-1) 应该和 padding.unsqueeze(0).t() 是一个意思 t(转置)
        # F.pad(待扩充，(左边填充数， 右边填充数， 上边填充数， 下边填充数))，默认填充 0
        # view相当于resize，resize成1列
        padding = F.pad(padding.unsqueeze(0).t(), (1, 0, 0, 0)).view(-1, 1)
        # 为啥不直接在上一行做view(1,-1)??
        padding = padding.view(1, -1)[0].tolist()

        # 制作了一个 (左边填充数， 右边填充数， 上边填充数， 下边填充数， 前边填充数，后边填充数), 这是3D的情况
        padded_data = F.pad(data, padding, value=pad_value)
        # print(padded_data)
        padded_dataset[i] = padded_data

    return padded_dataset


# array1 = torch.Tensor([[[1,2,3,4,5]]])
# array2 = torch.Tensor([[[1,3,4,5],[4,2,1,7]]])
# array3 = torch.Tensor([
#                         [
#                             [1,3,4,5,10,20,40],[10,2,1,2,3,4,8]
#                         ],
#                         [
#                             [1,3,4,5,10,20,40],[10,2,1,2,3,4,8]
#                         ]
#                         ])
# list_array = [array1,array2,array3]
# print(pad_data_to_same_shape(list_array))


def letter_to_num(string, dict_):
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')

    # lambda m:m.group(0) used for go through the string one by one
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [int(i) for i in num_string.split()]
    return num


def one_hot_seq_2D(seq):
    """Gets a one-hot encoded version of a protein sequence"""
    return F.one_hot(torch.LongTensor(letter_to_num(seq, _aa_dict)),
                     num_classes=20)


def one_hot_seq_1D(seq):
    return torch.LongTensor(letter_to_num(seq, _aa_dict))


def get_matrix_A():
    path = 'D:\\Work_Study\\Work\\Syneron\\Code\\apptest\\reproduction_v0.1\\aaindex\\aaindex_features.txt'
    f = open(path)
    aaindex_details = f.readlines()
    f.close()

    # (20, 556) contains all the aa and corresponding aaindex features
    # do not contain non-standard aa
    natures = list(_aa_dict.keys())
    aaindex_dictionary = defaultdict(list)

    for line in aaindex_details:
        line = line.strip()
        key, value = line.split(": ")
        contain_NA = False
        # go through aaindex
        j = 0
        # go through aa
        if value.count("NA") > 0:
            contain_NA = True
        if not contain_NA:
            for v in value.split(", "):
                aaindex_dictionary[natures[j]].append(v)
                j += 1

    aaindex_matrix = []
    for key, value in aaindex_dictionary.items():
        aa_aaindex = []
        for v in value:
            aa_aaindex.append(float(v))
        aaindex_matrix.append(aa_aaindex)

    # (20,553)
    return aaindex_matrix


def get_matrix_S(seq, pad=True):
    return F.normalize(torch.matmul(pad_data_to_max_shape(one_hot_seq_2D(seq), type="s"), pca(get_matrix_A(), 15)).type(
        torch.float)) if pad else F.normalize(
        torch.matmul(one_hot_seq_2D(seq), pca(get_matrix_A(), 15)).type(torch.float))


def pca(matrix, n):
    pca = PCA(n_components=n)
    pca.fit(matrix)
    return torch.LongTensor(pca.transform(matrix))

# print(get_matrix_S("AANYV").shape)
# print(one_hot_seq_1D("AANYV"))
