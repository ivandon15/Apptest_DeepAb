import csv
import sys
import os
import numpy as np
import json
import argparse
from prody import *
from sidechainnet.utils.measure import *
from tqdm import tqdm
import random
import pdb

# TODO: Need to create train, val, test
if __name__ == "__main__":
    pdb_path = "/home/gt/Projects_dir/pdb_10k/"
    output_path = "/home/gt/Project_code/HERN/data/pdb_10k/"
    json_path = "/home/gt/Project_code/HERN/data/pdb_10k/total.jsonl"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    """
    # Total data
    with open(os.path.join(output_path, 'total.jsonl'), 'w') as f:
        for pdb_file in tqdm(os.listdir(pdb_path)):
            pdb_file_full = os.path.join(pdb_path, pdb_file)
            peptide = parsePDB(pdb_file_full)
            try:
                _, pcoords, pseq, _, _ = get_seq_coords_and_angles(peptide)
                # 第一个参数: (seq_len, 12), 骨架二面角，键角，侧链二面角; hcoords: (1680, 3), 其余没原子的地方就填零;
                #  hseq: 蛋白质序列; 第四个参数: 每一个氨基酸的缩写;第五个参数:返回非标准氨基酸.
                pcoords = pcoords.reshape((len(pseq), 14, 3))
            except:
                print(pdb_file)
                continue
            # (seq_len, max_atom_num, coords)
            if len(pseq) <= 2:
                continue
            pcoords = eval(np.array2string(pcoords, separator=',', threshold=np.inf, precision=3, suppress_small=True))
            data = {
                "pdb": pdb_file,
                "peptide_seq": pseq,
                "peptide_coords": pcoords,
            }
            json.dump(data, f)
            f.write('\n')

    """
    # Split train and val dataset
    f = open(json_path, 'r')
    tmp = f.readlines()
    pdb_list = []
    ss_list = []
    for ii in tmp:
        pdb_list.append(json.loads(ii)['pdb'])
    f.close()
    random.shuffle(pdb_list)
    amount = len(pdb_list)
    print(f'Total amount = {amount}')
    for i in range(amount):
        print('{}  {}'.format(i, pdb_list[i]))
        if i <= 0.8*amount:
            with open(os.path.join(output_path, 'train.jsonl'), 'a') as f:
                pdb_file_full = os.path.join(pdb_path, pdb_list[i])
                peptide = parsePDB(pdb_file_full)
                try:
                    _, pcoords, pseq, _, _ = get_seq_coords_and_angles(peptide)
                    # 第一个参数: (seq_len, 12), 骨架二面角，键角，侧链二面角; hcoords: (1680, 3), 其余没原子的地方就填零;
                    #  hseq: 蛋白质序列; 第四个参数: 每一个氨基酸的缩写;第五个参数:返回非标准氨基酸.
                except:
                    print(pdb_file)
                try:
                    pcoords = pcoords.reshape((len(pseq), 14, 3))
                except AttributeError:
                    print(pdb_file)
                    continue

                pcoords = pcoords.reshape((len(pseq), 14, 3))
                # (seq_len, max_atom_num, coords)
                pcoords = eval(
                    np.array2string(pcoords, separator=',', threshold=np.inf, precision=3, suppress_small=True))
                data = {
                    "pdb": pdb_list[i],
                    "peptide_seq": pseq,
                    "peptide_coords": pcoords,
                }
                json.dump(data, f)
                f.write('\n')
        elif (i > 0.8*amount) and (i <= 0.9*amount):
            with open(os.path.join(output_path, 'val.jsonl'), 'a') as f:
                pdb_file_full = os.path.join(pdb_path, pdb_list[i])
                peptide = parsePDB(pdb_file_full)
                try:
                    _, pcoords, pseq, _, _ = get_seq_coords_and_angles(peptide)
                    # 第一个参数: (seq_len, 12), 骨架二面角，键角，侧链二面角; hcoords: (1680, 3), 其余没原子的地方就填零;
                    #  hseq: 蛋白质序列; 第四个参数: 每一个氨基酸的缩写;第五个参数:返回非标准氨基酸.
                except:
                    print(pdb_file)
                try:
                    pcoords = pcoords.reshape((len(pseq), 14, 3))
                except AttributeError:
                    print(pdb_file)
                    continue

                pcoords = pcoords.reshape((len(pseq), 14, 3))
                # (seq_len, max_atom_num, coords)
                pcoords = eval(
                    np.array2string(pcoords, separator=',', threshold=np.inf, precision=3, suppress_small=True))
                data = {
                    "pdb": pdb_list[i],
                    "peptide_seq": pseq,
                    "peptide_coords": pcoords,
                }
                json.dump(data, f)
                f.write('\n')
        elif i > 0.9*amount:
            with open(os.path.join(output_path, 'test.jsonl'), 'a') as f:
                pdb_file_full = os.path.join(pdb_path, pdb_list[i])
                peptide = parsePDB(pdb_file_full)
                try:
                    _, pcoords, pseq, _, _ = get_seq_coords_and_angles(peptide)
                    # 第一个参数: (seq_len, 12), 骨架二面角，键角，侧链二面角; hcoords: (1680, 3), 其余没原子的地方就填零;
                    #  hseq: 蛋白质序列; 第四个参数: 每一个氨基酸的缩写;第五个参数:返回非标准氨基酸.
                except:
                    print(pdb_file)
                try:
                    pcoords = pcoords.reshape((len(pseq), 14, 3))
                except AttributeError:
                    print(pdb_file)
                    continue

                pcoords = pcoords.reshape((len(pseq), 14, 3))
                # (seq_len, max_atom_num, coords)
                pcoords = eval(
                    np.array2string(pcoords, separator=',', threshold=np.inf, precision=3, suppress_small=True))
                data = {
                    "pdb": pdb_list[i],
                    "peptide_seq": pseq,
                    "peptide_coords": pcoords,
                }
                json.dump(data, f)
                f.write('\n')

