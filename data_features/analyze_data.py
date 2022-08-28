# coding=utf-8
import csv
import sys
import os
import numpy as np
import json
import argparse
from Bio.PDB import DSSP, PDBParser
from prody import *
from sidechainnet.utils.measure import *
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt
import tqdm

pdb_path = "/home/gt/Projects_dir/pdb_10k/"
output_path = "/home/gt/Project_code/HERN/data/peptide_yifan_2110/"
# TODO: Need to create train, val, test
if __name__ == "__main__":
    '''
    length = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, 'yifan_data.jsonl'), 'w') as f:
        for pdb_file in tqdm(os.listdir(pdb_path)):
            pdb_file_full = os.path.join(pdb_path, pdb_file)
            peptide = parsePDB(pdb_file_full)
            try:
                _, pcoords, pseq, _, _ = get_seq_coords_and_angles(peptide)
                length.append(len(pseq))
                # 第一个参数: (seq_len, 12), 骨架二面角，键角，侧链二面角; hcoords: (1680, 3), 其余没原子的地方就填零;
                #  hseq: 蛋白质序列; 第四个参数: 每一个氨基酸的缩写;第五个参数:返回非标准氨基酸.
            except AttributeError:
                print(pdb_file)

            pcoords = pcoords.reshape((len(pseq), 14, 3))
            # (seq_len, max_atom_num, coords)
            pcoords = eval(np.array2string(pcoords, separator=',', threshold=np.inf, precision=3, suppress_small=True))
            data = {
                "pdb": pdb_file,
                "peptide_seq": pseq,
                "peptide_coords": pcoords,
            }
            json.dump(data, f)
            f.write('\n')
        plt.hist(length, bins=12, rwidth=0.9, density=True)
    '''
    plist = os.listdir(pdb_path)
    for i in plist:
        print(i)
        pname = os.path.join(pdb_path, i)
        p = PDBParser()
        # pdb.set_trace()
        structure = p.get_structure("Model", pname)
        model = structure[0]
        dssp = DSSP(model, pname)
        for row in dssp:
            with open(os.path.join(output_path, '2nd_struct.txt'), 'a') as q:
                entity = str(row[2]) + ' '
                q.write(entity)
        # H: Alpha helix (4-12); B: Isolated beta-bridge residue; E: Strand; G: 3-10 helix;
        # I: Pi helix;           T: Turn;                         S: Bend;   -: None


def len_dist(data: str):
    json_path = f"/home/gt/Project_code/HERN/data/pdb_10k/{data}.jsonl"
    save_path = f"/home/gt/Project_code/HERN/data/pdb_10k/{data}_len.png"
    f = open(json_path, 'r')
    tmp = f.readlines()
    length = []
    for ii in tmp:
        length.append(len(json.loads(ii)['peptide_seq']))
    f.close()
    bin_num = len(set(length))
    plt.hist(np.array(length), bins=bin_num, rwidth=0.9, density=False)
    plt.title('Peptide Sequence Length distribution')
    plt.xlabel('Sequence Length/a.a.')
    plt.ylabel('Number')
    # plt.show()
    plt.savefig(save_path)
    plt.close()


def ss_dist(data: str):
    json_path = f"/home/gt/Project_code/HERN/data/pdb_10k/{data}.jsonl"
    save_path = f"/home/gt/Project_code/HERN/data/pdb_10k/{data}_ss.png"
    f = open(json_path, 'r')
    tmp = f.readlines()
    pdb_list = []
    ss_list = []
    for ii in tmp:
        pdb_list.append(json.loads(ii)['pdb'])
    f.close()
    iii = 0
    for ti in pdb_list:
        print(f'Index: {iii} | File: {ti}')
        iii += 1
        tpname = os.path.join(pdb_path, ti)
        tp = PDBParser()
        # pdb.set_trace()
        tstructure = tp.get_structure("Model", tpname)
        tmodel = tstructure[0]
        try:
            tdssp = DSSP(tmodel, tpname)
        except:
            continue
        for trow in tdssp:
            ss_list.append(str(trow[2]))

    # count
    ss_count = np.zeros(8)  # H, B, E, G, I, T, S, -
    length = len(ss_list)
    for ii in range(length):
        if ss_list[ii] == 'H':
            ss_count[0] += 1
        elif ss_list[ii] == 'B':
            ss_count[1] += 1
        elif ss_list[ii] == 'E':
            ss_count[2] += 1
        elif ss_list[ii] == 'G':
            ss_count[3] += 1
        elif ss_list[ii] == 'I':
            ss_count[4] += 1
        elif ss_list[ii] == 'T':
            ss_count[5] += 1
        elif ss_list[ii] == 'S':
            ss_count[6] += 1
        elif ss_list[ii] == '-':
            ss_count[7] += 1
    x_data = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
    y_data = ss_count

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    for ii in range(len(x_data)):
        plt.bar(x_data[ii], y_data[ii])
    plt.title('Peptide Second Structure distribution')
    plt.xlabel('Second Structure')
    plt.ylabel('Number')
    # plt.show()
    plt.savefig(save_path)
    plt.close()
