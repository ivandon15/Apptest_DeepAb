import os
import pathlib
import pickle
import warnings
from Bio import SeqIO
warnings.filterwarnings("ignore")
# from AAindex import AAindex


# def preprocess(pdb):
#     pdb.extractID()
#     pdb.modifiedPDBfile()
#     print(len(pdb.pdblist))
#     pdb.renumber()
#     pdb.getDistance("cb_distance/", "CB")
#     pdb.getDistance("cb_distance/", "CA")
#     pdb.getTorsion()
#     pdb.getCyclic("disulfide")
#
#     aaindex = AAindex("reproduction_v0.1/aaindex/aaindex list.txt", "aaindex/")
#     aaindex.download()


def getContent(pdb_file, cyclic_position_file, ca_distance_path, cb_distance_path, torsion_file):
# def getContent(pdb_file,torsion_file):

    """
    Using pdb id and chain to find all the information.
    :param pdb_file:
    :param cyclic_position_file:
    :param ca_distance_path:
    :param cb_distance_path:
    :param torsion_file:
    :return: seq, matrix_c, ca_distance, cb_distance, torsion
    """
    pdb_chain_name = pathlib.Path(pdb_file).stem
    seq = ""
    seq_len = 0

    # get sequence and sequence length
    for record in SeqIO.parse(pdb_file, "pdb-atom"):
        seq_len = len(record.seq)
        seq = str(record.seq)

    # get matrix_c
    cyclic_position = ""
    with open(cyclic_position_file, "r") as f:
        for line in f:
            if line.split(": ")[0] == pdb_chain_name:
                cyclic_position = line.split(": ")[1]
                break
    # print(cyclic_position)

    matrix_c = [[0 for _ in range(seq_len)] for _ in range(seq_len)]
    if cyclic_position !="":
        for pos in cyclic_position.split(", "):
            res_pos1, res_pos2 = pos.split("-")
            matrix_c[int(res_pos1) - 1][int(res_pos2) - 1] = 1
            matrix_c[int(res_pos2) - 1][int(res_pos1) - 1] = 1
    # print(np.asarray(matrix_c))

    # get distances
    ca_distance_file = os.path.join(ca_distance_path, pdb_chain_name + ".pkl")
    ca_distance = pickle.load(open(ca_distance_file, 'rb'))
    # print(np.asarray(ca_distance))

    cb_distance_file =  os.path.join(cb_distance_path,pdb_chain_name + ".pkl")
    cb_distance = pickle.load(open(cb_distance_file, 'rb'))
    # print(np.asarray(cb_distance))

    phis, psis = [],[]
    with open(torsion_file, "r") as f:
        for line in f:
            if line.split(": ")[0] == pdb_chain_name:
                str_torsion = line.split(": ")[1]
                for tor in str_torsion.split(";"):
                    if len(tor) > 1:
                        phi, psi = tor.split(", ")
                        phi = float(phi) if phi != 'None' else None
                        psi = float(psi) if psi != 'None' else None
                        # a list of phi-psi
                        phis.append(phi)
                        psis.append(psi)
                break
    return seq, matrix_c, ca_distance, cb_distance, phis, psis


# # 对原始数据的处理
# # 下载pdb->提取对应chain->pdb重编码->CA/CB距离计算; 二面角计算; 成环位置计算
# pdb = Pdb("apptest.fasta", "data/")
# # preprocess(pdb)
#
# arg: pdbfile train的目录, pdbfile test的目录, 成环位置文件, 两个距离目录, 二面角文件
# pdb_file = "test/1acw.A.pdb"
# cyclic_position_file = "cyclic_position_disulfide.txt"
# ca_distance_path = "ca_distance/"
# cb_distance_path = "cb_distance/"
# torsion_file = "torsion.txt"
# package = getContent(pdb_file,cyclic_position_file, ca_distance_path, cb_distance_path, torsion_file)
# print(package)
