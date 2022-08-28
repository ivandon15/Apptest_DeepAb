import argparse
import os
import sys

from PDBPreprocess import Pdb

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Download PDB file and screening CP.")

    parser.add_argument('FastaFile',
                        metavar='fastafile',
                        type=str,
                        help='the fasta file from PDB database')
    parser.add_argument('PDBPath',
                        metavar='pdbpath',
                        type=str,
                        help='the path of the PDB files')

    args = parser.parse_args()
    if not os.path.exists(args.FastaFile):
        print('Fasta File does not exist.')
        sys.exit()
    if not os.path.isdir(args.PDBPath):
        print('Data Path does not exist.')
        sys.exit()

    pdb = Pdb(args.FastaFile, args.PDBPath)
    # pdb.extractID()
    # pdb.download()
    pdb.modifiedPDBfile()
    pdb.renumber()
    # pdb.getCyclic("disulfide")
