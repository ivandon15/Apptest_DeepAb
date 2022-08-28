"""
This file contains:
1. PDB files download: Extract pdbid from fasta file
2. PDB files modified (extract specific chain)
3. PDB files filter: remove the empty modified files
4. Renumber the atom and residue number

5. Extract pdb's ca-ca & cb-cb distance and backbone torsion
"""
import math

from pymol import cmd
import os
import pathlib
import pickle
import urllib.request
from urllib.error import URLError
from Bio import SeqIO
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO, Select, is_aa
import warnings

warnings.filterwarnings('ignore')


class ChainResidueSelect(Select):
    """
    Override the Select class in Bio, it has four methods. More details
    plz check 11.1.7 Writing PDB files
    Change the Constructor, add "remain" param which means the chain id
    that we need to remain.
    """

    def __init__(self, remain):
        self.remain = remain

    def accept_chain(self, chain):
        if chain.get_id() == self.remain:
            return True
        else:
            return False

    def accept_residue(self, residue):
        # filter standard aa
        return is_aa(residue)


def isABResidue(file):
    is_AB = False
    with open(file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and (line[16] == 'A' or line[16] == 'B'):
                is_AB = True
                break
    return is_AB


class Pdb:

    def __init__(self, fasta_file="", download_path=""):
        """
        :param fasta_file: original input fasta file
        :param download_path:
        """
        self.fasta_file = fasta_file
        self.download_path = download_path
        self.pdblist = []

        files = os.listdir(self.download_path)
        for file in files:
            self.pdblist.append(pathlib.Path(file).stem)

    def extractID(self):
        """
        Extract pdb id and their split info from fasta file and save txt file into export pdb file.
        If you want to output the list to file, you may uncomment the following block.
        >6t9m.BBB|lcl|split_3
        GPAMK
        """
        print("Start extracting PDB ids...")
        for seq_record in SeqIO.parse(self.fasta_file, "fasta"):
            self.pdblist.append(seq_record.id + ", " + "temp")
            # ids = seq_record.id.split("|")
            # # ["1a0m.A, split_1","2ktc.A, split_2"...]
            # self.pdblist.append(ids[0] + ", " + ids[2])

        # uncomment it if you want save the list to file
        # with open("pdblist.txt", 'w') as f:
        #     for i in pdblist:
        #         f.write("%s\n" % i)
        #     print('Done')

        print("Extracted %d pdb ids with chains." % self.get_list_len())
        return self.pdblist

    """
    # download pdb
        pdb_url = "https://files.rcsb.org/view/%s.pdb" % pdb_id
        pdb_file = os.path.join(pdb_path, "%s.pdb" % pdb_id)
        if not os.path.exists(pdb_file) or os.path.getsize(pdb_file) == 0:
            os.system("wget -O %s %s" % (pdb_file, pdb_url))
    """

    def download(self):
        """
        Download the pdb file using pdblist from rcsb website
        :param download_path: output file path
        """
        print("Start downloading...")
        failed = []
        for item in tqdm(self.pdblist):
            # pdbid, chain = item.split(", ")[0].split(".")
            pdbid, chain = item.split(", ")[0].split("_")

            try:
                # urllib.request.urlretrieve('http://files.rcsb.org/download/' + pdbid
                #                            + '.pdb', self.download_path + pdbid + "." + chain + '.pdb')
                urllib.request.urlretrieve('http://files.rcsb.org/download/%s.pdb' % pdbid,
                                           self.download_path + "%s_%s.pdb" % (pdbid, chain))
            except urllib.error.HTTPError as e:
                # throw the HTTP error first
                # if something wrong with downloading, then remove the id in the list
                failed.append(item)
                print("Something wrong with the server.")
                print('Error code: ', e.code)
                print('Error PDB ID: ', pdbid)
                continue
            except urllib.error.URLError as e:
                failed.append(item)
                print("Cannot reach the server.")
                print('Reason: ', e.reason)
                print('Error PDB ID: ', pdbid)
                continue
        for item in failed:
            self.pdblist.remove(item)

        print("Successfully download %d pdb files with chains." % self.get_list_len())

        # now the pdblist only contain valid pdb file
        return self.pdblist

    def modifiedPDBfile(self):
        """
        Modified PDB file in the download_path to fit the pdblist (remain one of the chain and only remain ATOM)
        :param download_path: the folder contains pdb file
        """

        print("Start modifying the pdb files...")
        for item in tqdm(self.pdblist):
            print(item)

            pdbid = item.split(", ")[0]

            # get PDB parser
            parser = PDBParser(PERMISSIVE=1)
            # get the original pdb structure if the file exists
            if os.path.exists(self.download_path + pdbid + '.pdb'):
                if os.path.getsize(self.download_path + pdbid + '.pdb') == 0:
                    os.remove(self.download_path + pdbid + '.pdb')
                else:
                    try:
                        structure = parser.get_structure(pdbid, self.download_path + pdbid + '.pdb')
                        # write new PDB file
                        io = PDBIO()
                        io.set_structure(structure)
                        # io.save(self.download_path + pdbid + '.pdb', ChainResidueSelect(pdbid.split(".")[1]))
                        io.save(self.download_path + pdbid + '.pdb',
                                ChainResidueSelect(pdbid.split("_")[1]))  # pdbid.split("_")[1]
                    except Exception:
                        os.remove(self.download_path + pdbid + '.pdb')
        print("Before modified, there are %d PDB files." % self.get_list_len())
        print("Checking empty PDB files...")

        # check invalid(empty) pdb files
        removed = []
        self.pdblist = []
        files = os.listdir(self.download_path)
        for file in files:
            self.pdblist.append(pathlib.Path(file).stem)

        for item in self.pdblist:
            pdbid = item.split(", ")[0]
            parser = PDBParser(PERMISSIVE=1)
            structure = parser.get_structure(pdbid, self.download_path + pdbid + '.pdb')
            if len(structure) == 0:
                removed.append(item)
        for item in removed:
            self.pdblist.remove(item)
            os.remove(self.download_path + item.split(", ")[0] + '.pdb')

        print("PDB files have been modified. And remove the empty PDB files, remain %d PDB files" % self.get_list_len())

    def getSeq(self, file):
        for record in SeqIO.parse(file, "pdb-atom"):
            seq = str(record.seq)
            return seq

    def renumber(self):
        """
        Renumber the atom and residues on the download path files
        """

        print("Start renumbering the residue...")
        files = os.listdir(self.download_path)

        for file in tqdm(files):
            out = list()
            with open(self.download_path + file, "r") as f:
                atom_no = 1
                residue_no = 0
                # record the previous residue number
                previous_residue_no = ""
                for line in f:
                    if line.startswith(('ATOM', 'HETATM', 'TER')):
                        # ATOM or HETATM line
                        if len(line.split()) > 5:
                            # get the current residue number, may be int maybe 1C
                            origin_residue_no = line[22:26]
                            if previous_residue_no != origin_residue_no:
                                # count the residue number
                                residue_no += 1
                                previous_residue_no = origin_residue_no

                        # rewrite atom number
                        atom_num = str(atom_no)
                        while len(atom_num) < 5:
                            atom_num = ' ' + atom_num
                        line = '%s%s%s' % (line[:6], atom_num, line[11:])
                        atom_no += 1

                        # rewrite residue number
                        residue_num = str(residue_no)
                        while len(residue_num) < 4:
                            residue_num = ' ' + residue_num
                        new_row = '%s%s' % (line[:22], residue_num)
                        while len(new_row) < 29:
                            new_row += ' '
                        xcoord = line[30:38].strip()
                        while len(xcoord) < 9:
                            xcoord = ' ' + xcoord
                        line = '%s%s%s' % (new_row, xcoord, line[38:])

                        # if multiple chain or model, need to reset the count
                        if line.startswith('TER'):
                            atom_no = 1
                            residue_no = 0
                            previous_residue_no = ""
                    out.append(line)
            # rewrite the pdb file
            with open(self.download_path + file, 'w') as f:
                for line in out:
                    f.write(line)
        print("Renumber process is done.")

    def get_list_len(self):
        return len(self.pdblist)

    def filterABResidue(self):
        """
        Filter out the AB atoms
        """
        print("Start filtering out the AB atom...")
        files = os.listdir(self.download_path)
        for file in tqdm(files):
            if isABResidue(self.download_path + file):
                os.remove(self.download_path + file)
        self.pdblist = []
        files = os.listdir(self.download_path)
        for file in files:
            self.pdblist.append(pathlib.Path(file).stem)
        print("After AB atom filtering, we got %d files left." % self.get_list_len())

    def getDistance(self, output_path, atom_type):
        """
        Get the distance between two atoms
        :param atom_type: CA or CB
        :return: a matrix
        """

        print("Start calculating " + atom_type + " distance...")
        files = os.listdir(self.download_path)

        for file in tqdm(files):
            p = PDBParser()
            structure = p.get_structure("X", self.download_path + file)

            # get the sequence length
            seq_len = 0
            for model in structure.get_models():
                for _ in model.get_residues():
                    seq_len += 1
                break

            # initial the matrix
            matrix = [[0] * seq_len for _ in range(seq_len)]
            for model in structure:
                # only one chain in each pdb
                for chain in model:
                    i, j = 0, 0
                    # 保证CA存在，所以需要参考Bio的最大长度
                    maxlength = len(self.getSeq(self.download_path + file))
                    # TODO: could reduce the calculation
                    for residue_i in chain:
                        for residue_j in chain:
                            if i < maxlength and j < maxlength:
                                # every aa must have one CA but not CB
                                try:
                                    atom_i = residue_i[atom_type]
                                    atom_j = residue_j[atom_type]
                                    matrix[i][j] = atom_i - atom_j
                                except KeyError:
                                    # use CA to instead CB
                                    try:
                                        atom_i = residue_i["CA"]
                                        atom_j = residue_j["CA"]
                                        matrix[i][j] = atom_i - atom_j
                                        j += 1
                                    except KeyError:
                                        # 如果还有问题，那就是HETATM直接去掉
                                        print(f"non-standard: {file}")
                                        break
                                    continue
                            j += 1
                        i += 1
                        j = 0

                # only use first model
                break
            # save to pkl
            with open(output_path + pathlib.Path(file).stem + ".pkl", "wb") as f:
                pickle.dump(matrix, f)
        print("Distance calculating is done.")

    def getTorsion(self):
        """
        Used to get the dihedral psi and phi of the peptide
        :param file: pdb file
        :return: a list of dihedral [phi,psi]
        """
        print("Start calculating torsion...")
        files = os.listdir(self.download_path)
        for file in tqdm(files):
            p = PDBParser()
            structure = p.get_structure("X", self.download_path + file)

            phi_psi = pathlib.Path(file).stem + ": "

            # TODO:????
            maxlength = len(self.getSeq(self.download_path + file))

            for model in structure:
                # get the internal_coordinates to calculate the phi psi
                model.atom_to_internal_coordinates()
                for r in model.get_residues():
                    if r.internal_coord:
                        # Ca-N -> phi, Ca-C ->psi
                        # [phi, psi], the first aa does not has phi,
                        # last one does not has psi
                        phi_psi = phi_psi + "%s, %s;" % (str(r.internal_coord.get_angle('phi')),
                                                         str(r.internal_coord.get_angle('psi')))
                break

            with open("torsion_linear.txt", "a") as f:
                f.write(phi_psi + "\n")
        print("Torsion calculating is done.")

    def is_bond(self, atom1, atom2):
        """
        This function is used to check if two atoms are bonded and not neighbor, if the distance is <= 2.05A
        then there is a disulfide bond.
        :param atom1: atom 1 S
        :param atom2: atom 2 S
        :return: True if distance <= 2.05
        """
        pos1 = atom1.coord[0], atom1.coord[1], atom1.coord[2]
        pos2 = atom2.coord[0], atom2.coord[1], atom2.coord[2]
        if math.sqrt(sum(list(tuple(map(lambda i, j: pow(i - j, 2), pos1, pos2))))) <= 2.05:
            if int(atom1.resi) - 1 != int(atom2.resi) and int(atom1.resi) + 1 != int(atom2.resi):
                return True
        return False

    def getCyclic(self, check_type="disulfide"):
        """
        Find all the files' cyclic position
        :param type:  two types: disulfide and bond
        """
        print("Start calculating cyclic position...")
        files = os.listdir(self.download_path)
        for file in tqdm(files):
            pairs = []

            cmd.load(self.download_path + file)
            cmd.remove('solvent')

            if check_type == "disulfide":
                # get all the cys
                cmd.select('allCys', 'resn cys')

                # get all the S atom in CYS
                cys_s_atoms = []
                for atom_c in cmd.get_model("allCys").atom:
                    if atom_c.name == "SG":
                        cys_s_atoms.append(atom_c)

                for i in range(len(cys_s_atoms)):
                    for j in range(i + 1, len(cys_s_atoms)):
                        if self.is_bond(cys_s_atoms[i], cys_s_atoms[j]):
                            # add residue number
                            pairs.append(cys_s_atoms[i].resi + "-" + cys_s_atoms[j].resi)

            elif check_type == "bond":
                seq_len = 0
                for record in SeqIO.parse(self.download_path + file, "pdb-atom"):
                    seq_len = len(record.seq)
                cmd.select('allAA', ' resi 1-' + str(seq_len))

                # get all the cys position
                allposition = set([str(i) for i in range(1, seq_len + 1)])

                for aa in allposition:

                    # create multiple layers with each cys
                    cmd.select('AA' + aa, 'resi ' + aa)

                    # get the atoms that directly bonded with current aa
                    cmd.select('AA' + aa + 'nearby', 'neighbor AA' + aa)

                    # except for the previous and next
                    # consider head tail connection
                    if len(cmd.get_model('AA' + aa + 'nearby').atom) >= 3 or (
                            (int(aa) == 1 or int(aa) == int(seq_len)) and len(
                        cmd.get_model('AA' + aa + 'nearby').atom) == 2):
                        for atom in cmd.get_model('AA' + aa + 'nearby').atom:
                            pair = []
                            try:
                                if int(atom.resi) + 1 != int(aa) and int(atom.resi) - 1 != int(aa):
                                    pair.append(aa)
                                    pair.append(atom.resi)
                                    if "-".join(sorted(pair)) not in pairs:
                                        pairs.append("-".join(sorted(pair)))
                            except ValueError:
                                continue
            cmd.delete('all')
            with open("cyclic_position_" + check_type + ".txt", "a") as f:
                f.write(pathlib.Path(file).stem + ": " + ", ".join(pairs) + "\n")

        print("Cyclic calculation is done.")
