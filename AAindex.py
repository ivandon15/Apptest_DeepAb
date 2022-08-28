"""
This file is used to get the aaindex features
"""
"""
Download all the 566 aaindex features and save to pickle file as
{aaindex1: [aa1,aa2,...aa20],aaindex2...aaindex566} -> aaindex_details.pkl
"""
import pickle
from urllib.request import urlopen
from bs4 import BeautifulSoup
from tqdm import tqdm


class AAindex:
    def __init__(self, aaindex_list_file, download_path):
        self.aaindex_list_file = aaindex_list_file
        self.download_path = download_path

        self.aaindexid = []
        f = open(self.aaindex_list_file, 'r')
        for i in f.readlines()[2:]:
            title = i.split()[0]
            self.aaindexid.append(title)

    def download(self):
        """
        Download all the content in aaindex_list and save to the output_file
        :param aaindex_list:
        :param output_file: a pickle storing a dict {aaindex1: [xx,xx],aaindex2: [xx,xx]...}
        """

        print("Start download AAindex files...")

        dictionary = {}
        for aaindex in tqdm(self.aaindexid):
            url = 'https://www.genome.jp/entry/aaindex:' + aaindex
            html = urlopen(url).read()
            soup = BeautifulSoup(html, features="html.parser")

            # kill all script and style elements
            for script in soup(["script", "style"]):
                script.extract()  # rip it out

            # get text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())

            # the line below the I row is the digits
            digits = False
            count = 0
            features = []
            for line in lines:
                seg = line.split()

                if digits and count < 2:
                    #  A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
                    features.extend(line.split())
                    count += 1
                if len(seg) > 0 and seg[0] == "I":
                    digits = True
            with open(self.download_path + "aaindex_features.txt", 'a') as f:
                f.write(aaindex+": "+", ".join(features)+"\n")
        print("AAindex files downloading is done.")
