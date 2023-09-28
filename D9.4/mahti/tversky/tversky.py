#!/usr/bin/python

import numpy as np
#import joblib

# indeks Jaccarda
def jaccard(A, B):
    A_cap_B = len(A.intersection(B))
    A_cup_B = len(A.union(B))
    return A_cap_B / (A_cup_B)

class tversky_class:
    def __init__(self, alfa=0.5, beta=0.5):
        self.alfa = alfa
        self.beta = beta
    def __call__(self, A, B):
        A_cap_B = len(A.intersection(B))
        A_not_B = len(A.difference(B))
        B_not_A = len(B.difference(A))
        return A_cap_B / (A_cap_B + self.alfa*A_not_B + self.beta*B_not_A)
tversky = tversky_class()

def make_shinglets(seq, shingleton_length):
    s=set()
    for i in range(0, len(seq)-shingleton_length):
        s.add(seq[i:i+shingleton_length].encode('utf8'))
    return s

def make_shinglets_intersection(seq, shingleton_length, hpv_shinglets):
    s=set()
    for i in range(0, len(seq)-shingleton_length):
        shinglet = seq[i:i+shingleton_length].encode('utf8')
        if shinglet in hpv_shinglets: s.add(shinglet)
    return s


filename="hpv_viruses.fasta"
def main(shingleton_length=18):
    print(f"---- {shingleton_length=} ----", flush=True)
    print("Reading HPV virus file", flush=True)
    viruses={}
    virus=[]
    name=""
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line or line.startswith(">"):
                if len(virus)>0:
                    viruses[name]=virus
                if not line: break
                name = line.split("|")[0][1:]
                virus = ""
            else:
                virus += line.strip()

    print("Creating shinglets from HPV viruses", flush=True)
    sets={}
    minhashes={}
    superset=set()
    for name in viruses:
        virus = viruses[name]
        sets[name]=make_shinglets(virus, shingleton_length)
        superset=superset.union(sets[name])

    import pandas as pd
    metadata = pd.read_csv("10904_metadata.csv")
#    cervix=pd.DataFrame(metadata[(metadata["site_of_resection_or_biopsy"] == "Cervix uteri")])

    print("Reading *.fq.gz files", flush=True)
    import gzip, os
    for fn in metadata["sample"]:
        if not os.path.exists(fn): continue
        print(f"- {fn}: reading...", flush=True, end="")
        name = fn.split(".", 1)[0]
        sample = ""
        with open(fn, "rb") as fq_gz_file:
            with gzip.open(fq_gz_file, "rt") as fq_file:
                while True:
                    fq_file.readline()
                    line = fq_file.readline()
                    if not line: break
                    sample += line.strip()
                    fq_file.readline()
                    fq_file.readline()

        print(" shinglets...", flush=True, end="")
        shinglets = make_shinglets_intersection(sample, shingleton_length, superset)

        print(" crosscheck...", flush=True)
        s = []
        max_t,max_name = 0, ""
        for hpv_name in sorted(sets.keys()):
            hpv_shinglets = sets[hpv_name]
            t = jaccard(shinglets, hpv_shinglets)
            s += [t]
            if t>max_t: max_t, max_name = t, hpv_name

        csv_hpv=metadata[metadata["sample"] == fn]["diamond_most_abundant"].iloc[0]
        csv_label=metadata[metadata["sample"] == fn]["positivity"].iloc[0]
        print(f"{name:12s}  {min(s)=:.4f}   {max(s)=:.4f}   {np.mean(s)=:.4f}   {np.median(s)=:.4f}   {np.std(s)=:.4f}   {np.var(s)=:.4f}  {csv_label=!s:5s}  {max_name=:10s}  {csv_hpv=!s:10s}", flush=True)

if __name__=="__main__":
    main()

