from glob import glob
import re
import pandas as pd
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

KELM            = "https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-train.tsv"
DART            = "https://raw.githubusercontent.com/Yale-LILY/dart/master/data/v1.1.1/dart-v1.1.1-full-train.json"
WEBNLG          = "https://gitlab.com/shimorina/webnlg-dataset/-/archive/master/webnlg-dataset-master.zip?path=release_v3.0/en/train"


class Dataset:

    def __init__(self):
        self.train_set_df = []


    def WebNLG_parser(self, url = WEBNLG):
        urllib.request.urlretrieve(url, 'data/webNLG.zip')
        with zipfile.ZipFile('data/webNLG.zip', 'r') as zip_ref:
            zip_ref.extractall('data/webNLG')
        
        files = glob("data/webNLG/webnlg-dataset-master-release_v3.0-en-train/release_v3.0/en/train/**/*.xml", recursive=True)
        triple_re=re.compile('(\d)triples')
        data_dct={}
        for file in files:
            tree = ET.parse(file)
            root = tree.getroot()
            triples_num=int(triple_re.findall(file)[0])
            for sub_root in root:
                for ss_root in sub_root:
                    strutured_master=[]
                    unstructured=[]
                    for entry in ss_root:
                        unstructured.append(entry.text)
                        strutured=[triple.text for triple in entry]
                        strutured_master.extend(strutured)
                    unstructured=[i for i in unstructured if i.replace('\n','').strip()!='' ]
                    strutured_master=strutured_master[-triples_num:]
                    strutured_master_str=(' && ').join(strutured_master)
                    data_dct[strutured_master_str]=unstructured

        mdata_dct={"prefix":[], "input_text":[], "target_text":[]}

        for st,unst in data_dct.items():
            for i in unst:
                mdata_dct['prefix'].append('webNLG')
                mdata_dct['input_text'].append(st)
                mdata_dct['target_text'].append(i) 
                  
        df = pd.DataFrame(mdata_dct)
        df.to_csv('data/webNLG2020_train.csv')

        self.train_set_df = pd.read_csv('data/webNLG2020_train.csv', index_col=[0])