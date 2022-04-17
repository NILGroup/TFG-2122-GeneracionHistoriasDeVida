from glob import glob
import re
from tabnanny import check
import pandas as pd
import urllib.request
import zipfile
import xml.etree.ElementTree as ET



class Dataset:

    def __init__(self):
        self.__DEFAULT_URL = "https://gitlab.com/shimorina/webnlg-dataset/-/archive/master/webnlg-dataset-master.zip?path=release_v3.0/en/train"
        self.__DEFAULT_LOAD_CSV_URL = "data/webNLG2020_train.csv"
        self.train_set_df = []
        
    
    
    def genDataset(self, url = None):

        data_url = self.__checkUrl(url = url)

        self.__extractAllData(data_url = data_url)

        data_dct = self.__importData()
        
        df = self.__generateDataFrame(data_dct = data_dct)

        df.to_csv('data/webNLG2020_train.csv')


    def importDataset(self, csv_url = None):

        csv_url = self.__checkCSVUrl(csv_url = csv_url)
        self.train_set_df = pd.read_csv(csv_url, index_col=[0])

    
    

    
    def  __checkUrl(self, url : str):
        if url is None:
            url = self.__DEFAULT_URL
        return url

    def  __checkCSVUrl(self, csv_url : str):
        if csv_url is None:
            csv_url = self.__DEFAULT_LOAD_CSV_URL
        return csv_url
    

    def __extractAllData(self, data_url : str):
        urllib.request.urlretrieve(data_url, 'data/webNLG.zip')

        with zipfile.ZipFile('data/webNLG.zip', 'r') as zip_ref:
            zip_ref.extractall('data/webNLG')



    def __importData(self):
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

        return data_dct

    
    def __generateDataFrame(self, data_dct):
        
        mdata_dct={"prefix":[], "input_text":[], "target_text":[]}        

        for st,unst in data_dct.items():
            for i in unst:
                mdata_dct['prefix'].append('webNLG')
                mdata_dct['input_text'].append(st)
                mdata_dct['target_text'].append(i) 
                  
        df = pd.DataFrame(mdata_dct)

        return df


    
