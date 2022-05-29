from glob import glob
import re
import pandas as pd
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from os import remove,mkdir

from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV


# It downloads the WebNLG dataset, preprocesses it, and saves it as a csv file

class WebNLGDataset:

    def __init__(self, url = None):
        """
        It takes a url as an argument and returns a dictionary of dataframes
        
        :param url: The URL of the dataset. If not specified, the default URL is used
        """
        """
            Inicialización del conjunto de datos.
            Incluye tareas de preprocesamiento y limpieza de los datos de WebNLG.

            Devuelve los datos del conjunto de forma de diccionarios de pares -nombre del conjunto (train, test, dev)-DataFrame con los datos-.
        """

        self.URL_DATABASE = "https://gitlab.com/shimorina/webnlg-dataset/-/archive/master/webnlg-dataset-master.zip?path=release_v3.0/en"
        self.dataset = {'train' : [], 'test': {'test':[],'train':[]},'dev':[]}

        self.genDataset(url)
        
    
    def genDataset(self, url = None):
        """
        It takes the url of the dataset, extracts the data, preprocesses it, parses it, shuffles it, and
        saves it as a csv file
        
        :param url: The url of the dataset
        """

        data_url = self.checkUrl(url = url)

        self.extractAllData(data_url)

        data = self.preprocessAllData()

        for i in range(len(data)):
            data[i] = self.parseData(data[i])
            data[i] = self.randomShuffle(data[i]).drop_duplicates(subset=None, 
                                keep='first', 
                                inplace=False, 
                                ignore_index=False)

        mkdir('data/cleaned')
        self.save_csv(data[0],'data/cleaned/webNLG2020_train.csv')
        self.save_csv(data[1],'data/cleaned/webNLG2020_dev.csv')
        self.save_csv(data[2],'data/cleaned/webNLG2020_test.csv')
        self.save_csv(data[3],'data/cleaned/webNLG2020_testtrain.csv')

        self.dataset['train'] = data[0]
        self.dataset['dev'] = data[1]
        self.dataset['test']['test'] = data[2]
        self.dataset['test']['train'] = data[3]

        remove('data/webnlg.zip')


    def preprocessAllData(self):
        """
        It takes the raw data from the webNLG2020 dataset and preprocesses it into a csv file.
        :return: The return value is a list of 4 dataframes.
        """
        remove('data/webnlg/webnlg-dataset-master-release_v3.0-en/release_v3.0/en/test/rdf-to-text-generation-test-data-without-refs-en.xml')
        remove('data/webnlg/webnlg-dataset-master-release_v3.0-en/release_v3.0/en/test/semantic-parsing-test-data-with-refs-en.xml')
        
        sourceURL = "data/webnlg/webnlg-dataset-master-release_v3.0-en/release_v3.0/en/"

        mkdir('data/parsed')
        self.preprocessData(sourceURL+"train/**/*.xml","data/parsed/webNLG2020_train.csv", typefile='train')
        self.preprocessData(sourceURL+"dev/**/*.xml","data/parsed/webNLG2020_dev.csv", typefile='train')

        self.preprocessData(sourceURL+"test/*.xml","data/parsed/webNLG2020_test.csv", typefile='test')
        self.preprocessData(sourceURL+"train/**/*.xml","data/parsed/webNLG2020_testtrain.csv", typefile='test')

        return [self.load_csv('data/parsed/webNLG2020_train.csv'),self.load_csv('data/parsed/webNLG2020_dev.csv'),
                self.load_csv('data/parsed/webNLG2020_test.csv'),self.load_csv('data/parsed/webNLG2020_testtrain.csv')]
    

    def preprocessData(self, sourceUrl, targetUrl = 'webNLG2020.csv', typefile ="train"):
        """
        It takes the sourceUrl, which is the path to the XML file, and the targetUrl, which is the path
        to the CSV file, and the typefile, which is either "train" or "test". 
        
        It then creates a list of lists, where each list contains the input text and the target text. 
        
        It then creates a dictionary, where the keys are the column names and the values are the lists
        of input and target text. 
        
        It then creates a dataframe from the dictionary, and saves it as a CSV file.
        
        :param sourceUrl: The path to the folder containing the XML files
        :param targetUrl: the path to the output file, defaults to webNLG2020.csv (optional)
        :param typefile: train or test, defaults to train (optional)
        """
        files = glob(sourceUrl, recursive=True)
        data_list=[]

        for file in files:
            tree = ET.parse(file)
            root = tree.getroot()
            for entries in root: #entries
                for entry in entries: #entry
                
                    structure_master=[]
                    unstructured= []

                    if(typefile=="train"):
                        m = entry.findall("modifiedtripleset")
                    else:
                        m = entry.findall("modifiedtripleset")
                        
                    for modifiedtripleset in m: 
                        triples= (' && ').join([triple.text for triple in modifiedtripleset])
                        structure_master.append(triples)

                    
                    for lex in entry.findall("lex"): 
                        unstructured.append(lex.text)

                    triples_num = int(entry.attrib.get("size"))

                    if(typefile=="train"):
                        for text in unstructured:
                            for triple in structure_master:
                                data_list.append([triple,text])
                    else:
                        for structure in structure_master:
                            data_list.append([structure,unstructured])

        mdata_dct={"input_text":[], "target_text":[]}

        for item in data_list:
            mdata_dct['input_text'].append(item[0])
            mdata_dct['target_text'].append(item[1])

        df=pd.DataFrame(mdata_dct)

        df.to_csv(targetUrl)


    def load_csv(self, sourceUrl):
        """
        It reads a csv file from a given url and returns a dataframe.
        
        :param sourceUrl: The URL of the CSV file to load
        :return: A dataframe
        """
        return pd.read_csv(sourceUrl, index_col=[0])


    """
    - Remove @en
    - Change _ to ' '
    - Remove urls
    - Split relations according to uppercase tokens 
    - Remove xsd:
    - Remove ^
    
    :param example: the text of the example
    :return: the parsed instance.
    """
    def parseInstance(self,example):

        # remove @en
        example = re.sub('@en','', example)

        # change _ to ' '
        example = re.sub('[_]',' ', example)
        example = re.sub('""',' ', example)
        example = re.sub('"',' ', example)
        example = re.sub("associatedBand/associatedMusicalArtist",'associatedBand',example)
        #remove urls
        example = re.sub("\<http.*[^\>]\>", '',example)

        # split relations according to uppercase tokens 
        triplets = re.split("&&", example)
        for triple in triplets:
          entity = re.split("\|", triple)[1]
          entity2 = entity[1].upper() + entity[2:]
          uppercase = re.findall(r'[A-Z](?:[A-Z]*(?![a-z])|[a-z]*)', entity2)
          if(len(uppercase)>1):
            uppercase = ' '.join(uppercase)
            example = re.sub("{}".format(entity), " {} ".format(uppercase.lower()), example) 

        example = re.sub(r"xsd:[^\s]*\s", "",example)
        example = re.sub(r"xsd:[^\s]*$", "",example)

        example = re.sub('\^','', example)
        return example

    def parseTarget(self,example):
        """
        It replaces all periods with a space and a period, replaces all commas with a space and a comma,
        and replaces all left and right parentheses with a space and a left or right parenthesis
        
        :param example: the string to be parsed
        :return: the example with the punctuation replaced with spaces.
        """
        example = re.sub(r"\.", " .",example)
        example = re.sub(r"F .C .", "F.C.", example)
        example = example.replace(',',' ,')
        example = example.replace('(','( ')
        example = example.replace(')',' )')
        example = example.replace('""',' ')
        example = example.replace('"',' ')
        return example


    def parseData(self,df):
        """
        It takes a dataframe as input, and returns a dataframe with two columns: input_text and
        target_text. 
        
        The input_text column is a list of lists of strings. Each list of strings is a list of words in
        a sentence. 
        
        :param df: the dataframe that contains the input and target text
        :return: The dataframe with the input_text and target_text columns parsed.
        """
        df['input_text'] = df['input_text'].map(self.parseInstance)
        df['target_text'] = df['target_text'].map(self.parseTarget)
        return df


    """
    > This function takes a dataframe as input and returns a dataframe with the rows shuffled randomly
    
    :param df: The dataframe to be shuffled
    :param random_state: The seed used by the random number generator, defaults to 13 (optional)
    :return: A dataframe with the same number of rows as the original dataframe, but with the rows in a
    random order.
    """
    def randomShuffle(self, df, random_state = 13):
        return df.sample(frac = 1, random_state = random_state)


    def save_csv(self, df, targetUrl):
        """
        It takes a dataframe and a target url, and saves the dataframe as a csv file at the target url
        
        :param df: the dataframe you want to save
        :param targetUrl: The URL of the CSV file you want to download
        """
        df.to_csv(targetUrl)

    
    def checkUrl(self, url : str):
        """
        If the url is None, then return the URL_DATABASE, otherwise return the url
        
        :param url: The URL of the database
        :type url: str
        :return: The url is being returned.
        """
        if url is None:
            url = self.URL_DATABASE
        return url
   

   
    def extractAllData(self,url):
        """
        It downloads the zip file from the url, and then extracts the contents of the zip file into the
        data/webnlg folder
        
        :param url: The URL of the zip file to download
        """
        urllib.request.urlretrieve(url, 'data/webnlg.zip')

        with zipfile.ZipFile('data/webnlg.zip', 'r') as zip_ref:
            zip_ref.extractall('data/webnlg')



if __name__ == "__main__":
    dataset = WebNLGDataset()