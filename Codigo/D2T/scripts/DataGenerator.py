import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json


class DataGenerator:

    def __init__(self):
        self.graph_df = []
        self.graph = []
        self.graph_json = []
        self.triples = []


    #PUBLIC METHODS
    def generate(self): 

        data = self.__gen_data()
        self.graph_df,self.graph = self.__gen_graph(data)
        self.graph_json= self.__df_to_json(self.graph_df)
        self.triples = self.__gen_triples()
         

    def getTriples(self):
        return self.triples

    def drawGraph(self):
        plt.figure(figsize=(12,12))

        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)

        nx.draw_networkx_edge_labels(self.graph, pos)
        plt.show()

    def generateFile(self,path = "data/shakespeare_data.json"):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.graph_json, f, ensure_ascii=False, indent=4)



    #PRIVATE METHODS
    def __gen_data(self):
        data = []

        data.append({"data":["Shakespeare","name","William"],"tag":"basic"})
        data.append({"data":["Shakespeare","birthdate","April 23, 1564"],"tag":"basic"})
        data.append({"data":["Shakespeare","birthplace","Stratford-upon-Avon"],"tag":"basic"})
        data.append({"data":["Stratford-upon-Avon", "city", "England"],"tag":"basic"})
        data.append({"data":["Shakespeare","father's name", "John"],"tag":"basic"})
        data.append({"data":["Shakespeare","mother's name", "Mary"],"tag":"basic"})
        data.append({"data":["Shakespeare","number of siblings", "six"],"tag":"basic"})
        data.append({"data":["Shakespeare","is", "older brother"],"tag":"basic"})
        data.append({"data":["Shakespeare","little sister", "Anne"],"tag":"basic"})
        data.append({"data":["Shakespeare","little sister", "Joan"],"tag":"basic"})
        data.append({"data":["Shakespeare","little brother", "Gilbert"],"tag":"basic"})
        data.append({"data":["Shakespeare","little brother", "Richard"],"tag":"basic"})
        data.append({"data":["Shakespeare","little brother", "Edmund"],"tag":"basic"})


        data.append({"data":["John","occupation","leatherworker"],"tag":"family"})
        data.append({"data":["John","occupation","prosperous businessman"],"tag":"family"})
        data.append({"data":["John", "married", "Mary"],"tag":"family"})
        data.append({"data":["Mary", "family name", "Arden"],"tag":"family"})
        data.append({"data":["John","move on","Stratford"],"tag":"family"})
        data.append({"data":["John","move on","in 1569"],"tag":"family"})
        data.append({"data":["Stratford","city","England"],"tag":"family"})


        data.append({"data":["Shakespeare","citizen","Stratford"],"tag":"school"})
        data.append({"data":["Shakespeare","attended at","Stratford's grammar school"],"tag":"school"})
        data.append({"data":["Stratford's grammar school","location","Stratford"],"tag":"school"})
        data.append({"data":["Stratford's grammar school","subject","Latin classics"],"tag":"school"})
        data.append({"data":["Stratford's grammar school","subject","memorization"],"tag":"school"})
        data.append({"data":["Stratford's grammar school","writing","memorization"],"tag":"school"})
        data.append({"data":["Stratford's grammar school","subject","acting classic Latin plays"],"tag":"school"})
        data.append({"data":["Shakespeare","left at 15 years old","Stratford's grammar school"],"tag":"school"})
        
        return data

    def __gen_graph(self,data):
        entities = []
        relations = []

        for d in data:
            entities.append((d['data'][0],d['data'][2]))
            relations.append({'relation':d['data'][1],'tag':d['tag']})

        # extract subject and object
        source = [i[0] for i in entities]

        # extract relation
        target = [i[1] for i in entities]

        # create dataframe from 
        graph_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

        # create a directed-graph from a dataframe
        graph = nx.from_pandas_edgelist(graph_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

        return graph_df, graph


  
    def __df_to_json(self,graph_df):

        buf = graph_df.to_json(orient="records")

        return json.loads(buf)

    def __gen_triples(self):
        data=[]
        for line in self.graph_json:
            sentence = line['source'] + " | " + line['edge']['relation'] + " | " + line['target']
            data.append({'tag':line['edge']['tag'], 'input':sentence})

        return data

    



