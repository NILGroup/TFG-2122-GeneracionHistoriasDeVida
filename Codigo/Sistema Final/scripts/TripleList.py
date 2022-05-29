
from .Triple import Triple

# It's a list of triples, with some methods to filter the list by node, stage, or theme.
class TripleList(list):
    
    def __init__(self, data):
        """
        It takes a list of lists of strings, and returns a list of lists of strings, where each list of
        strings is a triple
        
        :param data: a list of lists, where each list is a list of strings
        :return: The object itself.
        """
        tripleList = self.__generateTriples(data)
        self = super().__init__(tripleList)
        return self


    #OTHER PUBLIC METHODS
    def triplesByTags(self):
        """
        It takes a list of triples and returns a dictionary of dictionaries of dictionaries of lists
        :return: A dictionary of dictionaries of dictionaries of lists.
        """
        triplesByTags = {}
    
        for triple in self:
            stage = triple.getStage()

            if stage not in triplesByTags:
                triplesByTags[stage] = {}
                triplesByTags[stage]['tag'] = {}         

            predecesor = triplesByTags[stage]
            for theme in triple.getThemes()[:-1]:
                p = predecesor['tag']
                if theme not in p:
                    p[theme] = {}

                if 'tag' not in p[theme]:
                    p[theme]['tag'] = {}

                predecesor = p[theme]
            
            theme = triple.getThemes()[-1]

            if theme not in predecesor['tag']:
                predecesor['tag'][theme] = {}
            
            if 'input' not in predecesor['tag'][theme]:
                predecesor['tag'][theme]['input'] = []

            predecesor['tag'][theme]['input'].append(triple.getInput())

        return triplesByTags

    def filterbyNode(self, node):
        """
        It takes a node and returns a list of triples that have that node
        
        :param node: a node object
        :return: A list of triples that have the node in them.
        """
        filter_triples = TripleList([])
        for triple in self:
            if(triple.haveNode(node)):
                filter_triples.append(triple)

        return filter_triples

    def filterbyStage(self, stage):
        """
        It takes a list of triples and returns a list of triples that have a certain stage
        
        :param stage: The stage to filter by
        :return: A list of triples that have the stage specified.
        """
        filter_triples = TripleList([])
        for triple in self:
            if(triple.haveStage(stage)):
                filter_triples.append(triple)

        return filter_triples

    def filterbyTheme(self, theme):
        """
        It takes a list of triples and returns a list of triples that have a certain theme
        
        :param theme: a string
        :return: A list of triples that have the theme.
        """
        filter_triples = TripleList([])
        for triple in self:
            if(triple.haveTheme(theme)):
                filter_triples.append(triple)

        return filter_triples
      
    def getTagThemes(self):
        """
        It returns a set of all the themes in the triple
        :return: A set of all the themes in the triple.
        """
        aux = set()
        for triple in self:
            for i in triple.getThemes():
                aux.add(i)
        return aux

    def getTagTrees(self):
        """
        It takes a list of triples and returns a dictionary 
        :return: A dictionary
        """
        triplesByTags = {}
    
        for triple in self:
            stage = triple.getStage()

            if stage not in triplesByTags:
                triplesByTags[stage] = {}      

            predecesor = triplesByTags[stage]
            for theme in triple.getThemes()[:-1]:
                p = predecesor
                if theme not in p:
                    p[theme] = {}


                predecesor = p[theme]
            
            theme = triple.getThemes()[-1]

            if theme not in predecesor:
                predecesor[theme] = {}
            

        return triplesByTags

    def getTagStages(self):
        """
        It returns a list of all the stages in the list of triples
        :return: A list of the stages of the triples in the list.
        """
        aux = list()
        for triple in self:
            if triple.getStage() not in aux:
                aux.append(triple.getStage())
        return aux

        
    #LIST METHODS
    def __add__(self, other):
        """
        The function takes two TripleList objects and returns a new TripleList object that is the
        concatenation of the two
        
        :param other: TripleList
        :return: A TripleList object
        """
        return TripleList(list.__add__(self,other))

    def __getslice__(self,i,j):
        """
        It returns a new TripleList object that contains the elements of the original TripleList object
        between the indices i and j
        
        :param i: The start index of the slice
        :param j: The end index of the slice
        :return: A new TripleList object.
        """
        return TripleList(list.__getslice__(self, i, j))

    #PRIVATE METHODS
    def __generateTriples(self, data):
        """
        It takes a list of dictionaries, and returns a list of objects
        
        :param data: a list of dictionaries, each dictionary containing the following keys:
        :return: A list of triples
        """
        triples = []
        for item in data:
            try:
                triple = Triple(item['source'], item['relation'], item['target'], item['stage'], item['themes'])
            
                triples.append(triple)
            except:
                print("ERROR: " + str(item))

        return triples
        
    
        