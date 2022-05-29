# The class Triple is a subclass of dict. It has a constructor that takes in a subject, attribute,
# object, stage, and themes. It also has a few methods that return the tags, stage, themes, and input.
class Triple(dict):
    """Semantic Triple"""
    def __init__(self, subject : str, attribute : str, object : str, stage: str, themes: list):
        """
        The function takes in a subject, attribute, object, stage, and themes and returns a triple with
        the input as the subject, attribute, and object concatenated together and the tags as the stage
        and themes
        
        :param subject: The subject of the triple
        :type subject: str
        :param attribute: the attribute of the subject
        :type attribute: str
        :param object: The object of the triple
        :type object: str
        :param stage: the stage of the project
        :type stage: str
        :param themes: list of strings
        :type themes: list
        """
        triple = {'input' : subject + " | " + attribute + " | " + object, 'tags' : {'stage' : stage, 'themes' : themes}}
        self = super().__init__(triple)

    def getTags(self):
        return self['tags']
    
    def getStage(self):
        return self['tags']['stage']

    def getThemes(self):
        return self['tags']['themes']

    def haveTheme(self,theme):
        return theme in self['tags']['themes']

    def haveStage(self, stage):
        return self['tags']['stage']==stage
    
    def haveNode(self, node):
        return self['input'].split("|")[0].strip() == node.strip() or self['input'].split("|")[2].strip() == node.strip()

    def getInput(self):
        return self['input']

        