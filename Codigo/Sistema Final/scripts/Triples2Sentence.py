# It takes a list of triples and generates a sentence

class Triples2Sentence:

    def __init__(self, triples):
        """
        The function takes a list of triples as input and generates a sentence from the triples
        
        :param triples: a list of tuples containing the subject, verb, and object of the sentence
        """
        self.triples = triples
        self.text = self.__genSentence()

    def __genSentence(self):
        """
        It takes a list of triples and returns a string of triples separated by "&&"
        :return: The text of the sentence.
        """
        text = ""
        for triple in self.triples:
            text += triple+ " && "
                
        text = text[0:-3]

        return text

    def getText(self):
        """
        It takes the text from the text box and adds a period at the end of the sentence
        :return: WebNLG:The capital of the United Kingdom is London .</s>
        """
        return "WebNLG:" + self.text + "</s>"
