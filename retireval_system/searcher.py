from retireval_system.utils import text_preprocessing, word_preprocessing

class Searcher():
    def __init__(self, inverted_dict):
        self.inverted_dict = inverted_dict
        self.text_preprocessing = text_preprocessing
        self.word_preprocessing = word_preprocessing

    def search(self, query):
        query = text_preprocessing(query)
        answears = []
        for word in query:
            answears.append(set(self.inverted_dict.get(word, [])))
        answear = list(set.intersection(*answears))
        
        return answear, len(answear)
