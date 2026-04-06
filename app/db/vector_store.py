class VectorStore:
    def __init__(self):
        self.data = []
    
    def add(self, vector, text):
        self.data.append((vector, text))
        
    def search(self, query_vector):
        # DUMMY SEARCH
        return [text for _, text in self.data]