import pickle;


class ModelHolder:
    
    def __init__(self, model, most_dependent_columns):
        self.model = model
        self.most_dependent_columns = most_dependent_columns
    
    def get(self):
        return self.model, self.most_dependent_columns
    
    def save(self, filename):
        with open(filename, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)


def load_model(model_file):
    file = open(model_file, 'rb')
    mh = pickle.load(file)
    file.close()
    return mh;
