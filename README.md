# aspect_sentiment tri-learning

###  Documentation

#### Data
change configs in data.py 
    
    
    class DataConfig(object):
    def __init__(self):
        self.data_path = ''
        self.train = ''
        self.dev = ''
        self.test = ''
        self.output_name = ''
        self.embed_num =   # most freq words
        self.embed_dim = 
        self.pretrained_embed_path = ''
        self.is_stanford_nlp = False
        self.batch_size = 
        self.pickle_path = self.data_path + self.output_name + '.pkl'

preprocess with python data.py
