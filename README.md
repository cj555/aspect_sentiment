# Aspect Sentiment- Tri-Learning

###  Documentation

#### Data

Change configs in config.DataConfig and default.yaml
    
    
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
        
[statistics](https://docs.google.com/spreadsheets/d/18AfV2o6X47m8CaY0XKV24MgGeFJ_DqqGGdLNW9hGLVo/edit#gid=219973634)

Preprocess
    
    python data.py

#### Training
Run 

    python train.py



#### Reference