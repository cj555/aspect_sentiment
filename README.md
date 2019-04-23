# Aspect Sentiment- CRFAspectSent

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

    @InAAAI{bailin-lu:2018:AAAI2018,
    author    = {Bailin, Wang  and  Lu, Wei},
    title     = {Learning Latent Opinions for Aspect-level Sentiment Classification},
    year      = {2018},
    abstract  = {Aspect-level sentiment classification aims at detecting the sentiment expressed 
    towards a particular target in a sentence. Based on the observation that the sentiment 
    polarity is often related to specific spans in the given sentence, it is possible to make 
    use of such information for better classification. On the other hand, such information 
    can also serve as justifications associated with the predictions. We propose a segmentation 
    attention based LSTM model which can effectively capture the structural dependencies between 
    the target and the sentiment expressions with a linear-chain conditional random field (CRF) layer. 
    The model simulates human’s process of inferring sentiment information when reading: when given a                
    target, humans tend to search for surrounding relevant text spans in the sentence before making an 
    informed decision on the underlying sentiment information. We perform sentiment classification tasks
    on publicly available datasets on online reviews across different languages from SemEval tasks and
    social comments from Twitter. Extensive experiments show that our model achieves the 
    state-of-the-art performance while extracting interpretable sentiment expressions.}
    }