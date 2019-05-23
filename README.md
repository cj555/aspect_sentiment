# aspect_sentiment

## Survey on Domain Apaption (DA)

1. Co-regularized Alignment for Unsupervised Domain Adaptation [[Paper,NeurIPS2018]](http://papers.nips.cc/paper/8146-co-regularized-alignment-for-unsupervised-domain-adaptation.pdf)

2. Strong Baselines for Neural Semi-supervised Learning under Domain Shift [[paper, ACL2018]](https://arxiv.org/abs/1804.09530) [[code, dynet]](https://github.com/bplank/semi-supervised-baselines)

  >This paper reviews several neural bootstrapping methods like self-training, tri-training, tri-training with disgreement and proposed a multi-task tri-training (MT-Tri) model. The author applies the above methods on POS tagging (SANCL 2012) and Sentiment anlaysis(Amazon reviews) and shows that classic tri-training works the best and even outperforms a recent state-of-the-art method. 

  
3. Semi-Supervised sequence Modeling with Cross-View Training [[Paper, EMNLP2018]](https://arxiv.org/abs/1809.08370) 
4. Word Translation Without Parallel Data [[Paper, ICLR2017]](https://arxiv.org/abs/1710.04087) [[code,pytorch]](https://github.com/balasrini32/CSE293_NLP) [[code,pytorch]](https://github.com/facebookresearch/MUSE)
  > The word translations with parallel data boils down to the Procrustes problems. One can learn the mapping W with SVD. However, to find the word translations without parallel data is difficult. This paper proposed a domain-adversarial approach  (Adv) for learning W. The model contains a discriminator to discriminate between elements randomly sampled from WX (X is source, Y is target). To produce reliable matching pairs between two languages, the paper use cross-domain simiarity local scaling (CSLS) instead of nearest neighbors. The unspervision method with Adv shows comparable result with supervison method.


5. Multi-class Classification without Multi-class Labels [[Paper, ICLR2019]](https://arxiv.org/pdf/1901.00544.pdf)

6. A DIRT-T Approach to Unsupervised Domain Adaptation [[Paper, ICLR2018]](https://arxiv.org/abs/1802.08735)[[code, tensorflow]](https://github.com/RuiShu/dirt-t)

7. LEARNING DEEP REPRESENTATIONS BY MUTUAL INFORMATION ESTIMATION AND MAXIMIZATION [[Paper, ICLR2019]](https://arxiv.org/pdf/1808.06670.pdf)

8. Unsupervised Learning by Predicting Noise [[Paper, ICML2017]](https://arxiv.org/pdf/1704.05310.pdf) [[Code, Lua]](https://github.com/facebookresearch/noise-as-targets)
 >> In supervised settings, the learning process is to find the optimal functions to minimize the distance between features and targets. In unsupervised settings, the targets representation is missing. In this paper, the target Y = PC. where P is a Assignment matrix (only zeror or one)and C is a predefined features. For C, this paper uses uniformly samples from L2 unit spheres. The author explains that canonical basis's underlying assumption that each image belongs to one catergory is too strong is this task.  

9. META-LEARNING UPDATE RULES FOR UNSUPERVISED REPRESENTATION LEARNING [[Paper, ICLR2019]](https://openreview.net/pdf?id=HkNDsiC9KQ)

10. Learning Domain Representation for Multi-Domain Sentiment
Classification[[Paper, naacal2018]](https://leuchine.github.io/papers/naacl18sentiment.pdf) [[Code,tensorflow]](https://github.com/leuchine/multi-domain-sentiment/blob/master/multi_view_domain_embedding_memory_adversarial.py)

11. Aspect-augmented Adversarial Networks for Domain Adaptation [[Paper, ACL2017]](https://aclweb.org/anthology/Q17-1036)
[[Code, theano]](https://github.com/yuanzh/aspect_adversarial)

12. Transferrable Prototypical Networks for Unsupervised Domain Adaptation [[Paper,CVPR2019_Oral]](https://arxiv.org/abs/1904.11227)
13. Bidirectional Learning for Domain Adaptation of Semantic Segmentation [[Paper, CVPR2019]](https://arxiv.org/abs/1904.10620) 
14. Phrase-Based & Neural Unsupervised Machine Translation [[Paper, EMNLP2018]](https://arxiv.org/pdf/1804.07755.pdf)
15. Hierarchical Attention Transfer Network for Cross-Domain Sentiment Classification [[Paper, AAAI2018]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16873) [[Code,tensorflow]](https://github.com/hsqmlzno1/HATN)

16. Interactive Attention Transfer Network for Cross-domain Sentiment Classification [[Paper,AAAI2019]](http://staff.ustc.edu.cn/~cheneh/paper_pdf/2019/Kai-Zhang-AAAI.pdf)
    
    >>  Gradient Reversal Layer (GRL) (Ganin and Lempitsky 2014; Ganin et al. 2016) to reverse the gradient direction in the training process. 

17. Exploiting Coarse-to-Fine Task Transfer for Aspect-level Sentiment Classification [[Paper, AAAI2018]](https://arxiv.org/pdf/1811.10999.pdf)
  >> This paper Proposes to use Aspect Categories (AC) to train Aspect Term (AT) task. To do so, C2A (Catories to aspect) layer is trained to label the aspect in Aspect Categories Dataset. Finally the feature is aligned by Contrastive Feature aligment (CFA)

18. Learning What and Where to Transfer [[Paper, ICML2019]](https://arxiv.org/abs/1905.05901) [[Code, Pytorch]](https://github.com/alinlab/L2T-ww)
19. Multi-Domain Neural Machine Translation with Word-Level Domain Context Discrimination [[Paper, ACL2018]](https://www.aclweb.org/anthology/D18-1041) [[Code,Theano]](https://github.com/DeepLearnXMU/WDCNMT)

20. Hierarchical Attention Generative Adversarial Networks for Cross-domain Sentiment Classification [[Paper, ]](https://arxiv.org/pdf/1903.11334.pdf)

## Survey On Graph neural network

1. How Powerful are Graph Neural Networks? [[Paper, ICLR2019]](https://openreview.net/pdf?id=ryGs6iA5Km)
  >> This paper uses Weisfeiler-Lehman test to measure the expressive power of GNN variants such as GCN and GraphSAGE and found that the above variants that map different neighborhoods to the same representations, hense not maximally powerful GNNs. The authors then propose GIN and then measure the overfitting on node and graph classification tasks. Different GNNs follow a different negithborhood aggregation strategy namely aggregate and combine function in eq (2.1)

## Survey on ASC 

1. Enhanced Aspect Level Sentiment Classification with Auxiliary Memory [[Paper, ACL2018]](https://www.aclweb.org/anthology/C18-1092)

2. Learning Latent Opinions for Aspect-level Sentiment Classification [[Paper, AAAI2018]](http://www.statnlp.org/wp-content/uploads/papers/2018/Learning-Latent/absa.pdf) [[Code, Pytorch]](https://github.com/berlino/SA-Sent)

3. Targeted Aspect-Based Sentiment Analysis via Embedding
Commonsense Knowledge into an Attentive LSTM [[Paper, AAAI2018]](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16541/16152)

4. Utilizing BERT for Aspect-Based Sentiment Analysis
via Constructing Auxiliary Sentence [[Paper, AAAI2019]](https://arxiv.org/pdf/1903.09588.pdf) [[Code, Pytorch]](https://github.com/HSLCY/ABSA-BERT-pair)

## others
1. An Attentive Survey of Attention Models [[Paper, IJCAI2019]](https://arxiv.org/abs/1904.02874)

## semi-supervised learning

1. MixMatch: A Holistic Approach to Semi-Supervised Learning [[Paper, ]](https://arxiv.org/pdf/1905.02249.pdf) [[Code, TensorFlow]](https://github.com/google-research/mixmatch) [[中文]](https://zhuanlan.zhihu.com/p/66281890）




# Other Resources

- [transferlearning](https://github.com/jindongwang/transferlearning)
- [awsome-domain-adaptation](https://github.com/zhaoxin94/awsome-domain-adaptation)
- [awesome-transfer-learning](https://github.com/artix41/awesome-transfer-learning)
- [barebell domain-adaption](https://github.com/barebell/DA/blob/master/README.md)
