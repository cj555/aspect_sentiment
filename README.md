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

## Survey On Graph neural network

1. How Powerful are Graph Neural Networks? [[Paper, ICLR2019]](https://openreview.net/pdf?id=ryGs6iA5Km)
  >> This paper uses Weisfeiler-Lehman test to measure the expressive power of GNN variants such as GCN and GraphSAGE and found that the above variants that map different neighborhoods to the same representations, hense not maximally powerful GNNs. The authors then propose GIN and then measure the overfitting on node and graph classification tasks. Different GNNs follow a different negithborhood aggregation strategy namely aggregate and combine function in eq (2.1)


# Other Resources

- [transferlearning](https://github.com/jindongwang/transferlearning)
- [awsome-domain-adaptation](https://github.com/zhaoxin94/awsome-domain-adaptation)
- [awesome-transfer-learning](https://github.com/artix41/awesome-transfer-learning)
- [barebell domain-adaption](https://github.com/barebell/DA/blob/master/README.md)
