# Prediction Churn Reduction

## Motivation


## Milestones
1. [x] Experiments and existing techniques implementation
2. [ ] Python package creation
3. [ ] Novel prediction churn reduction method 

## Experiments

In order to test churn reduction methods we conducted experiments using textual, tabular, and images data. For more details please look at README files:

1. [Tabular data experiment](experiments/tabular/README.md)

### Metrics
1. $`Churn(f_{old}, f_{new}) = \mathbb{E}_{(X, Y) \sim D}{[\mathbb{1}_{f_{old}(X) \neq f_{new}(X)}]}`$
2. $`ChurnRatio = \frac{Churn(f_{teacher}, f_{student})}{Churn(f_{teacher}, f_{baseline})}`$
3. $`GoodChurn(f_{old}, f_{new}) = \mathbb{E}_{(X, Y) \sim D}{[\mathbb{1}_{f_{old}(X) \neq Y = f_{new}}]}`$
4. $`BadChurn(f_{old}, f_{new}) = \mathbb{E}_{(X, Y) \sim D}{[\mathbb{1}_{f_{old}(X) = Y \neq f_{new}}]}`$
5. $`WinLossRatio = \frac{GoodChurn(f_{old}, f_{new})}{BadChurn(f_{old}, f_{new})}`$

### Models
- Images 
- Text
- Tabular: [Fully Connected Neural Network](experiments/tabular/models.py)

### Results


## Resources
1. [Churn Reduction via Distillation](https://arxiv.org/pdf/2106.02654.pdf)
2. [Launch and Iterate: Reducing Prediction Churn](https://papers.nips.cc/paper/2016/file/dc5c768b5dc76a084531934b34601977-Paper.pdf)
3. [Locally Adaptive Label Smoothing for Predictive Churn](https://arxiv.org/pdf/2102.05140.pdf)
4. [Knowledge Distillation - Keras implementation](https://keras.io/examples/vision/knowledge_distillation/) 

## Contributors
[Dauren Bizhanov](https://linkedin.com/in/dauren-bizhanov)  
[Himangshu raj Bhatntana]()  
[Satvik Kishore]()  
[Tigran Harutyunyan](https://linkedin/in/tigran-harutyunyan)  

