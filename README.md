&emsp;主要记录推荐系统学习的过程，包含相关的论文和代码。



# <font size=4>1. 点击率预估</font>

> GBDT + LR

论文: [Practical Lessons from Predicting Clicks on Ads at Facebook](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)

&emsp;本文发表于2014年，不同于传统人工特征工程的思想，本文提出通过GBDT对特征进行组合，生成组合特征喂给LR分类模型，通过机器学习特征，提高模型的准确率。同时，还提出了不同模型不同更新周期的思想。此外，本文还提出了一些工程化的优化方法，比如在线学习方法，使用data joiner，基于分布式架构将impression记录和click记录进行join。为了防止data joiner失效，还采用了数据流保护机制。对于CTR预估场景正负样本不均衡的问题，使用了降采样的方法，包括`uniform subsampling`和`negative downsampling`。

------

> FM和FFM

博客：[美团技术：深入FFM原理与实践](https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html)

&emsp;这篇博客比较深入的讲解了FM和FFM的原理以及在工程上的应用。相比原论文，博客的推导过程更清晰简洁，容易理解，同时介绍了一些工程上的优化方法。

------

## <font size=3>1.1 并行FM</font>

> DeepFM

论文：[A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)

&emsp;本文发表于2017年。虽然FM和FFM可以将特征进行组合，但是由于复杂度的问题，一般只进行二阶交叉组合，想要得到更高阶的组合时间和空间开销将剧增。于是，考虑使用神经网络这种高阶特征提取器。首先将高维稀疏特征embedding到相同长度的意向两种，而实际上FM中得到的隐向量`$v_{ik}$`就是embedding层的权重。将不同field的特征进行拼接，经过多个线性层就得到了高阶特征组合结果。**最后，将线性结果、FM结果和Deep结果进行concat，经过输出层得到最终结果**。

------

> DCN

论文：[Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf)

&emsp;本文发表于2017年。将特征分为类别性和数值型，类别型特征经过embedding之后与数值型特征直接拼接作为模型的输入。所有特征分别经过cross和deep网络，两个网络可以看做特征提取器，经过提取的特征向量拼接之后是常规的二分类。

------

> Wide & Deep

论文：[Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)

&emsp;本文发表于2016年。通常希望模型能够同时获得**记忆`memorization`和泛化`generalization`能力**。所谓记忆是指从历史数据中发现`item`或者特征之间的相关性；返回是指发现在历史数据中很少或者没有出现的新的特征组合。本文提出的Wide & Deep模型包含两个部分，Wide基于传统的特征工程和LR，使用FTRL优化算法得到结果，承担了记忆的部分；Deep部分采用深度神经网络，采用AdaGrad优化放大，承担了泛化的部分。最后，将两部分的结果加权得到最终结果，并进行联合训练。

## <font size=3>1.2 串行FM</font>

> **PNN**

论文：[Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf)

&emsp;本文发表于2016年。一般来说，则CTR预估领域的特征组合，“且”的方式比“和”的方式更恰当，“且”对应的就是求乘积，“和”对应的就是求和。于是，本文提出了基于Product的特征组合方式。网络共分为4层：Embedding层，Product Layer以及2个Hidden Layer。关键在于Product Layer，这一层用于组合特征。首先是线性特征的计算，采用常规的计算方法；然后就是特征组合，提出了**Inner Product**和**Outer Product**两种计算方法。**Inner Product**实际就是对不同field对应的特征两两进行内积计算得到一个向量再求和；**Outer Product**实际就是对不同field对应的特征两两相乘（其中一个转置），得到一个方阵，再求和。文章给出了两种计算方法的优化方法，同时提出可以将两种方法输出的向量拼接，再进入下一层，也就是**PNN***。

------

> **NFM**

论文：[Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)

&emsp;本文发表于2017年。实际上就是FM的改进版，将FM的结果经过DNN得到结果，再和线性结果以及偏置相加得到最终结果。

------

> **AFM**

论文：[Attentional Factorization Machines](https://arxiv.org/pdf/1708.04617.pdf)

&emsp;本文发表于2017年。仍然在FM的二阶交叉特征上做文章，考虑不同交叉特征对结果贡献度的区别，使用Attention机制对不同的交叉特征进行加权，最后和一阶特征以及偏置相加得到最终结果。

# <font size=3>1.3 其他</font>

> **NCF**

论文：[Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)

&emsp;本文发表于2017年，提出一种通过神经网络和广义矩阵分解相结合的方法，来预测用户对某些商品的隐性反馈。**广义矩阵分解`GMF`** 就是采用embedding对user和item进行嵌入，然后取出对应的embedding对应位置相乘，得到输出；**MLP部分**里另外生成embedding，将user和item的embedding进行concat，然后投入深度模型中计算，得到输出；最后将GMF和MLP的结果进行concat，得到最终输出CTR。