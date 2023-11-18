# Transfer Learning for Demand Prediction and Revenue Maximization

## Introduction
In the landscape of machine learning and predictive analytics, decision-makers are often constrained by the limited quantity of labeled "complete" data. This project addresses the challenge of predicting local store demand and maximizing revenue with incomplete feature spaces, as commonly seen in e-commerce to offline retail transitions. A novel hybrid transfer learning algorithm is proposed, aiming to learn transferable features and provide a sub-optimal pricing policy with objective optimality.

## Files Description
- `code/`: Directory containing all the source code files.
- `Data_preprocessing.ipynb`: Jupyter notebook for preprocessing the data used in the models.
- `baseline.py`: Python script implementing the first estimate then optimize framework using linear regression and Multi-Layer Perceptron (MLP) models.
- `draw_graph.py`: Python script to visualize results and draw graphs for analysis.
- `transfer.py`: Python script for implementing our transfer learning for end-to-end training using monotone neural networks.

## Documentation
- `E2E pricing.pdf`: A document providing a brief summary of the problem and the model used for end-to-end pricing strategy.

## Dataset
- `JDdata.zip`: Compressed file containing the real-world data utilized in our experiments.

## How to Use
1. Begin by preprocessing your dataset using the `Data_preprocessing.ipynb` notebook to fit the expected input format.
2. Run `baseline.py` to establish a baseline model for your prediction task.
3. Utilize `draw_graph.py` to generate visual representations of your data and results.
4. Apply `transfer.py` to perform the transfer learning-based end-to-end training with the proposed monotone neural network.
5. Consult `E2E pricing.pdf` for an in-depth understanding of the problem and the modeling approach.

## Contribution
We encourage contributions to this project. If you have suggestions or improvements, please fork the repository and submit a pull request.

## Citation
%DA
@article{ben2010theoryDA,
  title={A theory of learning from different domains},
  author={Ben-David, Shai and Blitzer, John and Crammer, Koby and Kulesza, Alex and Pereira, Fernando and Vaughan, Jennifer Wortman},
  journal={Machine learning},
  volume={79},
  number={1},
  pages={151--175},
  year={2010},
  publisher={Springer}
}
@inproceedings{adw,
  title={Decoupled Weight Decay Regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={International Conference on Learning Representations},
  year={2018}
}
@article{umnn,
  title={Unconstrained Monotonic Neural Networks},
  author={Wehenkel, Antoine and Louppe, Gilles},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  pages={1545--1555},
  year={2019}
}
@InProceedings{Ganin2015DA,
  title = 	 {Unsupervised Domain Adaptation by Backpropagation},
  author = 	 {Ganin, Yaroslav and Lempitsky, Victor},
  booktitle = 	 {Proceedings of the 32nd International Conference on Machine Learning},
  pages = 	 {1180--1189},
  year = 	 {2015},
  editor = 	 {Bach, Francis and Blei, David},
  volume = 	 {37},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Lille, France},
  month = 	 {07--09 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v37/ganin15.pdf},
  url = 	 {https://proceedings.mlr.press/v37/ganin15.html},
  abstract = 	 {Top-performing deep architectures are trained on massive amounts of labeled data. In the absence of labeled data for a certain task, domain adaptation often provides an attractive option given that labeled data of similar nature but from a different domain (e.g. synthetic images) are available. Here, we propose a new approach to domain adaptation in deep architectures that can be trained on large amount of labeled data from the source domain and large amount of unlabeled data from the target domain (no labeled target-domain data is necessary). As the training progresses, the approach promotes the emergence of "deep" features that are (i) discriminative for the main learning task on the source domain and (ii) invariant with respect to the shift between the domains. We show that this adaptation behaviour can be achieved in almost any feed-forward model by augmenting it with few standard layers and a simple new gradient reversal layer. The resulting augmented architecture can be trained using standard backpropagation. Overall, the approach can be implemented with little effort using any of the deep-learning packages. The method performs very well in a series of image classification experiments, achieving adaptation effect in the presence of big domain shifts and outperforming previous state-of-the-art on Office datasets.}
}

%DNN
@article{dann,
  title={Domain-adversarial training of neural networks},
  author={Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and Larochelle, Hugo and Laviolette, Fran{\c{c}}ois and Marchand, Mario and Lempitsky, Victor},
  journal={The journal of machine learning research},
  volume={17},
  number={1},
  pages={2096--2030},
  year={2016},
  publisher={JMLR. org}
}

@inproceedings{ae,
author = {Ballard, Dana H.},
title = {Modular Learning in Neural Networks},
year = {1987},
isbn = {0934613427},
publisher = {AAAI Press},
abstract = {In the development of large-scale knowledge networks much recent progress has been
inspired by connections to neurobiology. An important component of any "neural" network
is an accompanying learning algorithm. Such an algorithm, to be biologically plausible,
must work for very large numbers of units. Studies of large-scale systems have so
far been restricted to systems Without internal units (units With no direct connections
to the input or output). Internal units are crucial to such systems as they are the
means by which a system can encode high-order regularities (or invariants) that are
Implicit in its inputs and outputs. Computer simulations of learning using internal
units have been restricted to small-scale systems. This paper describes away of coupling
autoassociative learning modules Into hierarchies that should greatly improve the
performance of learning algorithms in large-scale systems. The Idea has been tested
experimentally with positive results.},
booktitle = {Proceedings of the Sixth National Conference on Artificial Intelligence - Volume 1},
pages = {279–284},
numpages = {6},
location = {Seattle, Washington},
series = {AAAI'87}
}

@inproceedings{vae,
  author    = {Diederik P. Kingma and
               Max Welling},
  editor    = {Yoshua Bengio and
               Yann LeCun},
  title     = {Auto-Encoding Variational Bayes},
  booktitle = {2nd International Conference on Learning Representations, {ICLR} 2014,
               Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings},
  year      = {2014},
  url       = {http://arxiv.org/abs/1312.6114},
  timestamp = {Thu, 04 Apr 2019 13:20:07 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/KingmaW13.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

%TL
@misc{zhang2021quantifying,
      title={Quantifying and Improving Transferability in Domain Generalization}, 
      author={Guojun Zhang and Han Zhao and Yaoliang Yu and Pascal Poupart},
      year={2021},
      eprint={2106.03632},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
@ARTICLE{pan2010survey,  author={Pan, Sinno Jialin and Yang, Qiang},  journal={IEEE Transactions on Knowledge and Data Engineering},   title={A Survey on Transfer Learning},   year={2010},  volume={22},  number={10},  pages={1345-1359},  doi={10.1109/TKDE.2009.191}}

@InProceedings{Blitzer2008DA,
  author    = {Blitzer, John and Crammer, Koby and Kulesza, Alex and Pereira, Fernando and Wortman, Jennifer},
  booktitle = {Advances in Neural Information Processing Systems},
  title     = {Learning Bounds for Domain Adaptation},
  year      = {2008},
  editor    = {J. Platt and D. Koller and Y. Singer and S. Roweis},
  publisher = {Curran Associates, Inc.},
  volume    = {20},
  url       = {https://proceedings.neurips.cc/paper/2007/file/42e77b63637ab381e8be5f8318cc28a2-Paper.pdf},
}

@InProceedings{BenDavid2007RepDA,
  author    = {Ben-David, Shai and Blitzer, John and Crammer, Koby and Pereira, Fernando},
  booktitle = {Advances in Neural Information Processing Systems},
  title     = {Analysis of Representations for Domain Adaptation},
  year      = {2007},
  editor    = {B. Sch\"{o}lkopf and J. Platt and T. Hoffman},
  publisher = {MIT Press},
  volume    = {19},
  url       = {https://proceedings.neurips.cc/paper/2006/file/b1b0432ceafb0ce714426e9114852ac7-Paper.pdf},
}

@Article{Bastani2021ProxiesTransferLearning,
  author   = {Bastani, Hamsa},
  journal  = {Management Science},
  title    = {Predicting with Proxies: Transfer Learning in High Dimension},
  year     = {2021},
  number   = {5},
  pages    = {2964-2984},
  volume   = {67},
  abstract = {Predictive analytics is increasingly used to guide decision making in many applications. However, in practice, we often have limited data on the true predictive task of interest and must instead rely on more abundant data on a closely related proxy predictive task. For example, e-commerce platforms use abundant customer click data (proxy) to make product recommendations rather than the relatively sparse customer purchase data (true outcome of interest); alternatively, hospitals often rely on medical risk scores trained on a different patient population (proxy) rather than their own patient population (true cohort of interest) to assign interventions. Yet, not accounting for the bias in the proxy can lead to suboptimal decisions. Using real data sets, we find that this bias can often be captured by a sparse function of the features. Thus, we propose a novel two-step estimator that uses techniques from high-dimensional statistics to efficiently combine a large amount of proxy data and a small amount of true data. We prove upper bounds on the error of our proposed estimator and lower bounds on several heuristics used by data scientists; in particular, our proposed estimator can achieve the same accuracy with exponentially less true data (in the number of features d). Finally, we demonstrate the effectiveness of our approach on e-commerce and healthcare data sets; in both cases, we achieve significantly better predictive accuracy as well as managerial insights into the nature of the bias in the proxy data.This paper was accepted by George Shanthikumar, big data and analytics.},
  doi      = {10.1287/mnsc.2020.3729},
  eprint   = {https://doi.org/10.1287/mnsc.2020.3729},
  url      = {https://doi.org/10.1287/mnsc.2020.3729},
}

@article{Bastani2019MetaDP,
  author    = {Hamsa Bastani and
               David Simchi{-}Levi and
               Ruihao Zhu},
  title     = {Meta Dynamic Pricing: Learning Across Experiments},
  journal   = {CoRR},
  volume    = {abs/1902.10918},
  year      = {2019},
  url       = {http://arxiv.org/abs/1902.10918},
  eprinttype = {arXiv},
  eprint    = {1902.10918},
  timestamp = {Tue, 21 May 2019 18:03:36 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1902-10918.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

%missing data
@article{muthen1987structural,
  title={On structural equation modeling with data that are not missing completely at random},
  author={Muth{\'e}n, Bengt and Kaplan, David and Hollis, Michael},
  journal={Psychometrika},
  volume={52},
  number={3},
  pages={431--462},
  year={1987},
  publisher={Springer}
}

@article{takahashi2017misingdataMCMC,
author={Takahashi, M.}, 
year={2017},
title={Statistical Inference in Missing Data by MCMC and Non-MCMC Multiple Imputation Algorithms: Assessing the Effects of Between-Imputation Iterations},
 journal={Data Science Journal}, volume={16}, 
 page={37}}
 
 @misc{yoon2018gain,
      title={GAIN: Missing Data Imputation using Generative Adversarial Nets}, 
      author={Jinsung Yoon and James Jordon and Mihaela van der Schaar},
      year={2018},
      eprint={1806.02920},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

%predictive
@InProceedings{Kao2009Regression,
  author    = {Kao, Yi-hao and Roy, Benjamin and Yan, Xiang},
  booktitle = {Advances in Neural Information Processing Systems},
  title     = {Directed Regression},
  year      = {2009},
  editor    = {Y. Bengio and D. Schuurmans and J. Lafferty and C. Williams and A. Culotta},
  publisher = {Curran Associates, Inc.},
  volume    = {22},
  url       = {https://proceedings.neurips.cc/paper/2009/file/0c74b7f78409a4022a2c4c5a5ca3ee19-Paper.pdf},
}

@InProceedings{Taskar2005Prediction,
  author    = {Taskar, Ben and Chatalbashev, Vassil and Koller, Daphne and Guestrin, Carlos},
  booktitle = {Proceedings of the 22nd International Conference on Machine Learning},
  title     = {Learning Structured Prediction Models: A Large Margin Approach},
  year      = {2005},
  address   = {New York, NY, USA},
  pages     = {896–903},
  publisher = {Association for Computing Machinery},
  series    = {ICML '05},
  abstract  = {We consider large margin estimation in a broad range of prediction models where inference involves solving combinatorial optimization problems, for example, weighted graph-cuts or matchings. Our goal is to learn parameters such that inference using the model reproduces correct answers on the training data. Our method relies on the expressive power of convex optimization problems to compactly capture inference or solution optimality in structured prediction models. Directly embedding this structure within the learning formulation produces concise convex problems for efficient estimation of very complex and diverse models. We describe experimental results on a matching task, disulfide connectivity prediction, showing significant improvements over state-of-the-art methods.},
  doi       = {10.1145/1102351.1102464},
  isbn      = {1595931805},
  location  = {Bonn, Germany},
  numpages  = {8},
  url       = {https://doi.org/10.1145/1102351.1102464},
}

@InProceedings{Osokin2017Prediction,
  author    = {Osokin, Anton and Bach, Francis and Lacoste-Julien, Simon},
  booktitle = {Advances in Neural Information Processing Systems},
  title     = {On Structured Prediction Theory with Calibrated Convex Surrogate Losses},
  year      = {2017},
  editor    = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
  publisher = {Curran Associates, Inc.},
  volume    = {30},
  url       = {https://proceedings.neurips.cc/paper/2017/file/38db3aed920cf82ab059bfccbd02be6a-Paper.pdf},
}

%prescriptive analysis/optimization

@Article{Bertsimas2020Prescriptive,
  author   = {Bertsimas, Dimitris and Kallus, Nathan},
  journal  = {Management Science},
  title    = {From Predictive to Prescriptive Analytics},
  year     = {2020},
  number   = {3},
  pages    = {1025-1044},
  volume   = {66},
  abstract = {We combine ideas from machine learning (ML) and operations research and management science (OR/MS) in developing a framework, along with specific methods, for using data to prescribe optimal decisions in OR/MS problems. In a departure from other work on data-driven optimization, we consider data consisting, not only of observations of quantities with direct effect on costs/revenues, such as demand or returns, but also predominantly of observations of associated auxiliary quantities. The main problem of interest is a conditional stochastic optimization problem, given imperfect observations, where the joint probability distributions that specify the problem are unknown. We demonstrate how our proposed methods are generally applicable to a wide range of decision problems and prove that they are computationally tractable and asymptotically optimal under mild conditions, even when data are not independent and identically distributed and for censored observations. We extend these to the case in which some decision variables, such as price, may affect uncertainty and their causal effects are unknown. We develop the coefficient of prescriptiveness P to measure the prescriptive content of data and the efficacy of a policy from an operations perspective. We demonstrate our approach in an inventory management problem faced by the distribution arm of a large media company, shipping 1 billion units yearly. We leverage both internal data and public data harvested from IMDb, Rotten Tomatoes, and Google to prescribe operational decisions that outperform baseline measures. Specifically, the data we collect, leveraged by our methods, account for an 88\% improvement as measured by our coefficient of prescriptiveness.This paper was accepted by Noah Gans, optimization.},
  doi      = {10.1287/mnsc.2018.3253},
  eprint   = {https://doi.org/10.1287/mnsc.2018.3253},
  url      = {https://doi.org/10.1287/mnsc.2018.3253},
}

@misc{bertsimas2017power,
      title={The Power and Limits of Predictive Approaches to Observational-Data-Driven Optimization}, 
      author={Dimitris Bertsimas and Nathan Kallus},
      year={2017},
      eprint={1605.02347},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}

@Article{Elmachtoub2021SPO,
  author   = {Elmachtoub, Adam N. and Grigas, Paul},
  journal  = {Management Science},
  title    = {Smart “Predict, then Optimize”},
  year     = {2021},
  number   = {0},
  pages    = {null},
  volume   = {0},
  abstract = {Many real-world analytics problems involve two significant challenges: prediction and optimization. Because of the typically complex nature of each challenge, the standard paradigm is predict-then-optimize. By and large, machine learning tools are intended to minimize prediction error and do not account for how the predictions will be used in the downstream optimization problem. In contrast, we propose a new and very general framework, called Smart “Predict, then Optimize” (SPO), which directly leverages the optimization problem structure—that is, its objective and constraints—for designing better prediction models. A key component of our framework is the SPO loss function, which measures the decision error induced by a prediction. Training a prediction model with respect to the SPO loss is computationally challenging, and, thus, we derive, using duality theory, a convex surrogate loss function, which we call the SPO+ loss. Most importantly, we prove that the SPO+ loss is statistically consistent with respect to the SPO loss under mild conditions. Our SPO+ loss function can tractably handle any polyhedral, convex, or even mixed-integer optimization problem with a linear objective. Numerical experiments on shortest-path and portfolio-optimization problems show that the SPO framework can lead to significant improvement under the predict-then-optimize paradigm, in particular, when the prediction model being trained is misspecified. We find that linear models trained using SPO+ loss tend to dominate random-forest algorithms, even when the ground truth is highly nonlinear.This paper was accepted by Yinyu Ye, optimization.},
  doi      = {10.1287/mnsc.2020.3922},
  eprint   = {https://doi.org/10.1287/mnsc.2020.3922},
  url      = {https://doi.org/10.1287/mnsc.2020.3922},
}

@Article{Ferreira2016PO,
  author   = {Ferreira, Kris Johnson and Lee, Bin Hong Alex and Simchi-Levi, David},
  journal  = {Manufacturing \& Service Operations Management},
  title    = {Analytics for an Online Retailer: Demand Forecasting and Price Optimization},
  year     = {2016},
  number   = {1},
  pages    = {69-88},
  volume   = {18},
  abstract = {We present our work with an online retailer, Rue La La, as an example of how a retailer can use its wealth of data to optimize pricing decisions on a daily basis. Rue La La is in the online fashion sample sales industry, where they offer extremely limited-time discounts on designer apparel and accessories. One of the retailer’s main challenges is pricing and predicting demand for products that it has never sold before, which account for the majority of sales and revenue. To tackle this challenge, we use machine learning techniques to estimate historical lost sales and predict future demand of new products. The nonparametric structure of our demand prediction model, along with the dependence of a product’s demand on the price of competing products, pose new challenges on translating the demand forecasts into a pricing policy. We develop an algorithm to efficiently solve the subsequent multiproduct price optimization that incorporates reference price effects, and we create and implement this algorithm into a pricing decision support tool for Rue La La’s daily use. We conduct a field experiment and find that sales does not decrease because of implementing tool recommended price increases for medium and high price point products. Finally, we estimate an increase in revenue of the test group by approximately 9.7\% with an associated 90\% confidence interval of [2.3\%, 17.8\%].},
  doi      = {10.1287/msom.2015.0561},
  eprint   = {https://doi.org/10.1287/msom.2015.0561},
  url      = {https://doi.org/10.1287/msom.2015.0561},
}

@Book{Sen2017LearningOptimization,
  author    = {Sen, Suvrajeet and Deng, Yunxiao},
  publisher = {Humboldt-Universität zu Berlin},
  title     = {Learning Enabled Optimization: Towards a Fusion of Statistical Learning and Stochastic Optimization},
  year      = {2017},
  doi       = {http://dx.doi.org/10.18452/18087},
}


%SGD
@misc{bertsekas2017cvx,
      title={Incremental Gradient, Subgradient, and Proximal Methods for Convex Optimization: A Survey}, 
      author={Dimitri P. Bertsekas},
      year={2017},
      eprint={1507.01030},
      archivePrefix={arXiv},
      primaryClass={cs.SY}
}

@article{Bottou2018optML,
title = "Optimization methods for large-scale machine learning",
abstract = "This paper provides a review and commentary on the past, present, and future of numerical optimization algorithms in the context of machine learning applications. Through case studies on text classification and the training of deep neural networks, we discuss how optimization problems arise in machine learning and what makes them challenging. A major theme of our study is that large-scale machine learning represents a distinctive setting in which the stochastic gradient (SG) method has traditionally played a central role while conventional gradient-based nonlinear optimization techniques typically falter. Based on this viewpoint, we present a comprehensive theory of a straightforward, yet versatile SG algorithm, discuss its practical behavior, and highlight opportunities for designing algorithms with improved performance. This leads to a discussion about the next generation of optimization methods for large-scale machine learning, including an investigation of two main streams of research on techniques that diminish noise in the stochastic directions and methods that make use of second-order derivative approximations.",
keywords = "Algorithm complexity analysis, Machine learning, Noise reduction methods, Numerical optimization, Second-order methods, Stochastic gradient methods",
author = "L{\'e}on Bottou and Curtis, {Frank E.} and Jorge Nocedal",
note = "Funding Information: ∗Received by the editors June 16, 2016; accepted for publication (in revised form) April 19, 2017; published electronically May 8, 2018. http://www.siam.org/journals/sirev/60-2/M108017.html Funding: The work of the second author was supported by U.S. Department of Energy grant DE-SC0010615 and U.S. National Science Foundation grant DMS-1016291. The work of the third author was supported by Office of Naval Research grant N00014-14-1-0313 P00003 and Department of Energy grant DE-FG02-87ER25047s. †Facebook AI Research, New York, NY 10003 (leon@bottou.org). ‡Department of Industrial and Systems Engineering, Lehigh University, Bethlehem, PA 18015 (frank.e.curtis@gmail.com). §Department of Industrial Engineering and Management Sciences, Northwestern University, Evanston, IL 60201 (j-nocedal@northwestern.edu). Publisher Copyright: {\textcopyright} 2018 Society for Industrial and Applied Mathematics.",
year = "2018",
doi = "10.1137/16M1080173",
language = "English (US)",
volume = "60",
pages = "223--311",
journal = "SIAM Review",
issn = "0036-1445",
publisher = "Society for Industrial and Applied Mathematics Publications",
number = "2",
}
