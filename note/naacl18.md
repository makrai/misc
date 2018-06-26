# Widening NLP workshop

* Salle+  Incorporating sub-word information into matrix factorization word

# Tutorial: Lexical Resources

* definition modeling
* dictionary example acquisition

# Session 2

* T Nguyen & D Chiang: Lex Choice in NMT 
  * bite per encoding 
  * morphologically rich languages
  * fix norm (+lex)
* Gu, Hassan, Devlin, & Li 39 
 * Universal Lexical representation
* Gururangan, Swayamdippta, Levy, Schwartz, Bowman & Smith: Annot artifacts in
  inference data (photo)

# Session 3

* Chang, Wang, Vilnis, & McCallum: Distributional inclusion ... embedding for
  unsup hypernymy detection
  * new SOTA
* Vulić, Glavaš, Mrkšić, & Korhonen: 
  Post-specialization: Retrofitting ... unseen in lex res

# Session 4

* Upadhyay, Vyas, Carpuat, Roth: Robust cross-ling hypernymy detection using
  dependency context
* Xie, Genthial, Xie, Ng, & Jurafsky: Noising and Denoising nat lang:
  diverse back-translation for grammar correction
* Grundkiewicz & Junczys-Dowmunt: Grammatical Error Correction with hybrid MT
  * Grundkiewicz is an author of the SOTA GEC sys
  * eval metrics, Inherent Biases in Reference-based (Choshen & Abend ACL 2018)
  * "following our own recommendation"

# Session 6

* Antoniak & Mimno: stability of embedding-based similarities
  * order of sentences in the corpus 
* Jiang, Yu, Hsieh, & Chang: positive-unlabeled (PU) learning
  * \rho: confidence threshold for true zero vs missing value
* Nguyen, Nguyen, & Nguyen: novel embed for KB comp based on CNN (photo)
* Wang, Shen, & Jin: ...semantic frame... (photo)

#  Session 7

* Damonta & Cohen: Cross-lingual AMR parsing
  * parszolt kinai mondatból angol AMR
* Shaw, Uszkoreit, & Vaswani: ...relative position (photo)

# Session 8 Test-of-Time Papers (2002)

* Pang+: Thumbs up? Sentiment
  * equal number of positive and negative examples -- is that a good idea?

# Session 9

* Cai and Wang: KBGAN: Adversarial Learning for Knowledge Graph Embeddings
  * negative pairs should be sensible

# Session 10

* Wenhu Chen, Xiong, Yan, and Wang: Variational Knowledge Graph Reasoning
* Li, Robin Jia, He, & Liang: sentiment and style transfer...
  * AE
  * adversarial setting: make the attribute indistinguishable from the
    bottle-neck representation
* Iyyer, Wieting, Gimpel, and Zettlemoyer:  Adversarial Example Generation with
  Syntactically Controlled Paraphrase Networks

# Session 11

* Beuchel & Hahn: Word emotion induction for multiple languages 
  as a deep multi-task learning problem
  * valence (kellemes), arousel (calm--excite), dominance
  * bigger emo lexicon in LREC paper
  * new SOTA
* Fernández-González and Gómez-Rodríguez: Non-proj dep pars w non-local tarns
  * new SOTA
* Sanchez, Mitchell, & and Riedel: Anal of NLI models: ...
  three factors of robustness
  * insensitivity, polarity, and unseen pairs
  * e.g. sunset-sunrise <- polar pairs => contradiction
* Wendlandt, Kummerfeld, and Mihalcea: Factors... instability of [embeddings]
  * ...
  * POS
  * GloVe > w2v
  
# Session 12: Outstanding papers

* Peters, Neumann, Iyyer, Gardner, Clark, Lee, Zettlemoyer
  * new SOTA in 6 tasks
  * bi- deep RNN
* Clark, Ji, & Smith: Neural text gen in stories using entity repr as context
  * Hobbs (1979)
* Chen, Gilroy, Maletti, May, & Knight
  * RNN vs WFSA
  * `P(\Sigma*) ?= 1`
  * consistency, never-ending derivations
  * decidable? no
    * trained RNNs -- conjecture
  * best path: Dijkstra vs Knuth vs no
  * best string
  * equivalence
  * minimization
  * class of RNN languages ?= LSTM

#  `*`SEM

* Bakarov, Suvorov, Sochenkov: The Limitations of Cross-lang word embed eval
  * bib (photo)
* Kallmeyer, QasemiZadeh, Jackie Chi Kit Cheung
  * {PCFG optim & verb clust} simult
* Pierrejean & Tanguy: Predicting word embeddings variability
  * some words' NNs are more stable, e.g. family, co-hyponyms
* Tu Vu & V Shwartz: Integrating multiplicative features unto supervised distri
  methods of lex entail
  * point-wise product motivated by cos sim
* Poliak, Naradowsky, Haldar, Rudinger and Van Durme: 
  Hypothesis only baselines (best paper)
  * jókedélyű előadó
  * sem proto-roles --> Reisinger, bib 
* Allen, Choh Man Teng: Putting sem into sem roles
* Mahabal, Roth, Mittal: Polysemy via Sparse

# SemEval

* Pavlick: Compositionality
  * deep learning is bad at the ling level (compositionality)
  * RepEval: general purpose
  * _Bush travels on Monday to Michigan to remark on (# Japanese) economy._

## 7 Sem Rel Extract and Classif Scientific

* Rotszejn, Hollenstein, Ce Zhang: relative position embedding
* reverse resemble non-reverse

## 8 Hypernym

* Crim
  * discover co-hyponyms
  * e.g.
    * aquamarine: crystal, color
    * vegetarian: dish

# Generalization in Deep

* Ndapa: word translation
  * are words (Artetxe 18)
  * reduce supervision (Nakashole emnlp 17)
  * linear --> locally linear,
  * bib Connean+ 18
  * neighborhood: semantic field
* Mitchell, Stenetorm, Minervini, & Riedel: Extrapolation in NLP
  * theoretical
    * symmetry
    * linearity

# SCLeM: Sub-word and Char Level

* Neubig: Morphology, when is is useful for neural models?
  * transfer from Turkish & Hindi
  * different writing system, phonemes help
  * fi/hu, da/sv, ru/bg, es/pt
* Salle & Villavicencio: sub-word info into mx factor embeds (photo)
