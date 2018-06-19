#A konferencia előtti workshopok

* Salle+ (Wildening NLP workshop) Incorporating subword information into matrix factorization word

##Tutorial: Lexical Resources

* definition modeling
* dictionary example acquisitoin

#Session 2

* bite per encoding

##Gu, Hassan, Devlin, & Li 39

Universal Lexical representatin:wq

#Session 6

* Antoniak & Mimno: stability of embedding-based similarities

* Jiang, Yu, Hsieh, & Chang: positive-unlabeled learning

# Session 7

* Damonta & Cohen: Cross-lingual AMR parsing
  * parszolt kinai mondatból angol AMR

#Session 8 Test-of-Time Papers (2002)

* Pang+: Thumbs up? Sentiment
  * equal number of positive and negative examples -- is that a good idea?

#Session 9

* Cai and Wang: KBGAN: Adversarial Learning for Knowledge Graph Embeddings
  * negative paris should be sensible

#Session 10

* Wenhu Chen, Xiong, Yan, and Wang: Variational Knowledge Graph Reasoning
* Li, Robin Jia, He, & Liang: sentiment and style transfer...
  * AE
  * adversarial setting: make the attribute indistinguishable from the
    bottle-neck representation
* Iyyer, Wieting, Gimpel, and Zettlemoyer:  Adversarial Example Generation with
  Syntactically Controlled Paraphrase Networks

#Session 11

* Beuchel & Hahn: Word emotion induction for multiple languages 
  as a deep multi-task learnin problem
  * valence (kellemes), arousel (calm--excite), dominance
  * bigger emo lexicon in LREC paper
  * new SOTA
* Sanchez, Mitchell, & and Riedel: Anal of NLI models: ...
  three fractors of robustness
  * insensitivity, polarity, and inseen pairs
  * e.g. sunset-sunrise <- polar paris => contradiction
* Wendlandt, Kummerfeld, and Mihalcea: Factors... instability of [embeddings]
  * ...
  * POS
  * GloVe > w2v
  
#Session 12: Outstanding papers

* Peters, Neumann, Iyyer, Gardner, Clark, Lee, Zettlemoyer
  * new SOTA in 6 tasks
  * bi- deep RNN
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

# `*`SEM

* Durme: Hypothesis only baselines (best paper)
  * jókedélyű előadó
  * sem proto-roles

#SemEval

* Pavlick: Compositionality
  * deep learning is bad at the ling level (compositionality)
  * RepEval: general purpose
  * _Bush travels on Monday to Michigan to remark on (#Japanese) economy._

##8 Hypernym

* Crim
  * discover co-hyponyms
  * e.g.
    * aquamarine: crystal, color
    * vegetarian: dish

#Generalization in Deep

* Ndapa: word transslation
    * are words (Artetxe 18)
    * reduce supervision (Nakashole emnlp 17)
    * linear --> locally linear,
    * bib Connean+ 18
    * neighborhood: semantic field

#SCLeM: Subword and Char LEvel

Neubig: Morphology, when is is useful for neural models?
  * transfer from Turkish & Hindi
  * differend writing system, phonemes help
  * fi/hu, da/sv, ru/bg, es/pt
