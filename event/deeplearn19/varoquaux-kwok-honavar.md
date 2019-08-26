# Gaël Varoquaux (INRIA) Representation Learning in Limited Data Settings

##1 Representations for machine learning 5

* Non-asymptotic supervised learning 6 
  * Három tagra bontja a felügyelt tanulásban várható hibát: 
  a modellezendő jelenség zajából adódó tag, a tanulóadatban levő zajból adódó,
  és a hipotézistér szűkösségéből adódó (vagyis hogy a valódi függvény nincs a
  hipotézisek között).  
* Learning with representations 18
* Supervised learning of representations 26

##2 Matrix factorization and its variants 35

* For signals 36
  * PCA 35
  * ICA 36
  * Dictionary learning 38
* For discrete objects: Gamma-Poisson for factorizing counts 51
  * Application: sub-string representation 53
  * Natural language processing: topic-modeling history 56
  * metric learning 62

##3 Fisher kernels 66

* Kernels feature maps 67
* From likelihoods to Kernels 71
* Fisher Kernel applications 77

(References 80)


# James Kwok (Hong Kong Uni) Compressing Neural Networks

Az előadó nagyon gyorsan (mint egy gyorsított felvétel) de nagyon érthetően
(nemcsak kínai mércén) beszél. Még a kötelező bevezető szlájdok sem hatottak
unalmasnak az előadásában.  

A fő fejezetek : 
Hálózatok ritkítása metszéssel (pruning)  és ritkasági regularizálókkal,
Hálózatok kvantálása kevesebb bittel, Alacsony rangú közelítés, 
Lepárlás (distillation) és további kompakt modellek, és 
A neurális architectúra keresése.  

##Network Sparsification

* unstructured pruning
* structured pruning
* topology of pruned networks
  * Networks and Power Law 50
    * truncated power-law distribution (TPL) [Kolyukhin and Torabi, 2013]
      * [x min , x max ]: range over which the power law is valid
    * Preferential Attachment TODO 56

##Quantization

##Knowledge Distillation


# Vasant Honavar (Pennsylvania S Uni) Causal Models for Making Sense of Data
