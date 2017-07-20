# neural-semantic-role-labeler
## Semantic Role Labeler with Recurrent Neural Networks

This repo contains Theano implementations of the models described in the following paper:
- [End-to-end Learning of Semantic Role Labeling Using Recurrent Neural Networks](http://www.aclweb.org/anthology/P15-1109), ACL 2015

The model gives about F1 = 78.00 on the dev & test set in the following hyperparameter setting:
- word embedding=SENNA, unit=GRU, hidden dimension=128, L2 regularization=0.0005, layers=4

##### Data
- CoNLL-2005 Shared Task (http://www.cs.upc.edu/~srlconll/)
- Word Embedding: SENNA (http://ronan.collobert.com/senna/)

##### Example Comand
`python -m srl.cons_srl.main -mode train --train_data path/to/data/conll2005.train.txt --dev_data path/to/data/conll2005.dev.txt --test_data path/to/data/conll2005.test.txt --init_emb path/to/data/senna.txt --unit gru --layer 1 --hidden 32 --reg 0.0001`

##### CoNLL04 format
```
An                   DT    B-NP         (S*             -                   (A0*            (A0*       
A.P.                 NNP   I-NP           *             -                      *               *       
Green                NNP   I-NP           *             -                      *               *       
official             NN    I-NP           *             -                      *A0)            *A0)    
declined             VBD   B-VP           *             decline              (V*V)             *       
to                   TO    I-VP           *             -                   (A1*               *       
comment              VB    I-VP           *             comment                *             (V*V)     
on                   IN    B-PP           *             -                      *               *       
the                  DT    B-NP           *             -                      *            (A1*       
filing               NN    I-NP           *             -                      *A1)            *A1)    
.                    .     O              *S)           -                      *               *       
```

##### CoNLL05 format
```
Ms.                 NNP   (S1(S(NP*          *          -                    (A0*      
Haag                NNP           *)     (LOC*)         -                       *)     
plays               VBZ        (VP*          *          play                  (V*)     
Elianti             NNP        (NP*))        *          -                    (A1*)     
.                   .             *))        *          -                       *      
```
