# Learning Centrality By Learning To Route
The code base for the paper "Learning Centrality by Learning to Route." International Conference on Complex Networks and Their Applications. Springer, Cham, 2021.


## Prerequisites
The code was implemented in python 3.7 with anaconda environment. 
All requirements are included in the requirements.txt file. 


## Components
### RBC
 Computing Routing Betweenness Centrality (RBC) of graph.
### LRC
#### LRCNN
 The neural-network with the forward flow logic. 
#### ModelHandler
  Hnadling the training of the model.
#### ModelTester
  The main flow of LRC, in charge of intialization, training and computing the correlation scores. 
### Utils
#### CommonStr
String Constant
#### Optimizer
Wrapper for optimizer initialization
#### ParamsManager
  Managing model parameters



#### Citation
In case of mentionaning the paper please use the following bibtex citation:
```
@inproceedings{bachar2021learning,
  title={Learning Centrality by Learning to Route},
  author={Bachar, Liav and Elyashar, Aviad and Puzis, Rami},
  booktitle={International Conference on Complex Networks and Their Applications},
  pages={247--259},
  year={2021},
  organization={Springer}
}
```
