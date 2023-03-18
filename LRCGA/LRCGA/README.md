# Learning a routing centrality based on graph auralization
The code base for method " learning a routing centrality based on graph auralization"(LRCGA) in the paper "Echolocation in networks: learning to route based on the sound of nodes"

This code is based on: learned_routing_centrality (code by Liav Bachar can be found here: https://github.com/liavbach/LRC )

## Prerequisites
The code was implemented in python 3.9 with anaconda environment. 
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
### ModelTester.py
Use to train the model
### auralization.py
Use to generate graph auralization embedding
### generating_data.py
Use to generate training and test data-set

## Running
Before running, please modify the parameter you want in the tail of file "ModelTester.py"

```bash
cd LCGA
python ModelTester.py
```
After you run this code, the result will show in "log.txt".  Models in every step will store in "\saved_model" folder.

