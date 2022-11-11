# Can one hear the position of nodes? 

Wave propagation through nodes and links of a network forms
the basis of spectral graph theory. Nevertheless, the sound emitted by
nodes within the resonating chamber formed by a network are not well
studied. The sound emitted by vibrations of individual nodes reflects the
structure of the overall network topology but also the location of the node
within the network. In this article a sound recognition neural network
is trained to infer centrality measures from the nodes’ wave-forms. In
addition to advancing network representation learning, sounds emitted by
nodes are plausible in most cases. Auralization of the network topology
may open new directions in arts, competing with network visualization.

[Article](https://github.com/puzis/centrality-learning/raw/main/centrality_from_auralized_nodes/Can_one_hear_the_position_of_nodes.pdf)
 | 
[Slides](https://github.com/puzis/centrality-learning/raw/main/centrality_from_auralized_nodes/Can_one_hear_the_position_of_nodes_slides.pdf)

```
@inproceedings{puzis2021can,
  title={Can one hear the position of nodes?},
  author={Puzis, Rami},
  booktitle={International Conference on Complex Networks and Their Applications},
  pages={to appear},
  year={2022},
}
```

This folder contains a collection of Jupiter notebooks for experimenting with network aurlization, training centrlaity inference from auralized nodes, and evaluating saved model checkpoints on random graphs. Notebooks containing a centrality (Betweenness, CLoseness, Degree, Eigenvecor) in their names are used for training. Playground is a playground as its name suggests. Checkpoints currently stored here are those that were used for producing results presented at the [Complex Networks conference](https://complexnetworks.org/) 2022 edition. 

Contributions are appreciated. Especially extracting classes and auxuliary functions from the notebooks into a Python module and building a pip library. 
It would also be nice to have more sound classificiation models here :) 

Enjoy. 
