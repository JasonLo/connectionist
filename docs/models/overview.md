# Connectionist.models

There are 2 models in this module:

- Orthography-to-Phonology model ([PMSP][connectionist.models.PMSP]) by [Plaut, McClelland, Seidenberg and Patterson (1996)](https://www.cnbc.cmu.edu/~plaut/papers/abstracts/PlautETAL96PsyRev.wordReading.html), simulation 3.
- [Hub-and-spokes][connectionist.models.HubAndSpokes] model by [Rogers et. al. (2004)](https://doi.org/10.1037/0033-295X.111.1.205).

## Brain damaging APIs

*Currently only available in [PMSP][connectionist.models.PMSP]*

- Reduce number of units in a layer
- Set some portion of weight in a connection to zero (and make it un-trainable)
- Remove the entire connection between two layers
- Introduce noise to a layer's input
- Put stress to keep the weights small (L2 regularization)
