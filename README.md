# Integartive Network Modeling

The broad goal of inm is to contribute to the characterization of biological networks by integrating data from diverse sources.
inm is under development and will be distributed as an Integrative Modeling Platform ([IMP](https://github.com/salilab/imp/)) module.

Modeling proceeds in 5 stages: (i) gathering AP-MS protein spectral counts under multiple conditions, (ii) representing a model where nodes represent protein types
and edges represent a protein interaction between at least one pair of molecules of the corresponding types, (iii) a scoring function
that scores the agreement between AP-MS spectral counts and a network model, (iv) sampling alternative configurations of edges using an MCMC scheme,
(v) analysis of the output sample of network models.

inm includes an example system and benchmarks against Protein Data Bank and Humap 2.0 derived interactions.  
