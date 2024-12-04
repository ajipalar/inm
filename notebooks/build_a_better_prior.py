"""
A structure informed prior for network models should be built in the following way.

1. Obtain a same of structures from the PDB
2. Filter structures to remove homomers
3. Calculate (or look up) pairwise BSASA for every PDB file
4. Write out an edgelist per complex

Given the per chain BSASA, write the networks of maximal BSASA or the sum of BSASA?

Obtain per node networks
"""
