# Conducting Local Dimensionality Reduction While Maintaining Global View Consistency

## Introduction
### WHY LOCAL?
Interactive demand: sometimes users may want to focus on the local area and explore the details of a DR layout.
### WHY RE-COMPUTE?
Reducing the data scale to a local scope can exclude the influence from other (irrelevant) data points in the global scope, 
which lead to better representation of the local data points.
Thus, we look forward to find new patterns that fail to appear in the global scope through local re-computation.
### Problems for Local Dimensionality Reduction
The layout of simply re-computed local DR result usually ends up greatly different with the original layout of global result, 
which would cause inconsistency in visualization, which further leads to inconvenience for comparison and analysis.

## Basic Algorithms
### K Nearest Neighbor (KNN)
### PCA, MDS, LLE, Isomap, t-SNE
### Stress Majorization 

## LDR Algorithm (PCA Version)
