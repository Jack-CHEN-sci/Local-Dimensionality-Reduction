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
K-Nearest Neighbors, or KNN for short, is one of the simplest machine learning algorithms and is used in a wide array of institutions. 
KNN is a non-parametric, lazy learning algorithm.
#### Pros:
+ No assumptions about data
+ Simple algorithm — easy to understand
+ Can be used for classification and regression
#### Cons:
+ High memory requirement — All of the training data must be present in memory in order to calculate the closest K neighbors
+ Sensitive to irrelevant features
+ Sensitive to the scale of the data since we’re computing the distance to the closest K points
#### Algorithm:
```
Preprocessing:
  1. Pick a value for K (i.e. 6);
  2. Take the K nearest neighbors of the new data point according to their Euclidean distance;
Prediction:
  Among these neighbors, count the number of data points in each category 
  and assign the new data point to the category where you counted the most neighbors;
```
### PCA, MDS, LLE, Isomap, t-SNE
### Stress Majorization 

## LDR Algorithm (PCA Version)
