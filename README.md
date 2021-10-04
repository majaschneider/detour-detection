## Overview

This repository implements a privacy metric that measures privacy risks in trajectory data by analyzing detours.

The metric will be implemented as a Flask app that receives requests and data from a Springboot backend.

## Concept Overview and Background

The metric tries to measure the degree of privacy in trajectory data **without clusters of locations at hand**, such as stay points or stay regions. We assume these have been removed beforehand by an algorithm, for example _Promesse_.

Consequently, this metric is able to measure the remaining trajectories regarding their _difference from the shortest path considering the route given_. The idea is that the higher the difference (= the larger the detours), the higher the risk for privacy breaches. This concept builds on top of the following assumption: Detours likely represent the "last mile" to customers of a courier service and therefore could reveal its customers.

## Implementation Steps

1. Sample geographical points from a given route at a certain rate.
2. Map collected points to real traffic structures such as streets and obtain routes connecting every pair of two consecutive points, represented as _line strings_. The complete route between start and end point is represented as a list of line strings. 
3. Compute the shortest paths between pairs of two sampled points and their positions on the road map. For this task we use [Nominatim](https://nominatim.org/). The Nominatim service returns _line strings_.
4. Calculate the distances between the actual and the optimal routes between each pair of sampled points.
5. Calculate a threshold value for the average difference and subtract it. The threshold value should adapt to the path between the two original samples. Set a lower threshold for the distance to account for inaccurate GPS measurements in both trajectories.
6. Derive a privacy risk from the average difference. Visualize the privacy risk of single line strings in the trajectory by assigning a certain color to them in the user interface.
