A Tutorial on Forward and Augmented Difference-in-Differences 
==============

***Revisiting Hong Kong's Economic Integration***

**Author:** *Jared Greathouse*
<!-- 
  Code to Justify Text

-->
# Introduction
This tutorial uses publicly available data to demonstrate the utility of the [Forward](https://doi.org/10.1287/mksc.2022.0212) and [Augmented](https://doi.org/10.1287/mksc.2022.1406) Difference-in-Differences estimators. It is based on the MATLAB code kindly provided by Kathleen Li.

We estimate the counterfactual GDP Growth for Hong Kong had their economy never economically integrated, [revisiting](https://doi.org/10.1002/jae.1230) one of the original case studies for the panel data approach method to program evaluation. This tutorial will consist of two parts: firstly it will go over the anatomy of the class itself. Then I will go over the way to actually estimate the model. However, first we have some preliminaries:
## Prerequisite Libraries
```python
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
import cvxpy as cp
```
Strictly speaking, you don't need to import ```matplotlib```, I only do so because I am customizing my own graphics.
## Model Primitives
Before I continue, I introduce the potential outcomes framework. Our treatment of interest is the economic integration of Hong Kong. The basic problem of causal inference however is that time moves forward-- we cannot go into the past and see how Hong Kong's GDP Growth had their economy never economically integrated. Thus, we observe $y_{jt} = d_{jt} y_{jt}^1 + (1 - d_{jt}) y_{jt}^0$ where $d \in \[0,1\]$ is a dummy variable indicating exposure to the intervention of interest. Here, we have units indexed to $j$ across $t \in \left(1, T\right) \cap \mathbb{N}$ time periods. Here, $j=0$ is our treated unit, leaving us with $1 \ldots N$ control units. $y_{jt}^1$ and $y_{jt}^0$ respectively are the outcomes we would observe if a given unit $j$ is treated at time $t$. However for Hong Kong, we never observe $y_{jt}^0$ while $t \ge T_0$, we only observe its outcomes under treatment or not while $t< T_0$. Fundamentally, the DID estimator is valid based on the parallel trends assumption. The classic parallel trends assumption posits that had the intervention never occured, the post-intervention trend of our treated unit would be parallel to the average of the control group $\mathcal{N}_0$. Practically speaking, this has a few implications: firstly, no heterogeneous treatment effects. Parallel trends posits that the average, and the average only, of our controls is a sufficient proxy for our coutnerfactual Hong Kong. As a corollary, this means that all of our control units are actually similar enough to Hong Kong on observed and onobserved factors to serve as proxy. In other words, this means we ideally would have no outliers or units that are too different from Hong Kong in the control group.

However in practive, this is rarely the case in any standard dataset. There are 195 national economies in the world; presumably, most of them would not be sufficient comparison units for Hong Kong.
## Introducing Forward Selection
Hence, it's useful to have an algorithm which can formally select the optimal set of control units for a given treated unit. Here, we use the forward selection method. In FDID, we take the full pool of control units outcomes for $t< T_0$ and iteratively use each control unit to predict, via OLS, $t< T_0$ outcomes of the treated unit. We then store the model which has the highest $R^2$ statistic. For example, if we have Los Angeles, New York City, Chicago, and Miami as controls and Los Angeles has the higest $R^2$, we store Los Angeles as the first selected unit. Then, we predict the pre-intervention outcomes for the treated unit using Los Angeles, looping through the other unselected control units. If NYC and Los Angeles have the highest $R^2$, we then add NYC to the selected pool. Then, we estimate DID with these two control units, and calculate its $R^2$ statistic. Then we keep going, until we estimate And so on, until we estimate the sum of $1 \ldots N$ models. After we've done so, we then keep whichever selected model that has the highest $R^2$ statistic. Intuitively, this is an improvement over the standard average: if the average of control does not provide a good approximation fro our treated unit, then using some subset of the controls must perform at least as well. The final pool of control units is the final DID model we estimate. Now that we have the basics out of the way, we can finally go over the actual Python class.
# Forward Selection Differences in Differences
```python
class FDID:
    def __init__(self, df, unitid, time, outcome, treat,
                 figsize=(12, 6),
                 graph_style="default",
                 grid=True,
                 counterfactual_color="red",
                 treated_color="black",
                 filetype="png",
                 display_graphs=True,
                 ):
        self.df = df
        self.unitid = unitid
        self.time = time
        self.outcome = outcome
        self.treated = treat
        self.figsize = figsize
        self.graph_style = graph_style
        self.grid = grid
        self.counterfactual_color = counterfactual_color
        self.treated_color = treated_color

        self.filetype = filetype
        self.display_graphs = display_graphs
```
The FDID class makes a few assumptions about ones data structure. Firstly, it presumes that the user has a long panel dataset of 4 columns, where we have one column for the outcomes, one column for the time, one column of unit names, and one column for the treatment indicator, which in this case is 0 for all periods untreated, and 1 for Hong Kong during the treatment period. 
```python
      Country     GDP  Time  Integration
0   Hong Kong  0.0620     0            0
1   Hong Kong  0.0590     1            0
2   Hong Kong  0.0580     2            0
3   Hong Kong  0.0620     3            0
4   Hong Kong  0.0790     4            0
..        ...     ...   ...          ...
56      China  0.1110    56            0
57      China  0.1167    57            0
58      China  0.1002    58            0
59      China  0.1017    59            0
60      China  0.1238    60            0

[1525 rows x 4 columns]
```
The user specifies the dataframe they wish to use, as well as the 4 columns I just mentioned. The user may also customize, should they specify to see graphs, the colors of the trend lines for the observed and FDID predictions.
```python
    def DID(self, y, datax, t1):
        t = len(y)

        x1, x2 = np.mean(datax[:t1], axis=1).reshape(-1,
                                                     1), np.mean(datax[t1:t], axis=1).reshape(-1, 1)
        # Define the variables to be optimized
        constant = cp.Variable()
        beta = cp.Variable()
        constraint = [beta == 1]

        # Define the constraint: beta should be equal to 1
        # Define the objective function
        objective = cp.Minimize(cp.sum_squares(y[:t1] - (constant + beta * x1)))

        # Define the problem with the constraint
        problem = cp.Problem(objective, constraint)

        # Solve the problem
        problem.solve()

        # Get the optimized values
        constant_optimized = constant.value
        # Calculate y_DID using vectorized operations
        y_DID = constant_optimized + np.concatenate((x1, x2))
        y1_DID, y2_DID = y_DID[:t1], y_DID[t1:t]

        # DID ATT estimate and percentage

        ATT_DID = np.mean(y[t1:t] - y_DID[t1:t])
        ATT_DID_percentage = 100 * ATT_DID / np.mean(y_DID[t1:t])

        # DID R-square

        R2_DID = 1 - (np.mean((y[:t1] - y_DID[:t1]) ** 2)) / (
            np.mean((y[:t1] - np.mean(y[:t1])) ** 2)
        )

        # Estimated DID residual

        u1_DID = y[:t1] - y_DID[:t1]

        # \hat \Sigma_{1,DID} and \hat \Sigma_{2,DID}
        t2 = t - t1

        Omega_1_hat_DID = (t2 / t1) * np.mean(u1_DID**2)
        Omega_2_hat_DID = np.mean(u1_DID**2)

        # \hat Sigma_{DID}

        std_Omega_hat_DID = np.sqrt(Omega_1_hat_DID + Omega_2_hat_DID)

        # Standardized ATT_DID

        ATT_std_DID = np.sqrt(t2) * ATT_DID / std_Omega_hat_DID

        # P-value for H0: ATT=0

        p_value_DID = 2 * (1 - norm.cdf(np.abs(ATT_std_DID)))

        # P-value for 1-sided test

        p_value_one_sided = 1 - norm.cdf(ATT_std_DID)

        # 95% Confidence Interval for DID ATT estimate

        z_critical = norm.ppf(0.975)  # 1.96 for a two-tailed test
        CI_95_DID_left = ATT_DID - z_critical * std_Omega_hat_DID / np.sqrt(t2)
        CI_95_DID_right = ATT_DID + z_critical * std_Omega_hat_DID / np.sqrt(t2)
        CI_95_DID_width = [
            CI_95_DID_left,
            CI_95_DID_right,
            CI_95_DID_right - CI_95_DID_left,
        ]

        # Metrics of fit subdictionary
        Fit_dict = {
            "T0 RMSE": round(np.std(y[:t1] - y1_DID), 3),
            "R-Squared": round(R2_DID, 3)
        }

        # ATTS subdictionary
        ATTS = {
            "ATT": round(ATT_DID, 3),
            "Percent ATT": round(ATT_DID_percentage, 3),
            "SATT": round(ATT_std_DID, 3),
        }

        # Inference subdictionary
        Inference = {
            "P-Value": round(p_value_DID, 3),
            "95 LB": round(CI_95_DID_left, 3),
            "95 UB": round(CI_95_DID_right, 3),
        }

        # Vectors subdictionary
        Vectors = {
            "Observed Unit": np.round(y, 3),
            "DID Unit": np.round(y_DID, 3),
            "Gap": np.round(y - y_DID, 3)
        }

        # Main dictionary
        DID_dict = {
            "Effects": ATTS,
            "Vectors": Vectors,
            "Fit": Fit_dict,
            "Inference": Inference
        }

        return DID_dict

```
This method is the main workhorse for the selection algorithm. It uses convex optimization to estimate DID; to some this may seem odd. However, I think it provides a better understanding of what DID is actually doing under the hood. Traditionally, we think of it as 4 averages and 3 subtractions. However it is also OLS, a constrained form of OLS where we

```math
\begin{align*}
\text{minimize} \quad & \sum_{t=1}^{T_0} ||y_j - (\beta_0 + \boldsymbol{\beta} \cdot \mathbf{Y}_{\mathcal{N}_{0}||_{F}^2 \: \forall t \in \mathcal{T}_0 \\
\text{subject to} \quad & \boldsymbol{\beta} = 1
\end{align*}
```
As per the Augmented DD paper, we can think of DD in scalar form as a least-squares estimator $y_{0t}= \delta_1 + \delta_2\bar{y}_{jt}$ where $\bar{y}$ is the average of the control units. In other words, we seek the line which best fits the pre-intervention period of the treated unit, where the predictor is the average of the untreated units and its effect is constrained to be one plus some constant. This is what's meant above by the standard DD design not being robust to heterogeneous treatment effects, as in DD we constrain the effect
