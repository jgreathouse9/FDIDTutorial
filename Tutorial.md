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
The FDID class makes a few assumptions about ones data structure. Firstly, it presumes that the user has a long panel dataset of 4 columns, where we have one column for the outcomes, one column for the time, one column of unit names, and one column for the treatment indicator, which in this case is 0 for all periods untreated, and 1 for Hong Kong during the treatment period. I will first go over DID, then AUGDID and the selection algorithm, as these are the main helper functions. Then I'll go into the ```fit``` method, the one users actually interact with.
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
The DID method is the main workhorse for the selection algorithm. It uses convex optimization to estimate DID; to some this may seem odd. However, I think it provides a better understanding of what DID is actually doing under the hood. Traditionally, we think of it as 4 averages and 3 subtractions. However it is also OLS, a constrained form of OLS where we

```math
\begin{align*}
\min_{\delta_0} \quad &\left\| y_0 - (\delta_0 + \delta_1 \cdot \mathbf{Y}_{\mathcal{N}_0}) \right\|_F^2 \quad \forall t \in \mathcal{T}_0 \\
&\text{subject to} \quad \delta_1 = 1
\end{align*}
```
As per the Augmented DD paper, we can think of DD in scalar form as a least-squares estimator $y_{0t}= \delta_1 + \delta_2\bar{y}_{jt}$ where $\bar{y}$ is the average of the control units. In other words, we seek the line which best fits the pre-intervention period of the treated unit, where the predictor is the average of the untreated units and its effect is constrained to be one plus some constant. This is what's meant above by the standard DD design not being robust to heterogeneous treatment effects, as in DD we constrain the effect of the average of the donors to be 1. Why? The parallel trends assumption. Under vanilla DID, we presume that an average, and only that average, is a suitable counterfactual, plus some constant. This is in contrast to the the Augmented DID method, where we do not make these constraints.
```python
    def AUGDID(self, datax, t, t1, t2, y, y1, y2):
        const = np.ones(t)      # t by 1 vector of ones (for intercept)
        # add an intercept to control unit data matrix, t by N (N=11)
        x = np.column_stack([const, datax])
        x1 = x[:t1, :]          # control units' pretreatment data matrix, t1 by N
        x2 = x[t1:, :]          # control units' pretreatment data matrix, t2 by N

        # ATT estimation by ADID method
        x10 = datax[:t1, :]
        x20 = datax[t1:, :]
        x1_ADID = np.column_stack([np.ones(x10.shape[0]), np.mean(x10, axis=1)])
        x2_ADID = np.column_stack([np.ones(x20.shape[0]), np.mean(x20, axis=1)])
        # Define variables
        b_ADID_cvx = cp.Variable(x1_ADID.shape[1])

        # Define the problem
        objective = cp.Minimize(cp.sum_squares(x1_ADID @ b_ADID_cvx - y1))
        problem = cp.Problem(objective)

        # Solve the problem
        problem.solve()

        # Extract the solution
        b_ADID_optimized = b_ADID_cvx.value

        # Compute in-sample fit
        y1_ADID = x1_ADID @ b_ADID_optimized

        # Compute prediction
        y2_ADID = x2_ADID @ b_ADID_optimized

        # Concatenate in-sample fit and prediction
        y_ADID = np.concatenate([y1_ADID, y2_ADID])

        ATT = np.mean(y2 - y2_ADID)  # ATT by ADID
        ATT_per = 100 * ATT / np.mean(y2_ADID)  # ATT in percentage by ADID

        e1_ADID = (
            y1 - y1_ADID
        )  # t1 by 1 vector of treatment unit's (pre-treatment) residuals
        sigma2_ADID = np.mean(e1_ADID**2)  # \hat sigma^2_e

        eta_ADID = np.mean(x2, axis=0).reshape(-1, 1)
        psi_ADID = x1.T @ x1 / t1

        Omega_1_ADID = (sigma2_ADID * eta_ADID.T) @ np.linalg.inv(psi_ADID) @ eta_ADID
        Omega_2_ADID = sigma2_ADID

        Omega_ADID = (t2 / t1) * Omega_1_ADID + Omega_2_ADID  # Variance

        ATT_std = np.sqrt(t2) * ATT / np.sqrt(Omega_ADID)

        alpha = 0.5
        quantile = norm.ppf(1 - alpha)

        lower_bound = ATT - quantile * np.sqrt(sigma2_ADID) / np.sqrt(t2)
        upper_bound = ATT + quantile * np.sqrt(sigma2_ADID) / np.sqrt(t2)

        RMSE = np.sqrt(np.mean((y1 - y1_ADID) ** 2))
        RMSEPost = np.sqrt(np.mean((y2 - y2_ADID) ** 2))

        R2_ADID = 1 - (np.mean((y1 - y1_ADID) ** 2)) / np.mean((y1 - np.mean(y1)) ** 2)

        new_dd_dict = {
            "ATT": ATT,
            "T0 RMSE": RMSE,
            "T1 RMSE": RMSEPost,
            "Percent ATT": ATT_per,
            "SATT": ATT_std[0, 0],
            "R2": R2_ADID,
        }

        return new_dd_dict, y_ADID
```
Notice how both methods return model fit statistics and ATTs in dictionaries. Here is the selection algorithm.
```python
    def selector(self, no_control, t1, t, y, y1, y2, datax, control_ID, df):
        R2 = np.zeros(no_control)
        R2final = np.zeros(no_control)
        control_ID_adjusted = np.array(control_ID) - 1
        select_c = np.zeros(no_control, dtype=int)

        for j in range(no_control):
            ResultDict = self.DID(y.reshape(-1, 1), datax[:t, j].reshape(-1, 1), t1)
            R2[j] = ResultDict["Fit"]["R-Squared"]
        R2final[0] = np.max(R2)
        first_c = np.argmax(R2)
        select_c[0] = control_ID_adjusted[first_c]

        for k in range(2, no_control + 1):
            left = np.setdiff1d(control_ID_adjusted, select_c[: k - 1])
            control_left = datax[:, left]
            R2 = np.zeros(len(left))

            for jj in range(len(left)):
                combined_control = np.concatenate(
                    (
                        datax[:t1, np.concatenate((select_c[: k - 1], [left[jj]]))],
                        datax[t1:t, np.concatenate((select_c[: k - 1], [left[jj]]))]
                    ),
                    axis=0
                )
                ResultDict = self.DID(y.reshape(-1, 1), combined_control, t1)
                R2[jj] = ResultDict["Fit"]["R-Squared"]

            R2final[k - 1] = np.max(R2)
            select = left[np.argmax(R2)]
            select_c[k - 1] = select
        selected_unit_names = [df.columns[i] for i in select_c]

        return select_c, R2final
```
Here is the forward selection algorithm. It follows the procedure described above to select the control group from the universe of controls. It takes as inputs the number of pre-intervention periods, post-periods and total time periods, as well as the treatment vector and the donor matrix. It also prints the names of the optimal control units. Now, is the estimation for the Forward DD
```python
    def est(self, control, t, t1, t2, y, y1, y2, datax):

        x1_forward_DID = np.mean(control[:t1, :], axis=1)
        x2_forward_DID = np.mean(control[t1:t, :], axis=1)
        beta_forward_DID = np.mean(y1 - x1_forward_DID)

        y1_hat_forward_DID = beta_forward_DID + x1_forward_DID
        y2_hat_forward_DID = beta_forward_DID + x2_forward_DID

        y_FDID = np.hstack((y1_hat_forward_DID, y2_hat_forward_DID))

        ATT_FDID = np.mean(y2 - y2_hat_forward_DID)
        ATT_FDID_per = 100 * ATT_FDID / np.mean(y2_hat_forward_DID)

        R2_forward_DID = 1 - (np.mean((y1 - y1_hat_forward_DID) ** 2)) / np.mean(
            (y1 - np.mean(y1)) ** 2
        )

        u1_FDID = y1 - y1_hat_forward_DID
        Omega_1_hat_FDID = (t2 / t1) * np.mean(u1_FDID**2)
        Omega_2_hat_FDID = np.mean(u1_FDID**2)
        std_Omega_hat_FDID = np.sqrt(Omega_1_hat_FDID + Omega_2_hat_FDID)

        ATT_std_FDID = np.sqrt(t2) * ATT_FDID / std_Omega_hat_FDID

        # Standardized ATT, assuming H0: ATT = 0

        p_value_forward_DID = 2 * (1 - norm.cdf(np.abs(ATT_std_FDID)))

        # P-value for a 1-sided test

        p_value_f_one_sided = 1 - norm.cdf(ATT_std_FDID)

        # 95% Confidence Interval for FDID ATT estimate

        z_critical = norm.ppf(0.975)  # 1.96 for a two-tailed test
        CI_95_FDID_left = ATT_FDID - z_critical * std_Omega_hat_FDID / np.sqrt(t2)
        CI_95_FDID_right = ATT_FDID + z_critical * std_Omega_hat_FDID / np.sqrt(t2)
        CI_95_FDID_width = [
            CI_95_FDID_left,
            CI_95_FDID_right,
            CI_95_FDID_right - CI_95_FDID_left,
        ]

        DID_dict = self.DID(y.reshape(-1, 1), datax, t1)

        FDID_dict = {
            "ATT": ATT_FDID,
            "Percent ATT": ATT_FDID_per,
            "R2": R2_forward_DID,
            "SATT": ATT_std_FDID,
            "T0 RMSE": np.sqrt(np.mean((y1 - y1_hat_forward_DID) ** 2)),
            # Add more metrics to the dictionary as needed
        }

        AUGDID_dict, y_ADID = self.AUGDID(datax, t, t1, t2, y, y1, y2)
        time_points = np.arange(1, len(y) + 1)

        return FDID_dict, DID_dict, AUGDID_dict, y_FDID
```
Notice how it returns the relevant dictionaries of fit and effect size estimates.
