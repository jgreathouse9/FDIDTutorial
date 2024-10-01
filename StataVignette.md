# ```fdid``` for Stata Users

Here, I cover the forward selection difference-in-differences method for Stata. Note that I already do the equivalent in [the Python vignette](https://github.com/jgreathouse9/FDIDTutorial/blob/main/Vignette.md). So, I will but briefly restate the algorithm and the basic ideas. For the more technical treatment, see [my paper](https://jgreathouse9.github.io/publications/FDIDSJ.pdf). This vignette demonstrates how to use FDID for Stata 16 and up. Users need ```sdid_event``` to be [installed](https://github.com/DiegoCiccia/sdid/tree/main/sdid_event#github).

First we install ```fdid``` and its help file into Stata like

```stata
net inst fdid, from("https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main") replace
```
We can get the datasets I include like
```stata
net get fdid, all
```

# The Algorithm

Basically, ```fdid``` uses a forward selection algorithm to choose the optimal control group for a single treated unit.  We observe $\mathcal{N} = \{1, 2, \ldots, N\}$ units where $\mathcal N$ has cardinality $N = |\mathcal{N}|$. $j=1$ is treated and controls are $\mathcal{N}\_0 = \mathcal{N} \setminus \{1\}$. Time is indexed by $t$. Denote pre-post-policy periods as $\mathcal{T}\_1 = \{1, 2, \ldots, T\_0\}$ and $\mathcal{T}\_2 = \{T\_0+1, \ldots, T\}$, where $\mathcal{T}= \mathcal{T}\_1 \cup \mathcal{T}\_2$. The subset of controls we wish to select are $\widehat{U} \subset \mathcal{N}\_0$, or the subset of controls. DiD is estimated like $y_{1t}=\hat\alpha_{\mathcal{N}\_0}+ \bar{y}\_{\mathcal{N}\_0t} \: t \in \mathcal{T}\_1$, where $\bar{y}\_{\mathcal{N}\_0t}\coloneqq \frac{1}{N\_0} \sum_{j \in \mathcal{N}\_0} y_{jt}$. The estimated least-squares intercept is computed like $\hat\alpha_{\mathcal{N}\_0} \coloneqq T_{1}^{-1}\sum_{t \in \mathcal{T}\_{1}}\left(y_{1t}-\bar{y}_{\mathcal{N}_0t}\right)$.


# HCW

The one we're replicating here is [the HCW dataset](https://doi.org/10.1002/jae.1230). We begin by importing the data

```stata
u "hcw.dta", clear
```

Here, we study the impact of Hong Kong's [economic integreation](https://www.henleyglobal.com/residence-investment/hong-kong/cepa-hong-kong-china). We have 44 pretreatment periods and 17 post-treatment periods. Our goal is to estimate the impact for those final 17 periods. To estimate ```fdid```, we simply do
```stata
fdid gdp, tr(treat) unitnames(state) ///
gr1opts(scheme(sj) ti(Forward DID Analysis) ///
yti(GDP Growth) ///
note(Treatment is Economic Integration with Mainland China) ///
legend(order(1 "Hong Kong" 2 "FDID Counterfactual") pos(12)))
```
We specify the outcome of interest as ```gdp``` and we specify the treatment as ```treat```. We use the strings of the ```state``` variable to define the names of our units. This syntax produces the table
```stata
Forward Difference-in-Differences   |    T0 R2: 0.843  T0 RMSE: 0.016
-----------------------------------------------------------------------------
         gdp |     ATT     Std. Err.     t      P>|t|    [95% Conf. Interval]
-------------+---------------------------------------------------------------
       treat |   0.02540    0.00462     5.49    0.000     0.01634     0.03447
-----------------------------------------------------------------------------
Treated Unit: hongkong
FDID selects philippines, singapore, thailand, norway, mexico, korea, indonesia, newzealand, malaysia, as the optimal donors.
See Li (2024) for technical details.
```

Pleasingly, these are the exact same results Kathy gets in her MATLAB code. Here is the plot:
<p align="center">
  <img src="fithongkong.png" alt="Alt Text">
</p>

If we wish to see the returned results, we can do
```stata

ereturn list

macros:
                  e(U) : "philippines, singapore, thailand, norway, mexico, korea, indonesia, newzealand, malaysia,"
         e(properties) : "b V"
             e(depvar) : "gdp"

matrices:
                  e(b) :  1 x 1
                  e(V) :  1 x 1
             e(series) :  61 x 9
            e(setting) :  1 x 6
            e(results) :  2 x 7
             e(dyneff) :  61 x 6

```
The ```e(series)``` is a matrix containing the observed and counterfactual values, event time, individual treatment effects. Naturally, the other statistics pertain to the total number of controls, the number of controls selected, as well as inferential statistics. 

```stata

mat l e(results)

e(results)[2,7]
            ATT       PATT         SE          t         LB         UB         R2
FDID  .02540494  53.843074  .00462405  5.4940862  .01634196  .03446791   .8427835
 DID  .03172116    77.6203  .00298081  10.641796  .02556907  .03787324      .5046
```
Here DID uses the robust standard error as estimated by ```xtdidregress```. We can clearly see that the pre-intervention \(R^2\) for the selected control group of FDID is much higher than the DID method, suggesting that the parallel trends assumption holds.

# Proposition 99

Next, I'd like to replicate one of the more classic papers in synthetic control methods, the case of Proposition 99 for California. Prop 99 was an anti-tobacco campaign that sought to reduce the rate of smoking in the population via education, awareness, and taxation. To run this, we do

```stata

clear *

u "smoking.dta", clear

cls

fdid cigsale, tr(treated) unitnames(state)
```
which returns the table
```stata
Forward Difference-in-Differences          T0 R2:     0.988     T0 RMSE:     1.282

-----------------------------------------------------------------------------------------
     cigsale |     ATT       Std. Err.      t       P>|t|    [95% Conf. Interval]
-------------+---------------------------------------------------------------------------
     treated | -13.64671     0.46016      29.66     0.000    -14.54861  -12.74481
-----------------------------------------------------------------------------------------
Treated Unit: California
FDID selects Montana, Colorado, Nevada, Connecticut, as the optimal donors.
See Li (2024) for technical details.
```

With these results, we may produce the plot

```stata

svmat e(series), names(col)

tsset year

lab var cigsale3 "California"

lab var cf3 "FDID"
lab var cfdd3 "DID"

lab var ymeandid "DID Control Mean"
lab var ymeanfdid "FDID Control Mean"
lab var year "Year"
twoway (tsline cigsale3) ///
(tsline cfdd3, lcolor(black) lwidth(thick) lpattern(dash)) ///
(tsline ymeandid, lcolor(black) lwidth(thick) lpattern(solid)), ///
scheme(sj) name(did, replace) ///
yti(Cigarette Consumption per Capita) tli(1989) legend(ring(0) pos(7) col(1) size(large)) ///
ti(Uses all controls)

twoway (tsline cigsale3) ///
(tsline cf3,lcolor(gs6) lwidth(thick) lpattern(longdash)) ///
(tsline ymeanfdid, lcolor(gs6) lwidth(thick) lpattern(solid)), ///
scheme(sj) name(fdid, replace) tli(1989) legend(ring(0) pos(7) col(1) size(large)) ///
ti(Uses 4 controls)


graph combine did fdid, xsize(8)

```

<p align="center">
  <img src="FDIDP99Update.png" alt="Alt Text">
</p>

I'll just quote [my notes](https://jgreathouse9.github.io/GSUmetricspolicy/treatmenteffects.html):

> The DID counterfactual underpredicts the true values for California in the pre-intervention period from around 1970 to 1975. Beyond this, it also overestimates the observed California’s values between 1980 and 1988. This is particularly bad because if your predictions diverge significantly from the treated unit’s observed values in the years right before the intervention takes place, why would we think that the post-intervention smoking consumption predictions are valid?

The R-squared of DID here is 0.604, versus FDID's R-squared of 0.988. This naturally has real implications for the analysis' findings. Because the fit for DID in the pre-intervention period is so poor (as a result of non-parallel trends holding across all control units), the DID method badly overestimates the causal effect, returning an ATT of -27.349. FDID fits much better. Its pre-intervention R-squared is higher than DID, meaning that the parallel trends assumption is much more likely to hold for FDID in this instance relative to DID. The effect sizes also differ, with FDID returning an ATT of -13.647. We may also, for additional visualization, looks at the ```e(series)``` to see how the total control group's means differs from the mean of the FDID control group.

To put this another way, the ATT of FDID is basically half of the DID estimate due to parallel trends bias. This is a colossal reduction of effect. Also of interest is that FDID selects 4 control states which happen to be the exact same states as the original synthetic control method selected. On top of this, we can also see that FDID gets these results without needing to use retail price of cigarettes, age, income, taxation, and outcome lags to attain what is essentially the same results of other synthetic control methods ([which tend to vary](https://rpubs.com/dwrich27/941298) between -13 and -19, depending on which [flavor](https://doi.org/10.48550/arXiv.2203.11576) of SCM we use). Of course, FDID assumes that a uniformly weighted average is the ideal way to model the counterfactual, but the point here is that we can get very similar results to the findings of the original model using a relatively simpler estimator which also happens to be qualitatively similar. An added benefit of DID is that inference is more straightforward compared to synthetic controls. In the staggered adoption case, we simply estimate one ATT per treated unit (using the never treated units as controls) and average the effect sizes together. Okay, so that's it for the vignette. No doubt people will have questions, suggestions, ideas, or errors to report, concerns, so you may contact me as ususal.

# Contact
- Jared Greathouse: <jgreathouse3@student.gsu.edu> (see [my website](https://jgreathouse9.github.io/))
