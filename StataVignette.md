# ```fdid``` for Stata Users

I've already covered the basics of the algorithm in [the Python vignette](https://github.com/jgreathouse9/FDIDTutorial/blob/main/Vignette.md), so I will not repeat myself. This note simply demonstrates how to use FDID for Stata 16 and up. No special commands are needed to use ```fdid```.

First we install fdid into Stata like

```stata
net inst fdid, from("https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main") replace
```

We can get the datasets I include like

```stata
net get fdid, all
```

# HCW

The one we're replicating here is [the HCW dataset](https://doi.org/10.1002/jae.1230). We begin by importing the data

```stata
u "hcw.dta", clear
```

Here, we study the impact of Hong Kong's [economic integreation](https://www.henleyglobal.com/residence-investment/hong-kong/cepa-hong-kong-china). To do this in Stata, we simply do

```stata
fdid gdp, tr(treat) unitnames(state) ///
gr1opts(scheme(plottig) ti(Forward DID Analysis) /// using plottig: https://www.stata.com/meeting/switzerland16/slides/bischof-switzerland16.pdf
yti(GDP Growth) note(Treatment is Economic Integration with Mainland China) legend(order(1 "Hong Kong" 2 "FDID Counterfactual") pos(12)))
```

We specify the outcome of interest as ```gdp``` and we specify the treatment as ```treat```. We use the strings of the ```state``` variable to define the names of our units. This syntax produces the table

```stata

Forward Difference-in-Differences          T0 R2:  0.84278     T0 RMSE:  0.01638

-----------------------------------------------------------------------------
         gdp |     ATT     Std. Err.     t      P>|t|    [95% Conf. Interval]
-------------+---------------------------------------------------------------
       treat |   0.02540    0.00462     5.49    0.000     0.01634     0.03447
-----------------------------------------------------------------------------
Treated Unit: hongkong
FDID selects philippines, singapore, thailand, norway, mexico, korea, indonesia, newzealand, malaysia, as the optimal donors.
See Li (2024) for technical details.
```

Here is the plot:
<p align="center">
  <img src="fithongkong.png" alt="Alt Text">
</p>

If we wish to see the returned results, we can do
```stata

return list

macros:
                  r(U) : "philippines, singapore, thailand, norway, mexico, korea, indonesia, newzealand, malaysia,"

matrices:
             r(series) :  61 x 5
            r(results) :  1 x 7
```
The ```r(series)``` is a matrix containing the observed and counterfactual values, calendar time, event time, and individual treatment effects.

The results themselves can also be conveniently accessed like
```stata

mat l r(results)

r(results)[1,7]
                  ATT         SE          t         LB         UB         R2       RMSE
Statistics  .02540494  .00462405  5.4940862  .01634196  .03446791   .8427835     .01638
```
Pleasingly, these are the exact same results Kathy gets in her MATLAB code.

Frames, labeled by default *fdidcfframe* *panelid*, are also returned.

```stata
frame dir
```
returns

```

  default        1525 x 5; hcw.dta
* fdid_cfframe9  61 x 5

Note: Frames marked with * contain unsaved data.
```
These contain the observed and counterfactual values, the pointwise treatment effect and its standard error, as well as the event time variable.

## Staggered Adoption

Okay let's do staggered adoption. We begin by pulling in and cleaning the smoking data.
```stata

clear *

import delim "https://data.cdc.gov/api/views/7nwe-3aj9/rows.csv?accessType=DOWNLOAD&api_foundry=true"

keep locationdesc year data_value data_value_type

keep if strpos(data_value_type,"Pack")

drop data_value_type

rename (*) (state year packs)

destring year, replace

g treated = 1 if state == "California" & year >= 1989

replace treated = 1 if state == "Massachusetts" & year >= 1993

replace treated = 1 if state == "Arizona" & year >= 1995

replace treated = 1 if state=="Florida" & year >= 1998

replace treated = 1 if state== "Oregon" & year >= 1997

replace treated = 0 if treated ==.

egen id = group(state)

xtset id year

order id state year packs treated

drop if inlist(state, "Alaska", "Hawaii")

drop if inlist(state,"Maryland", "Michigan", "New Jersey", "New York", "Washington")
cls


```
We're concerened with [Proposition 99](https://ballotpedia.org/California_Proposition_99,_Tobacco_Tax_Increase_Initiative_(1988)), [the Massachusetts tobacco program](https://ballotpedia.org/Massachusetts_Question_1,_Excise_Tax_on_Cigarettes_and_Smokeless_Tobacco_Initiative_(1992)), [Arizona](https://ballotpedia.org/Arizona_Proposition_200,_Tobacco_Tax_and_Healthcare_Initiative_(1994)), [Florida](http://www.cnn.com/US/9805/08/tobacco.implementation/(https://www.swatflorida.com/get-to-know-us/)), and [Oregon](https://ballotpedia.org/Oregon_Measure_44,_Cigarette_and_Tobacco_Tax_Increase_Initiative_(1996)). So we estimate:

```stata
fdid packs if inrange(year,1970,2004), tr(treated) unitnames(state)
```
which displays the table
```stata
Staggered Forward Difference-in-Differences
-----------------------------------------------------------------------------
       packs |     ATT     Std. Err.     t      P>|t|    [95% Conf. Interval]
-------------+---------------------------------------------------------------
     treated | -15.71868    0.53376   -29.45    0.000   -16.76484   -14.67253
-----------------------------------------------------------------------------
See Li (2024) for technical details.
Effects are calculated in event-time using never-treated units.
```
This can also be accessed later by ```mat l r(results)```. For those curious about the specific controls selected for each treated unit, these may be found in the notes of the unit specific dataframe. For example, for California, we can change to its frame from the original one

```stata
frame change fdid_cfframe5
notes
```
which returns
```
_dta:
  1.  The selected units are "Montana, Colorado, Nevada, Connecticut,"
```

Note of course that this does not purport to be a comprehensive analysis of tobacco policy in this setting, it is purely for demonstration using additional examples that [Abadie (2010)](https://doi.org/10.1198/jasa.2009.ap08746) had to skip over.

We may also use the standard error of the pointwise treatment effect coefficient to create event-study style plots, complete with 95% confidence intervals. In the future, I may make this an option as well, along with [Cohort ATT](https://cran.r-project.org/web/packages/did/vignettes/TWFE.html) options, using [the not-yet treated units](https://bcallaway11.github.io/did/articles/multi-period-did.html) as controls, or other features that would make ```fdid``` appealing to a wider audience.


Okay, so that's it for the vignette. No doubt people will have questions, suggestions, ideas, or errors to report, concerns, so you may contact me as ususal.

# Contact
- Jared Greathouse: <jgreathouse3@student.gsu.edu> (see [my website](https://jgreathouse9.github.io/))
