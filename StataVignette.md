# Forward DID for Stata Users

I've already covered the basics of the algorithm in [the Python vignette](https://github.com/jgreathouse9/FDIDTutorial/blob/main/Vignette.md), so I will not here reinvent the wheel. So, this note simply demonstrates how to use FDID for Stata 16 and up.

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

Here, we study the impact of Hobng Kong's [economic integreation](https://www.henleyglobal.com/residence-investment/hong-kong/cepa-hong-kong-china). To do this in Stata, we simply do

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

If we wish to see the results, we can do
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

g treated = 1 if state == "California" & year > 1988

replace treated = 1 if state == "Massachusetts" & year > 1992

replace treated = 0 if treated ==.

egen id = group(state)

xtset id year

order id state year packs treated

drop if inlist(state,"Arizona", "Oregon", "Florida","Alaska", "Hawaii")

drop if inlist(state,"Maryland", "Michigan", "New Jersey", "New York", "Washington")

cls
```
We're concerened with [Proposition 99](https://en.wikipedia.org/wiki/1988_California_Proposition_99) and [the Massachusetts tobacco program](https://www.cdc.gov/mmwr/preview/mmwrhtml/00044337.htm#:~:text=The%20Massachusetts%20Tobacco%20Control%20Program%20(MTCP)%2C%20administered,early%201994%2C%20the%20program%20began%20funding%20local). So we estimate:

```stata
fdid packs if inrange(year,1970,2004) & id != 9, tr(treated) unitnames(state)

mat l r(results)
```
which returns in the ```return list```
```stata
r(results)[1,2]
                ATT        TATT
Effects  -21.150942  -592.22637
```
where we have the average treatment effect on the treated and the total treatment effect. The frame ```multiframe```, returned only when we have $N\_{\text{tr}}>1$, contains the event time effects, where users may create event study sytle plots should they wish. The matrix ```r(series)``` is the exact same thing in matrix form. For those curious about the specific controls selected, these may be found in the notes. For example, for California, we can change to tis frame from the default one

```stata
frame change fdid_cfframe5
notes
```
which returns
```
_dta:
  1.  The selected units are "Montana, Colorado, Nevada, Connecticut,"
```

Okay, so that's it for the vignette. No doubt people will have questions, suggestions, ideas, or concerns, so you may contact me as ususal.

# Contact
- Jared Greathouse: <jgreathouse3@student.gsu.edu> (see [my website](https://jgreathouse9.github.io/))
- Kathy Li: <kathleen.li@mccombs.utexas.edu>
