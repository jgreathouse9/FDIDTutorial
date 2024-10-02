# Forward Difference-in-Differences

This repository contains the Python and Stata code to estimate the Forward Difference-in-Differences estimator. The Python installation is under construction, but Stata users may install it by doing:
```
net install fdid, from("https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main") replace
```
where you may find the ancillary files with the Basque and HCW data. The vignette for Stata is [here](https://github.com/jgreathouse9/FDIDTutorial/blob/main/StataVignette.md).

- As of 8/11/2024, the Stata version reports Cohort ATTs, however this development is under construction. So, while users may do

```stata
net from "https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main"
net install fdid, replace
net get fdid, replace
// Reinstall fdid so you have the most recent version
clear *
qui u basque
// Import basque
replace treat = 1 if id==12 & year >=1976
// Pretend Extremadura (a unit that'll never be a donor) was treated
cls
// Estimate FDID
fdid gdpcap, tr(treat)
mat l e(results)
```

to obtain the basic results, I'm working on developing event study estimates for this as well. So, discussion of staggered adoption is omitted from the Stata Vignette (I'll also do this for the Python version too, after I send the Stata version to Stata Journal)

## Troubleshooting Forward DID

For anyone who wishes to open an issue about trouble with the Stata code, please provide a minimal worked example of what you did. For example,

```stata
clear *
cls
u "hcw.dta", clear


fdid gdp, tr(treatt) unitnames(state)  gr2opts(scheme(sj) name(hcwte))
```
Is a minimum worked example because it takes me, from start to finish, how the data was imported and it allows me to see what the error is (the fact the treat variable was spelled wrong).

Screenshots of your Stata terminal or other things that don't allow me to reproduce the problem aren't helpful in terms of debugging, so including an example which reproduces the problem is the best way to raise issues.
