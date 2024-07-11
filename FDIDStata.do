clear *

import delim "https://raw.githubusercontent.com/jgreathouse9/GSUmetricspolicy/main/data/GDP.csv", clear

//drop china
cls

* Creates the r-squared frame indexed to each individual unit

* Our time variable
gen time = _n

* Our treated unit
local treated_unit hongkong

qui ds

local strings `r(varlist)'
local test1 `treated_unit' time
local predictors: list strings- test1

* Get control units list
local predictors australia austria canada denmark finland france germany italy japan korea mexico netherlands newzealand norway switzerland unitedkingdom unitedstates singapore philippines indonesia malaysia thailand taiwan china

local treated_unit hongkong

local U // void

//local predictors australia newzealand austria canada denmark

tempvar ym cfp

tempname max_r2

while ("`predictors'" != "") {
    
    scalar `max_r2' = 0
    
    foreach var of local predictors {
        cap drop rss
    cap drop tss
        
        quietly {
            
            egen `ym' = rowmean(`U' `var')
            
            constraint 1 `ym' = 1
            
            cnsreg `treated_unit' `ym' if _n < 45 , constraint(1)
            
            predict `cfp' if e(sample)
        
            * Calculate the mean of observed values
        qui summarize `treated_unit'
        local mean_observed = r(mean)

        * Calculate the Total Sum of Squares (TSS)
        qui generate double tss = (`treated_unit' - `mean_observed')^2
        qui summarize tss
        local TSS = r(sum)

        * Calculate the Residual Sum of Squares (RSS)
        qui generate double rss = (`treated_unit' - `cfp')^2
        qui summarize rss
        local RSS = r(sum)

        * Calculate and display R-squared
        loc r2 = 1 - (`RSS' / `TSS')
        
            
            if `r2' > scalar(`max_r2') {
                
                scalar `max_r2' = `r2'
                local new_U `var'
                
            }
            
            drop `ym'
            drop `cfp'
            
        }
        
    }
    
    local U `U' `new_U'
    
    local predictors : list predictors - new_U

}


order time, first
order `U', a(`treated_unit')
cls

* Initialize an empty local to build the variable list
local varlist
local best_r2 = -1
local best_model = ""
* Loop through each variable in the list
foreach x of local U {
* Drop previous variables
    cap drop cf 
    cap drop ymean 
    cap drop tss
    cap drop rss
    * Add the current variable to the list
    local varlist `varlist' `x'
    
    * Display the current variable list (for debugging purposes)
    //display "`varlist'"
 
    
    * Generate the row mean for the current set of variables
    egen ymean = rowmean(`varlist')
    
    * Define the constraint
    constraint define 1 ymean = 1
    
    * Run the constrained regression
    qui cnsreg `treated_unit' ymean if time < 45, constraints(1)
    
    qui predict cf if e(sample)
    
    * Calculate the mean of observed values
	qui summarize `treated_unit'
	local mean_observed = r(mean)

	* Calculate the Total Sum of Squares (TSS)
	qui generate double tss = (`treated_unit' - `mean_observed')^2
	qui summarize tss
	local TSS = r(sum)

	* Calculate the Residual Sum of Squares (RSS)
	qui generate double rss = (`treated_unit' - cf)^2
	qui summarize rss
	local RSS = r(sum)

	* Calculate and display R-squared
	loc r2 = 1 - (`RSS' / `TSS')
    
    * Check if the current R-squared is the highest
    if (`r2' > `best_r2') {
        local best_r2 = `r2'
        local best_model = "`varlist'"
    }
}

* Display the best model and its R-squared value
di "Model with the highest R2 is with variables `best_model' having R2 = `best_r2'"

keep time `treated_unit' `best_model'

egen ymean = rowmean(`best_model')

* Define the constraint
constraint define 1 ymean = 1

* Run the constrained regression
qui cnsreg `treated_unit' ymean if time < 45, constraints(1)

qui predict cf

keep time `treated_unit' cf

su `treated_unit' if time > 44

loc yobs = r(mean)


su cf if time > 44

loc ypred = r(mean)

loc ATT = `yobs' - `ypred'

di "Our ATT is `ATT'"

twoway (connected hongkong time, mcolor(black) msymbol(diamond) lcolor(black) lwidth(medthick) connect(direct)) (connected cf time, mcolor(red) msymbol(triangle) lcolor(red) lwidth(medium)), legend(order(1 "Hong Kong" 2 "FDID Hong Kong") ring(0))



