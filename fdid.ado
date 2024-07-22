*****************************************************
set more off

*****************************************************

* Programmer: Jared A. Greathouse

* Institution:  Georgia State University

* Contact: 	j.greathouse200@gmail.com

* Created on : Jul 12, 2024

* Contents: 1. Purpose

*  2. Program Versions

*****************************************************

* 1. Purpose

** Programs Forward DD from 
** Li : https://doi.org/10.1287/mksc.2022.0212 

* 2. Program

prog define fdid, eclass

cap frame drop __reshape
qui frame 
loc originalframe: di r(currentframe)

/**********************************************************

	
	
	* Preliminaries*


If the data aren't a balanced panel, something about the
user's dataset ain't right.
**********************************************************/


cap qui xtset
if _rc {
	
	disp as err "The data are not xtset"
	exit 498
}

cap qui xtset

cap as r(balanced)=="strongly balanced"

if _rc {
	disp as err "The data must be strongly balanced"
	exit 498	
	
}

cap as r(gaps)==0


if _rc {
	disp as err "The data cannot have gaps"
	exit 498	
	
}


loc time: disp "`r(timevar)'"

loc panel: disp "`r(panelvar)'"



marksample touse

_xtstrbal `panel' `time' `touse'

	syntax anything [if], ///
		TReated(varname) /// We need a treatment variable as 0 1
		[gr1opts(string asis)] [gr2opts(string asis)] ///
		[unitnames(varlist max=1 string)]
		

		
   /* if unitname specified, grab the label here */
if "`unitnames'" != "" {
	
	local curframe = c(frame)
	

cap frame drop __dfcopy

frame copy `curframe' __dfcopy

	cwf __dfcopy
	
    
	qui levelsof `panel',local(levp)
   	
	
	
	
   /* check if var exists */
    capture confirm string var `unitnames'
            if _rc {
                di as err "`unitnames' does not exist as a (string) variable in dataset"
                exit 198
            }
    /* check if it has a value for all units */
    tempvar pcheck
    qui egen `pcheck' = sd(`panel') , by(`unitnames')
    qui sum `pcheck'
    if r(sd) != 0 {
        di as err "`unitnames' varies within units of `panel' - revise unitnames variable "
        exit 198
    }
    local clab "`panel'"
    tempvar index
    gen `index' = _n
   /* now label the pvar accoringly */
    foreach i in `levp' {
        qui su `index' if `panel' == `i', meanonly
        local label = `unitnames'[`r(max)']
        local value = `panel'[`r(max)']
        qui label define `clab' `value' `"`label'"', modify
     }
   label value `panel' `clab'
   }
			


if "`: value label `panel''" == "" & "`unitnames'" == ""  {
	
	di as err "Your panel variable NEEDS to have a value label attached to it."
	
	di as err "Either specify -unitnames- or pre-assign your panel id with string value labels."
	
	exit 198
}


tempvar touse
mark `touse' `if' `in'

if (length("`if'")+length("`in'")>0) {
    
    qui keep if `touse'
}

		
gettoken depvar anything: anything

unab depvar: `depvar'

local y_lab: variable lab `depvar'

loc outlab "`y_lab'" // Grabs the label of our outcome variable

local tr_lab: variable lab `treated'

qui levelsof `panel' if `treated'==1, loc(trunit)

local nwords :  word count `trunit'

if `nwords' > 1 {
    matrix empty_matrix = J(1, 8, .)  // Initialize an empty matrix with 0 rows and 7 columns

    matrix combined_matrix = empty_matrix  // Initialize combined_matrix as empty
}



foreach x of loc trunit {
	
local curframe = c(frame)

cap frame drop __dfcopy

if "`curframe'" != "__dfcopy" {

frame copy `curframe' __dfcopy
}
frame __dfcopy {


loc trunitstr: display "`: label (`panel') `x''"

if "`cfframename'" == "" {
	
	cap frame drop fdid_cfframe`x'
	
	loc defname fdid_cfframe`x'
}

else if `nwords' > 1 {
	
	loc defname `cfframename'`x'
}

else {
	
	loc defname `cfframename'
}

numcheck, unit(`panel') ///
	time(`time') ///
	transform(`transform') ///
	depvar(`depvar') /// Routine 1
	treated(`treated') cfframe(`defname') trnum(`x')


// Routine 2

treatprocess, time(`time') ///
	unit(`panel') ///
	treated(`treated') trnum(`x')
	
loc trdate = e(interdate)

	
/**********************************************************

	
	
	* Estimation*


This is where we do estimation if the dataset above passes.
**********************************************************/

est_dd, time(`time') ///
	interdate(`trdate') ///
	intname(`tr_lab') ///
	outlab(`outlab') ///
	gr1opts(`gr1opts') ///
	gr2opts(`gr2opts') ///
	treatst(`trunitstr') ///
	panel(`panel') ///
	outcome(`depvar') ///
	trnum(`x') treatment(`treated') ///
	cfframe(`defname') ntr(`nwords')
	
        if `nwords' > 1 {
            matrix resmat = e(results)
            matrix combined_matrix = combined_matrix \ resmat
        }

}

if `nwords'==1 {


mat series = e(series)
ereturn mat series= series
}
}

if `nwords' > 1 {
	
mkf fdidmatrixres
cwf fdidmatrixres
	
* Convert the matrix "results" into a dataset named "myresults"
qui svmat combined_matrix, names(col)
keep c1 c2 c3

qui drop in 1

qui rename (*) (ATT PATT SE)

qui su ATT, mean

// Calculate the combined ATT
scalar ATT_combined = r(mean)

qui su PATT, mean

scalar PATT_combined = r(mean)

qui su SE

// Calculate the combined variance
scalar var_combined = r(Var)

// Calculate the combined SE
scalar SE_combined = sqrt(scalar(var_combined))

// Calculate the 95% Confidence Interval
scalar CI_lower = scalar(ATT_combined) - (invnormal(0.975) * scalar(SE_combined))
scalar CI_upper = scalar(ATT_combined) + (invnormal(0.975) * scalar(SE_combined))

    // Assume you have already computed ATT_combined and SE_combined
    // Calculate t-statistic
    scalar tstat = scalar(ATT_combined) / scalar(SE_combined)

    // Calculate p-value (two-sided)
    scalar p_value = 2 * (1 - normal(abs(scalar(tstat))))

* Clear any previous display
di as text ""
di as res "`tabletitle'"
di as text ""
di as text "{hline 13}{c TT}{hline 75}"
di as text %12s abbrev("`depvar'",12) " {c |}     ATT       Std. Err.      t       P>|t|    [95% Conf. Interval]" 
di as text "{hline 13}{c +}{hline 75}"
di as text %12s abbrev("`treated'",12) " {c |} "%9.5f scalar(ATT_combined) "   "  %9.5f scalar(SE_combined) "  " %9.2f scalar(tstat) " " %9.3f scalar(p_value) "    " %9.5f scalar(CI_lower) "  " %9.5f scalar(CI_upper)
di as text "{hline 13}{c BT}{hline 75}"
di as text "See Li (2024) for technical details."
di as text "Effects are calculated in event-time using never-treated units."


tempname my_matrix
matrix `my_matrix' = (scalar(ATT_combined), scalar(PATT_combined), scalar(SE_combined), scalar(tstat), scalar(CI_lower), scalar(CI_upper))
matrix colnames `my_matrix' = ATT PATT SE t LB UB

matrix rownames `my_matrix' = Result
ereturn clear
ereturn mat results= `my_matrix'

}
cwf `originalframe'
qui cap frame drop fdidmatrixres
cap frame drop `defname'
qui frame drop __dfcopy
end

/**********************************************************

	*Section 1: Data Setup
	
**********************************************************/

prog numcheck, eclass
// Original Data checking
syntax, ///
	unit(varname) ///
	time(varname) ///
	depvar(varname) ///
	[transform(string)] ///
	treated(varname) cfframe(string) trnum(numlist min=1 max=1 >=1 int)
	
		
/*#########################################################

	* Section 1.1: Extract panel vars

	Before DD can be done, we need panel data.
	
	a) Numeric
	b) Non-missing and
	c) Non-Constant
	
*########################################################*/

/*
di as txt "{hline}"
di as txt "Forward Difference in Differences"
di as txt "{hline}"
*/


/*The panel should be balanced, but in case it isn't somehow, we drop any variable
without the maximum number of observations (unbalanced) */


	foreach v of var `unit' `time' `depvar' {
	cap {	
		conf numeric v `v', ex // Numeric?
		
		as !mi(`v') // Not missing?
		
		qui: su `v'
		
		as r(sd) ~= 0 // Does the unit ID change?
	}
	}
	
	if _rc {
		
		
		
		disp as err "All variables `unit' (ID), `time' (Time) and `depvar' must be numeric, not missing and non-constant."
		exit 498
	}
	


frame put `time' if `unit' == `trnum', into(`cfframe')
	
end


prog treatprocess, eclass
        
syntax, time(varname) unit(varname) treated(varname) trnum(numlist min=1 max=1 >=1 int)

/*#########################################################

	* Section 1.2: Check Treatment Variable

	Before DD can be done, we need a treatment variable.
	
	
	The treatment enters at a given time and never leaves.
*########################################################*/


qui xtset
loc time_format: di r(tsfmt)

qui su `time' if `treated' ==1 & `unit'==`trnum'

loc last_date = r(max)
loc interdate = r(min)

qui su `unit' if `treated'==1

loc treated_unit = r(mean)

qui insp `time' if `treated' ~= 1 & `unit'==`treated_unit'

loc npp = r(N)


	if !_rc {
		
		su `unit' if `treated' ==1, mean
		
		loc clab: label (`unit') `treated_unit'
		loc adidtreat_lab: disp "`clab'"
		
		
		qui: levelsof `unit' if `treated' == 0 & `time' > `interdate', l(labs)

		local lab : value label `unit'

		foreach l of local labs {
		    local all `all' `: label `lab' `l'',
		}

		loc controls: display "`all'"

		//display "Treatment is measured from " `time_format' `interdate' " to " `time_format'  `last_date' " (`npp' pre-periods)"
		
		
		qui su `unit' if `treated' == 0
		
		loc dp_num = r(N) - 1
		
		cap as `dp_num' >= 2
		if _rc {
			
		di in red "You need at least 2 donors for every treated unit"
		exit 489
		}
		//di as res "{hline}"

	}	

ereturn loc interdate = `interdate'


end



prog est_dd, eclass
	
syntax, ///
	time(varname) ///
	interdate(numlist min=1 max=1 >=1 int) ///
	[intname(string)] ///
	[outlab(string)] ///
	[gr1opts(string asis)] ///
	[gr2opts(string asis)] ///
	treatst(string asis) ///
	panel(string asis) ///
	outcome(string asis) ///
	trnum(numlist min=1 max=1 >=1 int) ///
	treatment(string) [outlab(string asis)] ///
	cfframe(string) ntr(numlist min=1 max=1 >=1 int)

	
local curframe = c(frame)

tempname __reshape

frame copy `curframe' `__reshape'

cwf `__reshape'


qbys `panel': egen et = max(`treatment')

qui keep if et ==0 | `panel'==`trnum'


qui keep `panel' `time' `outcome'


qui reshape wide `outcome', j(`panel') i(`time')

qui: tsset `time'
loc time_format: di r(tsfmt)


format `time' `time_format'

order `outcome'`trnum', a(`time')

qui ds

loc temp: word 1 of `r(varlist)'

loc time: disp "`temp'"

loc t: word 2 of `r(varlist)'

loc treated_unit: disp "`t'"

local strings `r(varlist)'


local trtime `treated_unit' `time'

local predictors: list strings- trtime

loc N0: word count `predictors'

// Set U is a now empty set. It denotes the order of all units whose values,
// when added to DID maximize the pre-period r-squared.

local U

// We use the mean of the y control units as our predictor in DID regression.
// cfp= counterfactual predictions, used to calculate the R2/fit metrics.

tempvar ym cfp


// Here is the placeholder for max r2.
tempname max_r2


// Forward Selection Algorithm ...
  
qui summarize `treated_unit'
local mean_observed = r(mean)
    
* Calculate the Total Sum of Squares (TSS)
qui generate double tss = (`treated_unit' - `mean_observed')^2
qui summarize tss
local TSS = r(sum) 

while ("`predictors'" != "") {

scalar `max_r2' = 0

	foreach var of local predictors {
	
	// Drops these, as we need them for each R2 calculation
	
	cap drop rss
	cap drop tss

		 {
			
		// We take the mean of each element of set U and each new predictor.
			
		    
		 egen `ym' = rowmean(`U' `var')
		    
		 // The coefficient for the control average has to be 1.
		    
		 constraint 1 `ym' = 1
		    
		 // We use constrained OLS for this purpose.
		    
		 * 45 is the treatment date for HongKong, but this will be
		 * any date as specified by the treat variable input by the
		 * user.
		 
		    
		qui cnsreg `outcome'`trnum' `ym' if `time' < `interdate' , constraint(1)
		    
		// We predict our counterfactual
		    
		qui predict `cfp' if e(sample)
		
		// Now we calculate the pre-intervention R2 statistic.


		* Calculate the Residual Sum of Squares (RSS)
		qui generate double rss = (`treated_unit' - `cfp')^2
		qui summarize rss
		local RSS = r(sum)


		loc r2 = 1 - (`RSS' / `TSS')


		    
		    if `r2' > scalar(`max_r2') {
		    	
			
			scalar `max_r2' = `r2'
			local new_U `var'
			
		    }
		    
		    // Here we determine which unit's values maximize the r2.
		    
		    drop `ym'
		    drop `cfp'
		    
		    // We get rid of these now as they've served their purpose.
		    
		}

		}

		
	local U `U' `new_U' // We add the newly selected unit to U.
	
	// and get rid of it from the predictors list.

	local predictors : list predictors - new_U
	
	    if `r2' < 0 {
	    	


        continue, break
    }

}


// we don't need to, but I put time first.

order `time', first

order `U', a(`treated_unit')

// Now we have the set of units sorted in the order of which maximizes the R2
// For each subsequent sub-DID estimation


* Now that we have them in order, we start with the first unit, estimate DID, and calculate
* the R2 statistic. We then in succession add the next best unit, and the next best one...
* until we have estimated N0 DID models. We wish to get the model which maximizes R2.

tempvar ymean rss tss

	egen `ymean' = rowmean(`U')


	constraint define 1 `ymean' = 1


	qui cnsreg `treated_unit' `ymean' if `time' < `interdate', constraints(1)

	qui predict cfdd
	
	
	
	* Calculate the mean of observed values
	qui summarize `treated_unit' if `time' < `interdate'
	local mean_observed = r(mean)

	* Calculate the Total Sum of Squares (TSS)
	qui generate double `tss' = (`treated_unit' - `mean_observed')^2 if `time' < `interdate'
	qui summarize `tss' if `time' < `interdate'
	local TSS = r(sum)

	* Calculate the Residual Sum of Squares (RSS)
	qui generate double `rss' = (`treated_unit' - cfdd)^2 if `time' < `interdate'
	qui summarize `rss' if `time' < `interdate'
	local RSS = r(sum)

	* Calculate and display R-squared
	loc r2dd = 1 - (`RSS' / `TSS')
	
	scalar DDr2 = `r2dd'
	

* Initialize an empty local to build the variable list

local varlist
local best_r2 = -1
local best_model = ""


* Loop through each variable in the list

qui foreach x of loc U {
	
	* Drop previous variables
	cap drop cf 
	cap drop ymean 
	cap drop tss
	cap drop rss
	* Add the current variable to the list
	local varlist `varlist' `x'


	egen ymean = rowmean(`varlist')


	constraint define 1 ymean = 1


	qui cnsreg `treated_unit' ymean if `time' < `interdate', constraints(1)

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
} // end of Forward Selection
di as text ""

qui drop ymean cf tss rss

cwf `cfframe'

qui frlink 1:1 `time', frame(`__reshape')

qui frget `treated_unit' `best_model' cfdd, from(`__reshape')

//di as txt "{hline}"

egen ymean = rowmean(`best_model')


// And estimate DID again.

* Define the constraint
constraint define 1 ymean = 1

* Run the constrained regression
qui cnsreg `treated_unit' ymean if `time' < `interdate', constraints(1)
loc RMSE = e(rmse)

qui predict cf
//qui predict se, stdp

* Calculate the mean of observed values
qui summarize `treated_unit' if `time' < `interdate'
local mean_observed = r(mean)

* Calculate the Total Sum of Squares (TSS)
qui generate double tss = (`treated_unit' - `mean_observed')^2 if `time' < `interdate'
qui summarize tss
local TSS = r(sum)

* Calculate the Residual Sum of Squares (RSS)
qui generate double rss = (`treated_unit' - cf)^2 if `time' < `interdate'
qui summarize rss
local RSS = r(sum)

* Calculate and display R-squared
scalar r2 = 1 - (`RSS' / `TSS')

// Now we calculate our ATT

qui su `treated_unit' if `time' >= `interdate'

loc yobs = r(mean)

* Here is the plot

if ("`gr1opts'" ~= "") {

if "`outlab'"=="" {
	
	loc outlab `outcome'
	
}


local fitname = "fit" + "`treatst'"

local fitname_cleaned = subinstr("`fitname'", " ", "", .)

// Define the string
local myString "`gr1opts'"

// Check if the word "name" is in the string
local contains_name = strpos("`myString'", "name")

// Return 1 if the string contains the word "name", otherwise return 0
local namecont = cond(`contains_name' > 0, 1, 0)

cap as `namecont'==1

if _rc != 0 {


local fitname = "fit" + "`treatst'"

local fitname_cleaned = subinstr("`fitname'", " ", "", .)

loc grname name(`fitname_cleaned', replace)	
	
	
}

twoway (line `treated_unit' `time', lcol(black) lwidth(medthick) lpat(solid)) ///
(line cf `time', lpat(dash) lwidth(thin) lcol(gs10)) ///
(line cfdd `time', lpat(shortdash) lwidth(medthick) lcol(gs6)), ///
yti("`treatst' `outlab'") ///
legend(order(1 "Observed" 2 "FDID" 3 "DD")) ///
xli(`interdate', lcol(gs6) lpat(--)) `grname' `gr1opts'

}



lab var cf "FDID `treatst'"
lab var `treated_unit'"Observed `treatst'"

frame __dfcopy {

local n = ustrregexra("`best_model'","\D"," ")

loc selected ""

local nwords :  word count `n'


// We see which units were selected

* Loop through each word in the macro `n`
forv i = 1/`nwords' {
    
    local current_word : word `i' of `n'

    * Extract the ith word from the macro `n`
    local units: display "`: label (`panel') `current_word''"
    
    local selected `selected' `: label (`panel') `current_word'',
    
    loc controls: display "`selected'"
    
    
}

frame `cfframe': qui note: The selected units are "`controls'"

}


// see prop 2.1 of Li


qui su `time' if `time' >=`interdate', d
tempname t2
scalar `t2' = r(N)

qui su `time' if `time' <`interdate', d
tempname t1
scalar `t1' = r(N)


g te = `treated_unit' - cf

g ddte =`treated_unit'-cfdd

lab var te "Pointwise Treatment Effect"
lab var ddte "Pointwise DD Treatment Effect"

qui g eventtime = `time'-`interdate'


qui g residsq = te^2 if eventtime <0

qui su if eventtime <0
scalar t1 = r(N)

qui su eventtime if eventtime>=0
scalar t2 = r(N)

qui su residsq, mean
scalar o1hat=(scalar(t2) / scalar(t1))*(r(mean))


qui su residsq, mean
scalar o2hat = (r(mean))

scalar ohat = sqrt(scalar(o1hat) + scalar(o2hat))

qui su te if `time' >= `interdate'


scalar ATT = r(mean)

qui su cf if eventtime >=0 

scalar pATT =  100*scalar(ATT)/r(mean)


scalar CILB = scalar(ATT) - (((invnormal(0.975) * scalar(ohat)))/sqrt(scalar(t2)))

scalar CIUB =  scalar(ATT) + (((invnormal(0.975) * scalar(ohat)))/sqrt(scalar(t2)))

qui su ddte if eventtime >= 0, mean
scalar DDATT = r(mean)


qui su cfdd if eventtime >=0 

scalar pATTDD =  100*scalar(DDATT)/r(mean)


if ("`gr2opts'" ~= "") {

	

twoway (connected te eventtime, connect(direct) msymbol(smdiamond)), ///
yti("Pointwise Treatment Effect") ///
yli(0, lpat(-)) xli(0, lwidth(vthin)) name(gap`treatst', replace) `gr2opts'
}


loc rmseround: di %9.5f `RMSE'
qui keep eventtime `time' `treated_unit' cf cfdd te ddte

qui mkmat *, mat(series)

scalar SE = scalar(ohat)/sqrt(scalar(t2))
scalar tstat = abs(scalar(ATT)/(scalar(SE)))

tempname my_matrix
matrix `my_matrix' = (scalar(ATT), scalar(pATT), scalar(SE), scalar(tstat), scalar(CILB), scalar(CIUB), scalar(r2), `rmseround')
matrix colnames `my_matrix' = ATT PATT SE t LB UB R2 RMSE
ereturn clear
matrix rownames `my_matrix' = Statistics

ereturn loc U "`controls'"
ereturn mat results = `my_matrix'

ereturn mat series = series

ereturn scalar T1 = `t1'
ereturn scalar T0= `t1'+1
ereturn scalar T2 = `t2'

ereturn scalar T= `t1'+`t2'


* Assuming you want to round to 3 decimal places for most values and 2 for others
* Round the scalar r2 to 3 decimal places
ereturn scalar r2 = round(scalar(r2), 0.001)

* Round the scalar DDr2 to 3 decimal places
ereturn scalar DDr2 = round(scalar(DDr2), 0.001)

* Round the scalar rmse to 3 decimal places
ereturn scalar rmse = round(`RMSE', 0.001)

* Assign the local N0U to the count of words in best_model and round if needed
local N0U: word count `best_model'
ereturn scalar N0U = `N0U'  // No rounding needed as this is a count

* Round the scalar DDATT to 3 decimal places
ereturn scalar DDATT = round(scalar(DDATT), 0.001)

* Round the scalar pDDATT to 3 decimal places
ereturn scalar pDDATT = round(scalar(pATTDD), 0.001)

* Round the scalar pATT to 3 decimal places
ereturn scalar pATT = round(scalar(pATT), 0.001)

* Round the scalar CILB to 3 decimal places
ereturn scalar CILB = round(scalar(CILB), 0.001)

* Round the scalar ATT to 3 decimal places
ereturn scalar ATT = round(scalar(ATT), 0.001)

* Round the scalar CIUB to 3 decimal places
ereturn scalar CIUB = round(scalar(CIUB), 0.001)

* Round the scalar se to 3 decimal places
ereturn scalar se = round(scalar(SE), 0.0001)

* Round the scalar tstat to 2 decimal places
ereturn scalar tstat = round(scalar(tstat), 0.01)


scalar p_value = 2 * (1 - normal(scalar(tstat)))

local tabletitle "Forward Difference-in-Differences"

if `ntr' == 1 {

di as text ""


di as res "`tabletitle'" "          " "T0 R2: " %9.3f scalar(r2) "     T0 RMSE: " %9.3f `RMSE'
di as text ""
di as text "{hline 13}{c TT}{hline 75}"


di as text %12s abbrev("`outcome'",12) " {c |}     ATT       Std. Err.      t       P>|t|    [95% Conf. Interval]" 
di as text "{hline 13}{c +}{hline 75}"


di as text %12s abbrev("`treatment'",12) " {c |} "%9.5f scalar(ATT) "   "  %9.5f scalar(SE) "  " %9.2f scalar(tstat) " " %9.3f scalar(p_value) "    " %9.5f scalar(CILB) "  " %9.5f scalar(CIUB)
di as text "{hline 13}{c BT}{hline 75}"

* Display the footer information
di as text "Treated Unit: `treatst'"
di as res "FDID selects `controls' as the optimal donors."
di as text "See Li (2024) for technical details."


}
end
