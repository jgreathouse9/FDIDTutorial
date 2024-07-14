*****************************************************
set more off
set varabbrev off
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

prog define fdid, rclass


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

loc time: disp "`r(timevar)'"

loc panel: disp "`r(panelvar)'"



marksample touse

_xtstrbal `panel' `time' `touse'

	syntax anything [if], ///
		TReated(varname) /// We need a treatment variable as 0 1
		[gr1opts(string asis)] [gr2opts(string asis)] ///
		[unitnames(varlist max=1 string)] [cfframename(string asis)]
		

		
   /* if unitname specified, grab the label here */
if "`unitnames'" != "" {
	
    
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
	
	di as err "Either specify -unitnames- or pre-assign your panel id with string IDs."
	
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

foreach x of loc trunit {

qui frame copy `originalframe' __dfcopy

frame __dfcopy {


loc trunitstr: display "`: label (`panel') `x''"

if "`cfframename'" == "" {
	
	cap frame drop fdid_cfframe
	
	loc defname fdid_cfframe
}

else {
	
	loc defname `cfframename'
}



numcheck, unit(`panel') ///
	time(`time') ///
	transform(`transform') ///
	depvar(`depvar') /// Routine 1
	treated(`treated') cfframe(`defname')


// Routine 2

treatprocess, time(`time') ///
	unit(`panel') ///
	treated(`treated')
	
loc trdate = e(interdate)


//Routine 3

data_org, unit(`panel') ///
	time(`time') ///
	depvar(`depvar') ///
	treated(`treated')
	
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
	trnum(`trunit') treatment(`treated') ///
	cfframe(`defname')
	
}

loc optimalcontrols: di e(selected)
mat resmat =e(ATTs)
return loc U =  "`optimalcontrols'"
return mat results = resmat
mat series = e(series)
return mat series= series
ereturn clear
}
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
	treated(varname) cfframe(string)
	
		
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
	
qui levelsof `unit' if `treated'==1, loc(treated_unit)

qui frame put `time' if `unit' == `treated_unit', into(`cfframe')
	
end


prog treatprocess, eclass
        
syntax, time(varname) unit(varname) treated(varname)

/*#########################################################

	* Section 1.2: Check Treatment Variable

	Before DD can be done, we need a treatment variable.
	
	
	The treatment enters at a given time and never leaves.
*########################################################*/


qui xtset
loc time_format: di r(tsfmt)

qui su `time' if `treated' ==1

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
		
		
		qui distinct `unit' if `treated' == 0
		
		loc dp_num = r(ndistinct) - 1
		
		cap as `dp_num' >= 2
		if _rc {
			
		di in red "You need at least 2 donors for every treated unit"
		exit 489
		}
		//di as res "{hline}"
/*
		di "{txt}{p 15 50 0} Treated Unit: {res}`adidtreat_lab' {p_end}"
		di as res "{hline}"
		di "{txt}{p 15 30 0} Control Units: {res}`dp_num' total donor pool units{p_end}"
		di as res "{hline}"
		di "{txt}{p 15 30 0} Specifically: {res}`controls'{p_end}"
*/

	}	

ereturn loc interdate = `interdate'


end




prog data_org, eclass
        
syntax, time(varname) depvar(varname) unit(varname) treated(varname)

/*#########################################################

	* Section 1.3: Reorganizing the Data into Matrix Form

	We need a wide dataset to do what we want.
*########################################################*/

qui su `unit' if `treated' ==1, mean

loc treat_id = `r(mean)'

qui su `time' if `treated' ==1

keep `unit' `time' `depvar'

local curframe = c(frame)

tempname __reshape

frame copy `curframe' __reshape

cwf __reshape

qui reshape wide `depvar', j(`unit') i(`time')

qui: tsset `time'
loc time_format: di r(tsfmt)


format `time' `time_format'

order `depvar'`treat_id', a(`time')


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
	treatment(string) [outlab(string asis)] cfframe(string)

qui ds

loc temp: word 1 of `r(varlist)'

loc time: disp "`temp'"

loc t: word 2 of `r(varlist)'

loc treated_unit: disp "`t'"

local strings `r(varlist)'


local trtime `treated_unit' `time'

local predictors: list strings- trtime

// Set U is a now empty set. It denotes the order of all units whose values,
// when added to DID maximize the pre-period r-squared.

local U

// We use the mean of the y control units as our predictor in DID regression.
// cfp= counterfactual predictions, used to calculate the R2/fit metrics.

tempvar ym cfp


// Here is the placeholder for max r2.
tempname max_r2


// Forward Selection Algorithm ...
di ""
    dis "Selecting the optimal donors via Forward Selection..."
    dis "----+--- 1 ---+--- 2 ---+--- 3 ---+--- 4 ---+--- 5"

qui while ("`predictors'" != "") {
    
scalar `max_r2' = 0

	foreach var of local predictors {
	
	// Drops these, as we need them for each R2 calculation
	
	cap drop rss
	cap drop tss

		quietly {
			
		// We take the mean of each element of set U and each new predictor.
			
		    
		 egen `ym' = rowmean(`U' `var')
		    
		 // The coefficient for the control average has to be 1.
		    
		 constraint 1 `ym' = 1
		    
		 // We use constrained OLS for this purpose.
		    
		 * 45 is the treatment date for HongKong, but this will be
		 * any date as specified by the treat variable input by the
		 * user.
		    
		cnsreg `treated_unit' `ym' if `time' < `interdate' , constraint(1)
		    
		// We predict our counterfactual
		    
		predict `cfp' if e(sample)
		
		// Now we calculate the pre-intervention R2 statistic.


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

}


// we don't need to, but I put time first.

order `time', first

order `U', a(`treated_unit')

// Now we have the set of units sorted in the order of which maximizes the R2
// For each subsequent sub-DID estimation


* Now that we have them in order, we start with the first unit, estimate DID, and calculate
* the R2 statistic. We then in succession add the next best unit, and the next best one...
* until we have estimated N0 DID models. We wish to get the model which maximizes R2.

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

qui frlink 1:1 `time', frame(__reshape)

qui frget `treated_unit' `best_model', from(__reshape)

//di as txt "{hline}"

egen ymean = rowmean(`best_model')


// And estimate DID again.

* Define the constraint
constraint define 1 ymean = 1

* Run the constrained regression
qui cnsreg `treated_unit' ymean if `time' < `interdate', constraints(1)
loc RMSE = e(rmse)

qui predict cf

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
if "`gr1opts'" != "" {

if "`outlab'"=="" {
	
	loc outlab `outcome'
	
}
	

twoway (connected `treated_unit' `time', connect(direct) msymbol(smdiamond)) (connected cf `time', lpat(--) msymbol(smsquare)), ///
yti(`treatst' `outlab') ///
legend(order(1 "Observed" 2 "FDID") pos(12)) ///
xli(`interdate', lcol(gs6) lpat(--)) `gr1opts'
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
}


// see prop 2.1 of Li


qui su `time' if `time' >=`interdate', d
tempname t2
scalar `t2' = r(N)

qui su `time' if `time' <`interdate', d
tempname t1
scalar `t1' = r(N)


g te = `treated_unit' - cf
lab var te "Pointwise Treatment Effect"
qui g eventtime = `time'-`interdate'

/*
cwf fdid_cfframe

g residsq = te^2 if eventtime <0
su residsq, mean
cls
scalar o1hat=(17 / 44)*(r(mean))
di `o1hat'
// 0.00010130090173830751

su residsq, mean
scalar o2hat = (r(mean))
di scalar(o2hat)
// 0.00026219056920503124

scalar ohat = sqrt(scalar(o1hat)+scalar(o2hat))
// 0.019065452287930093
di scalar(ohat)

di 0.025 - (((invnormal(0.975) * scalar(ohat)))/sqrt(17))

di 0.025 + (((invnormal(0.975) * scalar(ohat)))/sqrt(17))
*/


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

scalar CILB = scalar(ATT) - (((invnormal(0.975) * scalar(ohat)))/sqrt(scalar(t2)))

scalar CIUB =  scalar(ATT) + (((invnormal(0.975) * scalar(ohat)))/sqrt(scalar(t2)))

scalar SE = (invnormal(0.975) * scalar(ohat))/sqrt(scalar(t2))

if "`gr2opts'" != "" {

	

twoway (connected te eventtime, connect(direct) msymbol(smdiamond)), ///
yti("Pointwise Treatment Effect") ///
yli(0, lpat(-)) xli(0, lwidth(vthin)) `gr2opts'
}

loc rmseround: di %9.5f `RMSE'

qui mkmat *, mat(series)
scalar tstat = abs(scalar(ATT)/scalar(SE))

matrix my_matrix = (scalar(ATT), scalar(SE), scalar(tstat), scalar(CILB), scalar(CIUB), scalar(r2), `rmseround')
matrix colnames my_matrix = ATT SE t LB UB R2 RMSE

matrix rownames my_matrix = Statistics

ereturn loc selected "`controls'"
ereturn mat ATTs = my_matrix

ereturn mat series = series

keep `time' `treated_unit' cf te eventtime


frame drop __dfcopy
frame drop __reshape




scalar p_value = 2 * (1 - normal(scalar(tstat)))

local tabletitle "Forward Difference-in-Differences"

di as res "`tabletitle'"  "          " "T0 R2:" %9.3f scalar(r2) "     T0 RMSE:" %9.3f  `RMSE'
di as text "{hline 13}{c TT}{hline 77}"
di as text %12s abbrev("`outcome'",12) " {c |}      ATT        t           SE         [95% Conf. Interval]     p"
di as text "{hline 13}{c +}{hline 77}"
di as text %12s abbrev("`treatment'",12) " {c |} " as result %9.5f scalar(ATT) " "%9.3f scalar(tstat) "    " %9.5f scalar(SE) "    " %9.5f scalar(CILB) "   " %9.5f scalar(CIUB)  %9.3f scalar(p_value)     ""
di as text "{hline 13}{c BT}{hline 77}"
di as txt "Treated Unit: `treatst'"
di as res "FDID selects `controls' as the optimal donors."
di as text "See Li (2024) for technical details."


end
