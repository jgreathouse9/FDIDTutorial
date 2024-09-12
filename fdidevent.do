// requires sdid_event, fdid

clear * // We can leave comments in our code like this
cls

use /// or we can leave comments like
"https://github.com/jgreathouse9/FDIDTutorial/raw/main/smoking.dta" // this

qui fdid cigsale, tr(treat) unitnames(state)


ereturn list
loc post = e(T2)

mkf newframe
cwf newframe
svmat e(didframe), names(col)
xtset id year
qui sdid_event cigsale id year treat, method(did) brep(1000) placebo(all)

	
local row `= rowsof(e(H))' 
	


mat res = e(H)[2..`row',1..4]

mkf newframe2
cwf newframe2

svmat res

rename (res1 res2 res3 res4) (eff se lb ub)


gen eventtime = _n - 1 if !missing(eff)
replace eventtime = `post' - _n if _n > `post' & !missing(eff)
sort eventtime

        twoway (rcap  lb ub eventtime, lcolor(black)) ///
	(scatter eff eventtime, mc(blue) ms(d)), ///
	legend(off) ///
	title(sdid_event) xtitle(Time to Event) ///
	ytitle(Pointwise Effect) ///
	yline(0,lc(red) lp(-)) xline(0, lc(black) lp(solid))
