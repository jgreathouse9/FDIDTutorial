cap log close
log using fdidlog.log, replace

clear *

* Users Need sdid_event: https://github.com/Daniel-Pailanir/sdid/tree/main/sdid_event

net from "https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main"

net install fdid, replace

net get fdid, replace

u smoking, clear

fdid cigsale, treated(treat) unitnames(state)

mkf newframe

cwf newframe

svmat e(series), names(col)

cls
twoway (connected cigsale3 year, mcolor(black) msize(small) msymbol(smcircle) lcolor(black) lwidth(medthick)) ///
	(connected cfdd3 year, mcolor(gs11) msize(small) msymbol(smsquare) lcolor(gs11) lpattern(solid) lwidth(thin)) ///
	(connected ymeandid3 year, mcolor(gs11) msize(small) msymbol(smtriangle) lcolor(gs11) lwidth(thin)), ///
	ylabel(#10, grid glwidth(vthin) glcolor(gs8%20) glpattern(dash)) ///
	xline(1989, lwidth(medium) lpattern(solid) lcolor(black)) ///
	xlabel(#10, grid glwidth(vthin) glcolor(gs8%20) glpattern(dash)) ///
	legend(cols(1) ///
	position(9) ///
	order(1 "California" 2 "DID Prediction" 3 "DID y{subscript:N{subscript:co}} Mean") ///
	region(fcolor(none) lcolor(none)) ring(0)) ///
	scheme(sj) ///
	graphregion(fcolor(white) lcolor(white) ifcolor(white) ilcolor(white)) ///
	plotregion(fcolor(white) lcolor(white) ifcolor(white) ilcolor(white)) ///
	name(did, replace) yti(Cigarette Sales) ti("All Controls")

twoway (connected cigsale3 year, mcolor(black) msize(small) msymbol(smcircle) lcolor(black) lwidth(medthick)) ///
	(connected cf3 year, mcolor(gs11) msize(small) msymbol(smsquare) lcolor(gs11) lpattern(solid) lwidth(thin)) ///
	(connected ymeanfdid year, mcolor(gs11) msize(small) msymbol(smtriangle) lcolor(gs11) lwidth(thin)), ///
	ylabel(#10, grid glwidth(vthin) glcolor(gs8%20) glpattern(dash)) ///
	xline(1989, lwidth(medium) lpattern(solid) lcolor(black)) ///
	xlabel(#10, grid glwidth(vthin) glcolor(gs8%20) glpattern(dash)) ///
	legend(cols(1) ///
	position(9) ///
	order(1 "California" 2 "FDID Prediction" 3 "FDID y{subscript:N{subscript:co}} Mean") ///
	region(fcolor(none) lcolor(none)) ring(0)) ///
	scheme(sj) ///
	graphregion(fcolor(white) lcolor(white) ifcolor(white) ilcolor(white)) ///
	plotregion(fcolor(white) lcolor(white) ifcolor(white) ilcolor(white)) name(fdid, replace) ti("FDID Controls")
	
	graph combine did fdid, ///
	xsize(9) ///
	ysize(4.5) //
	
	graph export "FDIDP99.png", as(png) name("Graph") replace

qui log close

