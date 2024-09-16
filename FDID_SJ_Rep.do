cap log close
log using fdidlog.log, replace

clear *

net from "https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main"
net install fdid, replace

net get fdid, replace


u smoking, clear

fdid cigsale, tr(treated) unitnames(state)

// Making the First Plot

mkf resultframe
cwf resultframe

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
graph export "FDIDP99.png", as(png) name("Graph") replace

qui log close

