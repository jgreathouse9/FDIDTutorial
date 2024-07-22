qui {
clear *
set varabbrev off
cls
u "hcw.dta", clear
}
fdid gdp, tr(treat) unitnames(state)  gr1opts(scheme(sj) name(econint, replace))
fdid gdp if inrange(time,1,45), tr(polint) unitnames(state) gr1opts(name(polint, replace) scheme(sj))

qui u smoking, clear

fdid cigsale, tr(treated) unitnames(state) gr1opts(scheme(sj) name(p99, replace))

qui u basque, clear

fdid gdpcap, tr(treat) gr1opts(scheme(sj) name(eta, replace))

qui u barcelona, clear

fdid indexed_price, tr(treat) unitnames(fullname) gr1opts(name(barcelona, replace) scheme(sj))



