{smcl}
{* *! version 1.2.2  15may2018}{...}
{findalias asfradohelp}{...}
{vieweralsosee "" "--"}{...}
{vieweralsosee "[R] help" "help help"}{...}
{viewerjumpto "Syntax" "examplehelpfile##syntax"}{...}
{viewerjumpto "Description" "examplehelpfile##description"}{...}
{viewerjumpto "Options" "examplehelpfile##options"}{...}
{viewerjumpto "Remarks" "examplehelpfile##remarks"}{...}
{viewerjumpto "Examples" "examplehelpfile##examples"}{...}
{title:Forward Difference in Differences}

{phang}
{bf:fdid} {hline 2} Estimates the Forward Difference-in-Differences method. 


{marker syntax}{...}
{title:Syntax}

{p 8 17 2}
{cmdab:fdid}
[{depvar}]
{ifin},
{opt tr:eated}({varname}) [{opt gr1opts}({it:string}) {cmd: unitnames}({it:{varname}})]

{synoptset 20 tabbed}{...}
{synoptline}
{syntab:Requirements}
{synopt:{opt depvar}}The outcome of interest. {p_end}
{synopt:{opt treated}}Our treatment variable. {p_end}
{synoptline}
{p2colreset}{...}
{p 4 6 2}



{marker description}{...}
{title:Description}

{pstd}
{cmd:fdid} estimates the average treatment effect on the treated
for settings where we have one treated unit and multiple control units.
It uses an iterative forward selection algorithm to select the optimal
control group. After selecting the optimal control group, {cmd:fdid} calculates the treatment effect
along with confidence intervals. Note that the dependent variable must be a numeric, non-missing and non-constant.
The {opt tr:eated} variable must be a dummy variable equal to one when the unit is treated, else 0. {cmd: fdid} requires the data to be {cmd: xtset} and balanced.


{marker options}{...}
{title:Options}

{dlgtab:Main}

{phang}
{opt gr1opts}: edits the display of the observed versus predicted plot. It accepts the string literally as is. For example,
{cmd: fdid gdp, tr(treat) unitnames(state) gr1opts(scheme(sj) name(hcw, replace))} returns a plot formatted in the most recent version of the Stata Journal's scheme, with the plot being named hcw.

{phang}
{opt unitnames}: {cmd: fdid} presumes the panel variable has value labels, where a number is indexed to a string. However, if this is not the case, then the user simply specifies the string variable they wish to use as the panel variable's names. Note that each string number pair must be uniquely identified.

{marker remarks}{...}
{title:Remarks}

{pstd}
For theoretical justification and more details on the method overall, see ({browse "https://doi.org/10.1287/mksc.2022.0212":the original paper}).
Note, that in theory we may extend {cmd: fdid} to situations where there are many treated units at different points in time. However, at the moment
{cmd: fdid} only supports settings where there's one treated unit. As a workaround, assuming the treatment period begins at the same time for all units,
the user may iteratively estimate treatment effects for each treated unit (omitting the other treated units from the control group) and then
take the average of their treatment effects and confidence intervals.

{synoptline}

{title:Saved Results}

{p 4 8 2}
{cmd:fdid} ereturns the following results, which 
can be displayed by typing {cmd: ereturn list} after 
{cmd: fdid} is finished (also see {help ereturn}).  

{p 8 8 2}
{cmd: e(ATTs):}{p_end}
{p 10 10 2}
A matrix that contains the ATT, the upper and lower bounds of the 95% Confidence Interval for the ATT, and the R-squared statistic for the pre-intervention period.

{p 8 8 2}
{cmd: e(selected) :}{p_end}
{p 10 10 2}
A macro containing the list of selected units chosen by the forward selection algorithm.

{synoptline}

{title:Frames}
{p 4 8 2}
{cmd:fdid} returns frames for the user's convenience.

{p 4 8 2}
{cmd:cfframe}: Contains the outcome vector for the treated unit, the counterfactual vector, the time period, and the pointwise treatment effect.


{marker examples}{...}
{title:Examples}

{phang}


// Replicating HCW2012

u hcw, clear

qui fdid gdp, tr(treat) unitnames(state) gr1opts(scheme(sj) name(hcw, replace))

{phang}

// Replicating Abadie and Gardeazabaal 2003

u "agbasque.dta", clear


qui fdid gdpcap, tr(treat) gr1opts(scheme(sj) name(ag, replace))


{hline}

{title:References}
{p 4 8 2}

Li, K. T. (2024). Frontiers: A simple forward difference-in-differences method. Marketing Science, 43(2), 267-279. {browse "https://doi.org/10.1287/mksc.2022.0212"}

Abadie, A., & Gardeazabal, J. (2003). The economic costs of conflict: A case study of the basque country. Am. Econ. Rev., 93(1), 113-132. {browse "https://doi.org/10.1257/000282803321455188 "}

Hsiao, C., Steve Ching, H., & Ki Wan, S. (2012). A panel data approach for program evaluation: Measuring the benefits of political and economic integration of hong kong with mainland china. 
J. Appl. Econom., 27(5), 705-740. {browse "https://doi.org/10.1002/jae.1230"}

{title:Contact}

Jared Greathouse, Georgia State University -- {browse "https://jgreathouse9.github.io/":personal website}
Emails--
Student: {browse "jgreathouse3@student.gsu.edu"}
Personal: {browse "j.greathouse200@gmail.com"}

Email me with questions, comments, suggestions or bug reports.
 

{hline}

