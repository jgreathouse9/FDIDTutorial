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
{bf:fdid} {hline 2} Estimates Forward Difference-in-Differences. 


{marker syntax}{...}
{title:Syntax}

{p 8 17 2}
{cmdab:fdid}
[{depvar}]
{ifin},
{opt tr:eated}({varname}) [{opt gr1opts}({it:string}) {opt gr2opts}({it:string}) {cmd: unitnames}({it:{varname}}) {cmd: cfframename}({it:string})]

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
along with confidence intervals. Note that the dependent variable must be numeric, 
non-missing and non-constant. The {opt tr:eated} variable must be a dummy
variable equal to one when the unit is treated, else 0.
{cmd: fdid} requires the data to be {cmd: xtset} and balanced.


{marker options}{...}
{title:Options}

{dlgtab:Main}

{phang}
{opt gr1opts}: If specified, edits the display of the observed versus predicted plot. It accepts the string literally as is. For example,
{cmd: fdid gdp, tr(treat) unitnames(state) gr1opts(scheme(sj) name(hcw, replace))} returns a plot formatted in the most recent version of the Stata Journal's scheme, with the plot being named hcw. If not specified, no plot is created.

{phang}
{opt gr2opts}: If specified, edits the display of the treatment effect plot. It accepts the string literally as is. For example,
{cmd: fdid gdp, tr(treat) unitnames(state) gr2opts(scheme(sj) name(hcwte, replace))} returns a plot formatted in the most recent version of the Stata Journal's scheme, with the plot being named hcwte. If not specified, no plot is created.

{phang}
{opt unitnames}: {cmd: fdid} requires the panel variable to have value labels, where a number is indexed to a string (i.e., 1="australia"). If the panel variable already has them, no error is returned.
However, if the panel does not come with value labels,
then the user must specify the string variable they wish to use as the panel variable's value labels.
Note that each string value pair must be uniquely identified.

{phang}
{opt cfframename}: The name of the data frame containing the counterfactual, treatment effect, and observed values.
If nothing is specified, the name by default is {it:fdid_cfframe}.

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
{cmd:fdid} returns the following results, which 
can be displayed by typing {cmd: return list} after 
{cmd: fdid} is finished (also see {help return}).  

{p 8 8 2}
{cmd: r(results):}{p_end}
{p 10 10 2}
A matrix containing the ATT, standard error, t-statistic, and the upper and lower bounds of the 95% Confidence Interval.
It also has the R-squared and Root-Mean-Squared Error statistics for the pre-intervention period.

{p 8 8 2}
{cmd: r(series):}{p_end}
{p 10 10 2}
A matrix containing the time, observed values, counterfactual values, pointwise treatment effect, and event time.

{p 8 8 2}
{cmd: r(U):}{p_end}
{p 10 10 2}
A macro containing the list of selected units chosen by the forward selection algorithm.

{synoptline}

{title:Frames}
{p 4 8 2}
{cmd:fdid} returns a frame for the user's convenience.

{p 4 8 2}
{cmd:fdid_cfframe}: Contains the outcome vector for the treated unit, the counterfactual vector, the time period, and the pointwise treatment effect.


{marker examples}{...}
{title:Examples}

{phang}

Users may install {cmd:fdid} like {cmd:net install fdid, from("https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main") replace}.

To obtain the data files, we do: {cmd: net get fdid, from("https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main") replace}.


// Replicating HCW2012

{cmd:u hcw, clear}

{cmd:fdid gdp, tr(treat) unitnames(state) gr1opts(scheme(sj) name(hcw, replace))}

{phang}

// Replicating Abadie and Gardeazabaal 2003

{cmd:u "agbasque.dta", clear}


{cmd:fdid gdpcap, tr(treat) gr1opts(scheme(sj) name(ag, replace))}


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


