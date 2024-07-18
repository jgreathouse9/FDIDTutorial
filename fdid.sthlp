{smcl}
{* *! version 1.0.0  14jul2024}{...}
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

Firstly, {cmd: fdid} requires the data to be {cmd: xtset} and strongly balanced without gaps.


{synopt:{opt depvar}}The outcome of interest. {p_end}
{synopt:{opt treated}}Our treatment variable. It must be a 0 1 dummy. One may not use factor notation. {cmd:fdid} supports settings
where we have multiple treated units with different treatment dates.
Note that we assume the treatment is fully absorbed (once treated, always treated). {p_end}
{synoptline}
{p2colreset}{...}
{p 4 6 2}

{marker description}{...}
{title:Description}

{pstd}
{cmd:fdid} estimates the average treatment effect on the treated unit
as proposed by {browse "https://doi.org/10.1287/mksc.2022.0212": Li (2024)}. {cmd:fdid} selects the optimal pool
of control units via forward selection. Below, we describe the selection algorithm for {cmd:fdid}. {p_end}

{pstd} We begin with no selected control units. We first select the most optimal control unit by using linear regression to predict the
pre-intervention outcome vector of the treated unit using each of the control units in a single-predictor OLS model.
Of these controls, we select the control unit that has the highest R-squared statistic among them. Then, we select the next
optimal control unit by taking the average of the first selected control and each of the units in the now N0-1 control group.
Of these N0-1 models, we select the two-unit model with the highest R-squared statistic. We then add this next-best control unit
to the next model, and continue like so until we have as many DID models as we have control units. In the process, we put the unit
with the highest R2 at the front of the list of the new control group. {p_end}

{pstd} Next, we select the optimal DID model. We do this by iteratively adding new control units.
Since the algorithm has already sorted each new selected control by it's R-squared value, we store the value of the current highest R-squared statistic.
For example, suppose the first model, using the first selected unit, has an R-squared of 50, and when we add the next selected unit the R-squared goes to 60. We would prefer this model
using the two units since its R-squared statistic is higher than the one unit model. If another unit increases the R-Squared statistic, we add it to the optimal control group. We continue like so, using whatever DID model of the total N0 DID models has the highest R-squared statsitic.
Naturally, the DID model with the highest R-squared statistic becomes the optimal DID model because this means that it has selected the optimal control group.  {p_end}

{pstd} After selecting the optimal control group, {cmd:fdid} calculates the treatment effect
along with confidence intervals using the inference procedure as described in {browse "https://doi.org/10.1287/mksc.2022.0212": Li (2024)}. When there are many treated units,
we take the expectation of the event-time ATTs (that is, the average of all treatment effects for each unit in its respective treated periods). We then pool the variances of each ATT and calculate the standard error of the ATT. Using this, we
calculate t-statistics and 95% confidence intervals.  {p_end}


{marker options}{...}
{title:Options}

{dlgtab:Main}

{phang}
{opt gr1opts}: If specified, edits the display options of the observed versus predicted plot. It accepts the string literally as is. For example,
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
{opt cfframename}: The user-supplied name of the data frame containing the observed values, counterfactual, treatment effect,
standard error of the treatment effect, and event time.
If nothing is specified, the name by default is {it:fdid_cfframe}, with the numeric panel id as a suffix.
For example, if the treated unit is California and its panel id is 3, the name is fdid_cfframe3.

{synoptline}

{title:Saved Results}

{p 4 8 2}
{cmd:fdid} returns different results depending on the setup of treatment.
When only one unit is treated, {cmd:fdid} returns the following results which 
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

{p 4 8 2}
If there are many treated units, {cmd:fdid} returns the following:

{p 8 8 2}
{cmd: r(results):}{p_end}
{p 10 10 2}
A matrix containing the results from the table, including the ATT, standard error, t-statistic, and the upper and lower bounds of the 95% Confidence Interval.

{synoptline}

{title:Frames}
{p 4 8 2}
{cmd:fdid} returns a frame for the user's convenience.

{p 10 10 2}
{cmd:fdid_cfframe}: Contains the outcome vector for the treated unit, the counterfactual vector, the time period, and the pointwise treatment effect.
If many units are treated, then we have one frame per treated unit, with the names of the selected control units in the {help notes}.


{marker examples}{...}
{title:Examples}

{phang}

Users may install {cmd:fdid} like {stata "net install fdid, from(https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main) replace"}.

To obtain the data files, we do: {stata "net get fdid, from(https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main) replace"}.


Replicating HCW2012

{stata "u hcw, clear"}

{stata "fdid gdp, tr(treat) unitnames(state)"}

{phang}

For a more extended walkthrough, see the {browse "https://github.com/jgreathouse9/FDIDTutorial/blob/main/StataVignette.md":vignette}.


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


