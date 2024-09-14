{smcl}
{* *! version 1.0.0  22jul2024}{...}
{viewerjumpto "Syntax" "fdid##syntax"}{...}
{viewerjumpto "Description" "examplehelpfile##description"}{...}
{viewerjumpto "Options" "fdid##options"}{...}
{viewerjumpto "Examples" "fdid##examples"}{...}
{title:Forward Difference in Differences}

{phang}
{bf:fdid} {hline 2} Estimates Forward Difference-in-Differences.


{marker linkspdf}{...}
{title:Links to Online Documentation}

        For a more extended walkthrough, see the {browse "https://github.com/jgreathouse9/FDIDTutorial/blob/main/StataVignette.md":vignette}.

{pstd}


{marker syntax}{...}
{title:Syntax}

{p 8 17 2}
{cmdab:fdid}
[{depvar}]
{ifin},
{opt tr:eated}({varname}) [{opt gr1opts}({it:string}) {opt gr2opts}({it:string}) {cmd: unitnames}({it:{varname}}) {opt placebo}]


{synoptset 20 tabbed}{...}

{dlgtab:Requirements}

{p2colreset}{...}

{p 4 4 2}{helpb xtset} {it:panelvar} {it:timevar} must be used to declare a strongly balanced panel dataset without gaps. {cmd:sdid_event} is also {browse "https://github.com/DiegoCiccia/sdid/tree/main/sdid_event#github":required}.

{p 4 4 2}
{depvar} The numeric outcome of interest.{p_end}

{p 4 4 2}
{cmd: treated} Our treatment. It must be a 0 1 dummy, equal to 1 on and at the treatment date and afterwards. One may NOT use factor notation. {cmd:fdid} supports settings
where we have multiple treated units with different treatment dates.
Note that we assume the treatment is fully absorbed (once treated, always treated). {p_end}
{synoptline}
{p2colreset}{...}
{p 4 6 2}

{marker description}{...}
{title:Description}

{p 4 4 2}
{cmd:fdid} estimates the average treatment effect on the treated unit
as proposed by {browse "https://doi.org/10.1287/mksc.2022.0212": Li (2024)}. {cmd:fdid} selects the optimal pool
of control units via forward selection. Below, we describe the selection algorithm for {cmd:fdid}. {p_end}

{p 4 4 2}
We begin with no selected control units. We first select the most optimal control unit by using linear regression to predict the
pre-intervention outcome vector of the treated unit using each of the control units in a single-predictor OLS model.
Of these controls, we select the control unit that has the highest R-squared statistic among them. Then, we select the next
optimal control unit by taking the average of the first selected control and each of the units in the now N0-1 control group.
Of these N0-1 models, we select the two-unit model with the highest R-squared statistic. We then add this next-best control unit
to the next model, and continue like so until we have as many DID models as we have control units. In the process, we put the unit
with the highest R2 at the front of the list of the new control group. {p_end}

{p 4 4 2}
Next, we select the optimal DID model. We do this by iteratively adding new control units.
Since the algorithm has already sorted each new selected control by it's R-squared value, we store the value of the current highest R-squared statistic.
For example, suppose the first model, using the first selected unit, has an R-squared of 50, and when we add the next selected unit the R-squared goes to 60. We would prefer this model
using the two units since its R-squared statistic is higher than the one unit model. If another unit increases the R-Squared statistic, we add it to the optimal control group. We continue like so, using whatever DID model of the total N0 DID models has the highest R-squared statsitic.
Naturally, the DID model with the highest R-squared statistic becomes the optimal DID model because this means that it has selected the optimal control group.  {p_end}

{p 4 4 2}
After selecting the optimal control group, {cmd:fdid} calculates the treatment effect
along with confidence intervals using the inference procedure as described in {browse "https://doi.org/10.1287/mksc.2022.0212": Li (2024)}. When there are many treated units,
we take the expectation of the event-time ATTs across cohorts.  {p_end}


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
{opt unitnames}: {cmd: fdid} requires the panel variable to have value {help label}s, where a number is indexed to a string (i.e., 1="australia"). If the panel variable already has them, no error is returned.
However, if the panel does not come with value labels,
then the user must specify the string variable they wish to use as the panel variable's value labels.
Note that each string value pair must be uniquely identified.

{phang}
{opt placebo}: If specified, reports the placebo standard error for the ATT from {browse "https://github.com/DiegoCiccia/sdid/tree/main/sdid_event#github":sdid_event}.

{synoptline}

{marker results}{...}
{title:Stored Results}

{pstd}
{cmd:fdid} stores the following in e():

{synoptset 20 tabbed}{...}
{p2col 5 20 24 2: Scalars}{p_end}
{synopt:{cmd:e(T1)}}number of pre-intervention periods.{p_end}
{synopt:{cmd:e(T0)}}treatment point.{p_end}
{synopt:{cmd:e(T2)}}number of post-intervention periods{p_end}
{synopt:{cmd:e(T)}}number of time periods.{p_end}
{synopt:{cmd:e(N0)}}Number of controls.{p_end}
{synopt:{cmd:e(N0U)}}Number of controls selected by FDID.{p_end}

{synoptset 20 tabbed}{...}
{p2col 5 20 24 2: Macros}{p_end}
{synopt:{cmd:e(U)}}list of selected controls selected by FDID method (singe treated unit only).{p_end}
{synopt:{cmd:e(depvar)}}dependent variable.{p_end}
{synopt:{cmd:e(properties)}}list of properties.{p_end}

{synoptset 20 tabbed}{...}
{p2col 5 20 24 2: Matrices}{p_end}
{synopt:{cmd:e(series)}}A matrix containing the time, observed values, counterfactual values, pointwise treatment effect, event time, and means for the all controls and FDID controls.{p_end}
{synopt:{cmd:e(results)}}Table of results {p_end}
{synopt:{cmd:e(b)}}Coefficients.{p_end}
{synopt:{cmd:e(V)}}Covariance matrix.{p_end}
{synopt:{cmd:e(dyneff)}}Dynamic Treatment Effects.{p_end}

{marker examples}{...}
{title:Examples}

{phang}

Users may install {cmd:fdid} like {stata "net install fdid, from(https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main) replace"}.

To obtain the data files, we do: {stata "net get fdid, from(https://raw.githubusercontent.com/jgreathouse9/FDIDTutorial/main) replace"}.


Replicating HCW2012

{stata "u hcw, clear"}

{stata "fdid gdp, tr(treat) unitnames(state)"}


{phang}


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


