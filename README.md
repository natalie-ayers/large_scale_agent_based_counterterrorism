# Agent-Based Model of Israeli Counterterrorism
  
This is a project developed for a Large-Scale Computing for the Social Sciences course at the University of Chicago in Fall 2021, taught by Jon Clindaniel. The large-scale components were run using resources from the University of Chicago Midway 2 Research Computing Center and an AWS Educate Classroom account. 

*Note:* the primary code used in the final output of this model is stored in the [model_lambda](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/tree/master/model_lambda) and [midway_promising_models](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/tree/master/midway_promising_models) folders. I describe the purpose of the remaining contents below as I discuss the progression of the project.

## Model Background  
  
The model I developed for this project was inspired by the 2012 paper "Moving Beyond Deterrence: The Effectiveness of Raising the Expected Utility of Abstaining from Terrorism in Israel", written by Laura Dugan and Erica Chenoweth. These researchers collected a dataset of Israeli actions towards Palestinians from 1987-2004 and classified each action as either repressive or conciliatory. They further broke down each of these categories into either targeted or indiscriminate. For example, targeted repressive action would be action taken directly against known violent offenders, such as jailing or deportation. Indiscriminate repression, by contrast, affects innocent and guilty members of a population, such as curfews, metal detectors, or random searches. Targeted conciliatory actions, on the other hand, are intended to motivate convicted terrorists away from violence and can include deradicalization programs or trading lenient sentences for information. Indiscriminate conciliatory actions benefit the entire population, aiming to decrease the base of support for terrorism by increasing the satisfaction of the general population. These are actions such as providing financial support, jobs training, or increasing representation in government.  
  
Dugan and Chenoweth use this data along with data on terrorist attacks over the same period to estimate the effect of different kinds of actions on the number of terrorist attacks using logistic regression models. They are specifically testing hypotheses they've developed using the following utility functions for acts of terror and refraining from violence:  
  
Dugan and Chenoweth hypothesize the expected utility of a terror attack for an individual i to be:  
    
<img src="https://render.githubusercontent.com/render/math?math=E(u_{terror})_i - p_i U(y_i - F_i) + (1 - p_i)U(y_i)">
  
"where p is the perceived probability of being punished, y is the anticipated benefits of perpetrating the act, and F is the perceived penalty for the act."   
  
Anticipated benefits can come both individually and for the group:  
* individual preferences / desires
* advancing movement
* improving status in group 
  
Their value for the group of abstaining from terrorism is:  
  
<img src="https://render.githubusercontent.com/render/math?math=E(u_{nonterror}) = qU(x + G) + (1 - q)U(x)">
  
where "q represents the probability of receiving rewards for abstaining from terrorism, x represents the value of the current situation (i.e., the status quo), and G represents the anticipated rewards of abstaining from terrorism." To clarify, "x directly relates to the grievances that motivate people to commit acts of terror."   
  
## Model Components
  
In order to focus on the large-scale components of this project, I'll begin with a simple model of counterterrorism actions and terrorist behavior. This will be based off of literature on counterterrorism and ABMs of terrorism and warfare, but it will be much simplified and not a direct mirror of any one model, though I will use Dugan and Chenoweth's initial utility functions most strongly as guides.     
  
To build the model, I've used the [mesa](https://mesa.readthedocs.io/en/master/) package, which is an open-source agent-based modeling framework for Python. While this does provide some limitations to the custom scaling I'm able to perform, at least without a re-write of the source code (creating an extension written to use Dask would be a useful project I hope to have the chance to work on), the complexity I'm able to achieve using Mesa made it an attractive choice, particularly as I plan to build off of this work for future research, and Mesa's other available packages will be useful for the extensions I have planned.   
    
Using Mesa, I built a model with 2 kinds of agents: Palestinians and the Israeli Government.   
  
### Palestinian Agent Specifications
  
Palestinian agents are assigned a satisfaction level from -100 to 100 randomly in their initial creation, and this satisfaction level will be what is changed in response to government action and what determines their status as either a combatant (those who commit terrorist actions), sympathetic (those who do not participate in violence but understand or are supportive of combatants), neutral (those who do not support violence but also do not work to oppose it), and anti-violence (those who would actively oppose violence and try to stop it).  
  
If an agent is a combatant, each round they will consider commiting a violent act with a predefined probability. If an attack is perpetrated, the entire world will be given a positive value for __violence_aftermath__, which is the collective recent memory of violence hanging over the community. A combatant who commits an act of terrorism will be killed and removed from the model.  
  
All agents will also react to Israeli government policy and their surroundings each turn:  
  * First, if the grid they are in is overcrowded (determined as having more than 30 other individuals in the same grid square), their satisfaction will decrease. If the grid is not overcrowded, their satisfaction will decrease by a single point due to being subject to a system they view as oppressive.  
  * If there is a positive __violence_aftermath__ value - ie, if an attack has taken place within the last 10 turns, or more if there were multiple attacks in a row - then combatants will gain satisfaction due to their support of the attack.  
  * Depending on the Israeli government's action, the agent will either lose or gain satisfaction:
    * If the government performs an Indiscriminate Conciliatory action, the agent will gain <img src="https://render.githubusercontent.com/render/math?math=3^{sust-conc}"> satsifaction, a value which will increase exponentially as more conciliatory actions are performed. These actions are things like opening up movement between the Palestinian territories and other countries or increasing funding to the entire territories.  
    * If the government performs a Targeted Conciliatory action, then if an agent is in the same grid square where the action was performed, their satisfaction increases by 2. This does not increase exponentially because Dugan and Chenoweth don't find much support for targeted conciliatory actions' success. Examples of this kind of action are refurbishing a school or creating a food program in a specific neighborhood.  
    * If the government performs an Indiscriminate Repressive action, then an agent's satisfaction decreases by <img src="https://render.githubusercontent.com/render/math?math=3^{sust-reor}">, a value which increases exponentially with sustained repressive actions. Examples of this kind of behavior are increasingly restrictive borders or sanctions.  
    * If the government performs a Targeted Repressive action, then if an agent is a combatant or sympathetic to combatants, their satisfaction will decrease by 3, otherwise there will be no effect.  
  
Overall, in a single turn, every agent will react to their surroundings and government policy, and combatants will consider and potentially perform a violent attack.  
  
### Government Agent Specifications

There is only a single Israeli government agent created which acts once per turn as well. This government agent has as characteristics a 'policy', which determines whether it is more inclined to perform repressive actions, conciliatory actions, or has no preference. It also has a 'reactive_lvl', which determines the level of force with which the government responds to a violent attack, and for how long this government response is affected by an attack. Finally, the government has a 'status' characteristic, which is the action the government performs that turn.  
  
When the government acts, it will consider the state of the world - particularly, whether there was recently a violent attack - and this along with its policy preference will determine the probabilities for its choice of actions. Each turn, the government can either do nothing, perform an Indiscriminate Conciliatory action, perform a Targeted Conciliatory action, perform an Indiscriminate Repressive action, or perform a Targeted Repressive action. If it performs a targeted action, it will select a random grid cell to perform the action in, which will be the only one affected by the action. If it performs either a conciliatory or repressive action, the overall model's sustained repression or sustained conciliation scores will be affected - for example, if it performs a repressive action and in the previous turn also performed a repressive action, the __sust_repr__ score will increase from 1 to 2.  
  
### Model 
  
When the model is run, an initial number of Palestinian agents are created and randomly assigned satisfaction levels, and an Israeli government agent is also created. A _DataCollector_ is also set up to capture information about the run of each model, which is used to create visualizations and report on the performance of some of the more noteable models automatically, and which is processed and stored for any future analysis.  
  
The model is run for a specified number of turns, and each turn every Palestinian agent adjusts their status and the government chooses whether to act. After the actions, any deaths are removed from the model, and a small, random number of Palestinian agents are added to the model as new births (according to the World Population Report, the Palestinian population is growing by 2.4% per year, so I've used a population birth rate of 3%). 
  
### Parameters to Test 
  
There are a number of configuration options that are being tested with this model, in the style of a Cross-Validation, where every combination is tested. The parameters with different values being considered are:  
* Probability of commiting a violent act: 0.0001, 0.0005, 0.001, 0.003, 0.005, 0.008, 0.01  
  * This is the probability with which a combatant will commit an act of violence  
* Government policy: NONE, CONC, REPR  
  * This is the government's proclivity for a certain type of action towards the Palestinian people - it can be thought of as a moderate government (NONE), a liberal government (CONC, ie conciliatory), or a conservative government (REPR, ie repressive)  
* Reactive Level: high, mid-high, mid-low, low, none  
  * This is the level at which the government reacts to recent violence with repressive measures. It determines the probabilities which the government uses to select either no action, a repressive action (targeted or indiscriminate), or a conciliatory action (targeted or indiscriminate). For example, if the reactive level is high, then any violent attack will influence their decisions for a much longer time after the attack, whereas a low or no reactive level will lead them to only have a higher probability of repressive action for 2 turns, or to have no increased probability of repressive action. We can think of this as a combination of the government's political persuasion (conservative vs liberal) and the amount of pressure they're receiving from their citizens to react.  
* Discontent: high, mid, low  
  * This is the initial level of discontent of the Palestinian citizens. It determines at what level of satisfaction they transition between combatants, sympathetic, neutral, and anti-violence  
* Starting population: 200, 300, 400, 500, 600, 800, 1000  
  * This is the starting population of Palestinians for the model
  * The starting population also determines what grid size is selected for the model: the West Bank and Gaza are some of the most densely populated countries in the world (in 2020, 798 people per square km, which was the 13th most densely populated country according to the World Bank), so I test using fairly dense initial conditions, and if the density rises above a given level, the satisfaction of the Palestinians decreases even further.  
* Steps: 200, 300, 500, 700, 900  
  * This is the number of steps of the model to run through, which can be thought of as the number of weeks or months over which the model runs.   
    

## Running Model Locally

I first confirmed the model's ability to simulate 300 steps locally, by selecting only one value for each of the variable parameters. This code is located in [model.py](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/model.py) and run [local_model.ipynb](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/local_model.ipynb).  
  
Given the embarassingly parallel nature of performing these simulations for multiple combinations of parameters (11025 to be precise), this model provides a very strong use case for scaling out the calculations. Scaling out on GPUs would allow for far greater concurrency than CPU processing, and as I do not have a large amount of data to transfer to each GPU thread (we only need to transfer the initial parameters and the code), there should also not be slowdowns with data transfers. The primary concern with GPU processing would be the performance of each individual GPU thread as compared to a CPU core: GPUs are not capable of high-performance tasks at speed, and when this model is run with many agents, the GPUs will need to either perform each simulation on a single thread (which will be slow), or distribute the work of a single simulation among threads (which will require data transfer and also slow the process). Ultimately, however, the question of GPU performance was not one I was able to test, as the custom code I used with the _mesa_ package was not conducive to the limited functionality of GPUs without major updates to the source code. Therefore, I've opted to test running these models on different configurations of CPUs: running on multiple cores on Midway using MPI, and running with Lambda functions on AWS.  
  
## Testing on Midway CPUs

After initial single-run tests, I tested this process on 4 cores with only 10 runs each, selected from the beginning of the list of full parameters to test each core was assigned. The results of these runs are stored in the [four-cores-ten-each](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/tree/master/four-cores-ten-each) folder in the Git repo for this project, including the visualizations produced to display the model outcomes for the model with the highest number of total attacks (in this case 477 attacks, obtained when the parameters were set at: prob_violence: 0.0001, policy: 'NONE', reactive_lvl: 'high', discontent: 'high', starting_population: 400, steps: 500)) and for the model with the highest ending population (in this case 0, as all agents ended up killed, obtained when the parameters were set at: prob_violence: 0.0001, policy: 'NONE', reactive_lvl: 'high', discontent: 'high', starting_population: 400, steps: 500)). The same model was used in both in this case because in all parameter combinations tested, the final population was 0 - an indication that these parameters are not sustainable in the context of the model. The visualization of the number of attacks and number of agents demonstrates this problem (note that the number of attacks are cumulative):  
  
<img src="/four-cores-ten-each/attacks-and-agents-max_attacks-477.png">    
  
The other visualizations are less useful given this result, so I next attempted to run the entire set of visualizations across 10 midway cores. The initial results of this attempt are stored in the folder [ten-cores-all-runs](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/tree/master/ten-cores-all-runs), but unfortunately as the models became more accurate (that is, as all of the agents did not die in each model, leaving more actions to take), and as this occurred for models with higher initial populations and step counts, the processing time increased from 1-5 seconds for the initial models to 200-300 seconds for many models, including a few which took over 1000 seconds. Consequently, after approximately 2/3 of the parameter tests were completed, which took 30 minutes, Midway canceled the remainder of my job.  
  
Given this result, I reduced the number of parameter combinations from 11025 to 5400, but I still hoped to reduce the required processing before returning to Midway. Thus, I next considered running these models using Lambda functions on AWS.  
  
## Testing on AWS with Lambda
  
I decided to try running on Lambda rather than creating multiple EC2 instances because, while some of my models are a bit more performance-intensive (as noted by a few taking over 1000 seconds on the midway runs), for the most part, they require relatively little in the way of computation, and the number of models I was running (5400) made sense for running using 3000 available Lambda instances at one time.   
  
In order to set up my model to run with Lambda functions, I frst used Docker to create a zip file to use as a Lambda Layer, which was required to utilize Numpy and Mesa in my functions. I used a [Dockerfile](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/model_lambda/counterterrorism.Dockerfile) with Amazon Linux 2 as the base image to build a container in which I installed numpy and Mesa, which I then packaged into a .zip file and uploaded to AWS as a Lambda Layer, according to [this Towards Data Science post](https://towardsdatascience.com/how-to-install-python-packages-for-aws-lambda-layer-74e193c76a91). I then created a Lambda function, CounterterrorismModel, copies of which I've included in [model_lambda.ipynb](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/model_lambda/model_lambda.ipynb) and as a stand-alone file [lambda_function.py](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/model_lambda/lambda_function.py). I also included a copy of [model_lambda.py](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/model_lambda/model_lambda.py) in the Lambda environment to use in my Lambda function. I made some modifications to the function to input and output json values, which meant that I could not collect and pickle full dataframes with my results. Instead, I used these Lambda runs to collect summary statistics, which I will use below to select the most promising models to run and store the full results.  
  
The full steps I used are contained in [model_lambda.ipynb](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/model_lambda/model_lambda.ipynb): first, testing locally, then invoking the Lambda function with a single combination of parameters, before invoking with 4 parameters, and then finally running the full set of parameters broken into lists. Since we are unable to use Pywren, I used the _ThreadPoolExecutor_ to attempt 3000 concurrent functions, however upon running into runtime errors, I realized our accounts (or at least my account) have concurrency limits of only 50:  
  
<img src="/model_lambda/lambda_concurrency_limits.png">    

I thus adjusted to running 1000 models at a time, looping through my overall list of 5400 parameter options in chunks of 1000, and setting the _ThreadPoolExecutor_ to a max of 1000 despite the limit of 50 (an online forum suggested that the concurrency limits may be raised upon heavier use of the account, so I've left it at 1000 in the hope that the limit would be raised).  

Note that a few of these models resulted in memory errors on the Lambda instance, so I've also included a _try.. except_ statement to collect a record of which sets of parameters produced the memory errors, as these are likely of interest for research due to their business.  
   
The results of these runs were collected as a list (pickled and stored as [full_fiftyfourhundredd_results_lambda.p](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/model_lambda/full_fiftyfourhundredd_results_lambda.p)), turned into a DataFrame, and processed to determine which parameter combinations gave the most promising results, as determined by a realistic number of surviving agents. I've stored these final models as [promising_params.p](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/model_lambda/promising_params.p). These final models will be what I use to produce the results from this initial, large-scale exploration.  
  
## Final Model Performance
  
With the pickled [promising_params.p](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/midway_promising_models/promising_params.p), I'll adjust the mpi script I used originally to only use these combinations of parameters, while still producing full dataframe results from the models, which we were unable to do with Lambda. The script run for these final sets of models is in [midway_promising_models](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/tree/master/midway_promising_models), as [promising_models_mpi.py](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/midway_promising_models/promising_models_mpi.py) and [promising_models_mpi.sbatch](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/midway_promising_models/promising_models_mpi.sbatch). The results are stored in [promising_models_mpi.out](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/midway_promising_models/promising_models_mpi.out).  
  
Since we're now transferring data between ranks rather than performing easy calculations on each, the main change I've made is in calculating the distribution of the parameters on the root 0 node, then using *.scatter()* to provide each node its respective parameters.  
  
The model completed running 492 parameter combinations on 18 Midway FDR nodes in 51.25 seconds, a very strong runtime.  
  
On running the model with this final set of parameters, we are left with multiple DataFrames which were gathered, concatenated, and pickled on the root 0 node:  
* [full_models_df.pkl](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/midway_promising_models/full_models_df.pkl): this contains the progression of attacks and number of agents over the course of all of the models (identified by their parameter dictionaries)  
* [full_deaths_df.pkl](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/midway_promising_models/full_deaths_df.pkl): this contains the number of deaths pers step for each model  
* [govt_status_df.pkl](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/midway_promising_models/govt_status_df.pkl): this contains the action of the government at each point of each model  
* [govt_status_cum_df.pkl](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/midway_promising_models/govt_status_cum_df.pkl): this contains the cumulative count of each kind of action committed by the government  
* [palestinian_stati_df.pkl](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/midway_promising_models/palestinian_stati_df.pkl): this contains the number of agents of a given status at each step of every model  
  
A collection of additional statistics for every model is included in the [promising_models_mpi.out](https://github.com/lsc4ss-a21/final-project-agent_based_counterterrorism/blob/master/midway_promising_models/promising_models_mpi.out) file, in the second-to-last line under "final results from all runs". This can be used to create a dataframe for further study if desired, as was done from the Lambda function results.  
  
Finally, a series of pre-defined visualizations were created for the model which resulted in the highest final population and the model in which the most attacks occurred. 

*Highest Number of Attacks*   
  
For this run, the model which had the highest number of attacks had initial parameters:  
* prob_violence: 0.001 
* govt_policy: CONC
* reactive_lvl: none
* discontent: mid
* starting_population: 400
* steps: 400
  
This model resulted in a final population of 5,361 and a total number of attacks of 9,822.  
  
We can visualize the cumulative number of attacks considered against the number of Palestinian agents active at any time and see that the number of attacks continues to rise sharply despite a decline in population due to deaths from committing these attacks:    
  
<img src="/midway_promising_models/attacks-and-agents-max_attacks-9822.png">    
  
We can also consider the cumulative types of Israeli government actions which occurred over the same period, and see that Targeted Conciliatory actions were the most common when an action was selected. Dugan and Chenoweth found that targeted conciliatory actions were moderately to not effective at preventing attacks, and it is likely in this case that the effect of overcrowding took precedence on the unhappiness of the Palestinian agents.  
  
<img src="/midway_promising_models/govt_status_cumulative-max_attacks-9822.png">    
  
When we examine Palestinian sentiment over time more closely, we do see that a large portion of the population does indeed become sympathetic to violence by the conclusion of the model:  
  
<img src="/midway_promising_models/palestinian_stati_bar-max_attacks-9822.png">  
  
We can view this breakdown of sentiments by grid square with our final set of graphs, which display the dominant final sentiment in each grid square and the proportion of the grid square's population which shared that sentiment, respectively:  

<img src="/midway_promising_models/dominant_sentiments-max_attacks-9822.png">  
<img src="/midway_promising_models/perc_dominant_sentiments-max_attacks-9822.png">  

While 'Anti-Violence' was the dominant sentiment throughout, the second visualization demonstrates that for some grid squares (aka neighborhoods), this anti-violence sentiment only represented 44% or 45% of the population, leaving a significant amount available still to be more sympathetic to violence.  
  
A model such as this, which has an extremely high number of attacks, can be studied more closely to determine if there are any surprising factors driving the number of attacks which may provide insights for terrorist attack numbers in the real world.  
  
*Highest Final Population* 
  
The second model for which visualizations were created automatically, the model with the highest final population, had initial parameters:  

* prob_violence: 0.008
* govt_policy: CONC
* reactive_lvl: none
* discontent: low
* starting_population: 400 
* steps: 400
  
This model resulted in a final population of 8,145 and a total number of attacks of 5,289.  

We can visualize the cumulative number of attacks over time and the number of Palestinian agents active, as before, and see that in this case, while attacks do rise with population, the population growth far outpaces the growth in attacks. This is contributed to both by the lower probability of committing a violent act and the lower initial discontent (which assigns fewer Palestinian agents to Combatant status at the beginning of the model).  
  
<img src="/midway_promising_models/attacks-and-agents-highest_pop-8145.png">     
  
When we consider Israeli government actions, we see that, while Targeted Conciliatory actions are still the most frequent, they are nearly matched by Indiscriminate Conciliatory actions, which Dugan and Chenoweth found to be the most effective over time at preventing attack:  
  
<img src="/midway_promising_models/govt_status_cumulative-highest_pop-8145.png">     
  
The Palestinian status progression over time supports this view, with very few Palestinian agents demonstrating violent statuses by the end of the model and the vast majority in the Anti-Violence status:  
  
<img src="/midway_promising_models/palestinian_stati_bar-highest_pop-8145.png">     
  
This is reinforced by considering the geographic breakdown of Palestinian agent statuses, which shows all grid cells having a majority Anti-Violence, and the percentages who do have Anti-Violent statuses as high as the 70s-80s:  
  
<img src="/midway_promising_models/dominant_sentiments-highest_pop-8145.png">  
<img src="/midway_promising_models/perc_dominant_sentiments-highest_pop-8145.png">  

Overall, especially with a model more finely tuned to real-world parameters and behavior, this result which produces a highly satisfied Palestinian population can also provide indications of triggers of satisfaction or peace which may be helpful for further study.  

## Conclusion
  
This project is just the beginning of what can be accomplished using an an agent-based model to investigate the effects of counterterrorism actions. Though beyond the scope of this project, a next step would be to use the best results from our large-scale computation of a number of possible model parameters to compare against the true terrorism and conciliatory/repressive action datasets. We could additionally add geographic and other demographic parameters to the model to more closely mirror reality, though it would be helpful to perform large-scale tests such as the ones above to determine at what point adding additional input parameters may stop improving the model's performance. As mentioned above, creating a version of the Mesa code to be paralellized with Dask, and potentially creating a version that could be run on GPUs, would greatly benefit the process of testing models to determine which most closely mirror reality.  
  
Ultimately, these models can be used to provide insights to direct future research into causes of terrorist actions and policies to mitigate terrorism. While these models, unless significantly improved and tested, should not always be taken as a true reflection of reality (given the complexities of our world which cannot currently be captured in computational models), they can help support or even develop hypotheses concerning terrorism which can then be tested with an analysis of real-world data. By allowing researchers to model behavior from the expected to the extreme, new patterns of behavior and causal reactions may appear which can spur additional research, insight, and policy change. 
  






# Sources  
  
https://www.geeksforgeeks.org/random-choices-method-in-python/
https://towardsdatascience.com/introduction-to-mesa-agent-based-modeling-in-python-bcb0596e1c9a
https://stackoverflow.com/questions/62821720/deleting-agent-in-mesa
https://moonbooks.org/Articles/How-to-visualize-plot-a-numpy-array-in-python-using-seaborn-/
https://stackoverflow.com/questions/36227475/heatmap-like-plot-but-for-categorical-variables-in-seaborn
https://github.com/projectmesa/mesa/issues/784
https://data.worldbank.org/indicator/EN.POP.DNST?locations=PS&most_recent_value_desc=true
https://stackoverflow.com/questions/798854/all-combinations-of-a-list-of-lists
https://worldpopulationreview.com/countries/palestine-population
https://towardsdatascience.com/how-to-install-python-packages-for-aws-lambda-layer-74e193c76a91
https://docs.docker.com/get-started/
https://docs.aws.amazon.com/lambda/latest/dg/python-package.html
https://journals.sagepub.com/doi/10.1177/0003122412450573
https://stackoverflow.com/questions/29815129/pandas-dataframe-to-list-of-dictionaries
https://www.researchgate.net/publication/303692784_An_Agent_Based_Approach_for_Understanding_Complex_Terrorism_Behaviors
https://gist.github.com/VictorNS69/1c952045825eac1b5e4d9fc84ad9d384