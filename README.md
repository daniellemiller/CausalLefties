# CausalLefties
<br />
This project aimed to check the causal effect of tennis player's dominant hand on their results.<br />

### Data:
The data we used was downloaded from JeffSackmann repositories:

- https://github.com/JeffSackmann/tennis_atp
- https://github.com/JeffSackmann/tennis_wta

### Requirments:
The code runs in Python 3.6 and the following libraries are essential for it to work:
sklearn and tqdm

### Content:
There are several folders in the repository:<br />
data - the download tennis_atp dataset. A combined file called full_data.csv is included.
<br /><br />
women_data - the download tennis_wta dataset. A combined file called full_data.csv is included.
<br /><br />
scripts - contans the data.py which handles the initial digestion of the inputs.
<br /><br />
src - the code itself. potential_outcomes.matching.py addresees the matching approach<br />
	  and potential_outcomes.causal_utils.py the S-learner approach.<br />
outputs - results of our runs on the data folder:<br />
* MATCH...csv - results of matching algorithm for different scoring and treatment assignment functions
* SLEARNER.csv - results of s-learner
* ITE_by... - Figures used in the final paper, ITE results from several perspectives.
* enrichments_PVALS - Figure presenting the results from Fisher's exact test
* number_players and dominant_hand - Figures used in the paper to explain the data.
* data_for_learning and mapper - partial results saved to reduce operations
<br /><br />
women_outputs - results of our runs on the data folder:<br />
* MATCH...csv - results of matching algorithm for different scoring and treatment assignment functions
* SLEARNER.csv - results of s-learner
* ITE_by... - Figures used in the final paper, ITE results from several perspectives.
* enrichments_PVALS - Figure presenting the results from Fisher's exact test
