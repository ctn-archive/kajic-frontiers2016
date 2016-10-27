# Explanation of analysis methods

These methods have been developed and explained in Smith et al 2013. They have
been used to analyse human response data and test various hypotheses related to
the search process. Here we present only a subset of those methods.

## Running the script

The script is executed from the command line (use `chmod +x analyser.py` to set the permission for running if not already set). This script depends on the existence of the word vector space in the `../../data/wordvectors/' directory. If vectors are not available, the script will complain it can't find it. 


To get the statistics from the paper:

```
./analyser.py fbinary_google_symmetric_simulations
```

The `fbinary_google_symmetric_simulations.csv` file has been downloaded with
this repository, but it can be generated with the `RAT_Response_Filtering`
notebook.


## Analyses


Every response in the chain of responses is assigned a primary cue from the set
of three primary cues. All responses with the same primary cue are regarded as
within-cue cluster, whereas responses with different primary cues are regarded
as across (or between) cue cluster.

For clarity, each of the methods below will be accompanied with a short
description of it applies to the following simple example:

R1 -> R2 -> R3 -> R4 -> R5 -> R6
C1 -> C1 -> C1 -> C2 -> C2 -> C3

Where R_n stands for the n-response in a sequence of responses, and C_n for the
assigned primary cue. Primary cues are assigned based on the shortest distance
in some semantic space (e.g. WAS, LSA).

### 1. avg_response_similarity

Average response similarity, within vs. between cue clusters.
Sort all adjacent response pairs into within-cue pairs and across-cue pairs:

| Within cue | Across cue |
|------------|------------|
| R1->R2     | R3->R4     |
| R2->R3     | R5->R6     |
| R4->R5     |            |

### 2. permute_pcs_similarity

Permutation test to control for primary cue assignment. Responses that are next
to each other are more likely to be more similar to a common cue, and by chance
some adjacent pairs will be more similar than others. If primary cues don't
exist, then having a sequence of permuted cues should yield the same result.
For every sequence of primary cues another sequence of primary cues is assigned
from a different problem (shuffled). Then, type A responses are those who are
assigned the same primary cue for the real cues, and assigned different primary
cue for the shuffled ones. Type B is the opposite.

```
            R1 -> R2 -> R3 -> R4 -> R5 -> R6

real:       C1 -> C1 -> C1 -> C2 -> C2 -> C3

shuffle:    C2 -> C2 -> C1 -> C1 -> C3 -> C3
```

| Type A     | Type B     |
|------------|------------|
| R1->R2     | R3->R4     |
| R2->R3     | R5->R6     |


If there are no primary cues, then the similarity between Type A and Type B is
equal.

### 3. avg_response_similarity_adjacent_responses
Similar to analysis 1. but performed on cleaned-up responses. In first case all
direct responses from participants are considered, but in this case if
participants repeated the word or mentioned a word not in the corpus (TASA/WAS)
then a new pair was formed from responses surrounding that one.


### 4. across_cue_similarity
Testing whether people make breaks when switching from one cluster to another.
If there is a break, then the first response in a new cluster should not be
influenced by the preceding response. To test this, adjacent and non-adjacent
responses with different primary cues are compared.

From the above given example:
R1 -> R2 -> R3 -> R4 -> R5 -> R6
C1 -> C1 -> C1 -> C2 -> C2 -> C3

| Adjacent across cue | Non-adjacent across cue |
|---------------------|------------------------ |
|      R3->R4         |       R1->R4            |
|      R5->R6         |                         |
|                     |       R2->R4            |
|                     |       R4->R6            |


### 5. within_cue_similarity
Similar to 4. but restricted to responses within cluster. This is to test
whether there are intracluster dependencies. If all responses within a cluster
are equally likely to be sampled from the same cue (independently of other
responses within a cluster) than we can expect no difference in the similarity between
adjacent responses and non-adjacent responses with the same cue. If, however,
there is difference, this would point to sequential dependence.

### 6. same_cue_percentage

Baseline probability as the expected probability given frequencies of each
primary cue. Actual probability as counting frequencies.

### 7. rate_with_distance
Compute similarity betwen the final answer and every response in the response chain
(up to 10 responses before the answer was given). There are two classes:

- rate_correct

- rate wrong

Rate correct gathers response chains where the correct answer was given, and
rate wrong gathers responses where no correct answer was given (timeouts).

