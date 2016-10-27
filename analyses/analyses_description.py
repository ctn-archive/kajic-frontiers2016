descriptions = [
    {'Analysis': 'avg_response_similarity',
     'Description': 'Average response similarity within' +
     'vs. between cue clusters',
     'Group1': 'within cluster',
     'Group2': 'across cluster'},

    {'Analysis': 'avg_response_similarity_clean',
     'Description': 'Average response similarity within' +
     'vs. between cue clusters',
     'Group1': 'within cluster',
     'Group2': 'across cluster'},

    {'Analysis': 'permute_pcs_similarity',
     'Description': 'Controls for confounds in cue' +
     'assignment by assigning random primary cues',
     'Group1': 'typeA (within cue -> across cue)',
     'Group2': 'typeB (across cue -> within cue)'},

    {'Analysis': 'across_cue_similarity',
     'Description': 'Test for breaks between clusters ' +
     'by comparing adjacent and non-adjacent responses ' +
     'with different primary cues',
     'Group1': 'adjacent responses',
     'Group2': 'non-adjacent responses'},

    {'Analysis': 'within_cue_similarity',
     'Description': 'Test for sequential dependence of ' +
     'responses with same primary cue',
     'Group1': 'adjacent responses',
     'Group2': 'non-adjacent responses'},

    {'Analysis': 'rate_with_distance',
     'Description': 'Testing the rate of change with' +
     'distance to the final answer',
     'Group1': 'correct answer',
     'Group2': 'wrong answer'},

    {'Analysis': 'same_cue_percentage',
     'Description': 'Response pairs with the same primary cue',
     'Group1': 'baseline',
     'Group2': 'actual'}
    ]
