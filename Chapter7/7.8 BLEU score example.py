


from nltk.translate.bleu_score import sentence_bleu

# Score on exactly similar text
reference = [['this','is','a','sentence','check']]
candidate = ['this','is','a','sentence' ,'check']

# Individual scores
print("\nResults for exactly similar sentences:")
print("Individual 1-gram score :",sentence_bleu(reference,candidate,weights=(1,0,0,0)))
print("Individual 2-gram score :",sentence_bleu(reference,candidate,weights=(0,1,0,0)))
print("Individual 3-gram score :",sentence_bleu(reference,candidate,weights=(0,0,1,0)))
print("Individual 4-gram score :",sentence_bleu(reference,candidate,weights=(0,0,0,1)))

# Cumulative N-gram score
print("Cumulative 4-gram score :",sentence_bleu(reference,candidate,weights=(0.25,0.25,0.25,0.25)))


# Score on almost similar text
reference_2 = [['this','is','a','sentence','check']]
candidate_2 = ['this','is','a','sentence' ,'fragment']

# Cumulative N-gram score
print("\nResults for almost similar sentences:")
score_val = sentence_bleu(reference_2,candidate_2,weights=(0.25,0.25,0.25,0.25))
print("Cumulative 4-gram score :",round(score_val,3))



