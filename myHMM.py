import nltk
from nltk.corpus import brown

brown_tags_words=[]
for sent in brown.tagged_sents():
    brown_tags_words.append(("START","START"))
    brown_tags_words.extend([(tag[:2],word) for (word,tag) in sent])
    brown_tags_words.append(("END","END"))

cfd_tagwords=nltk.ConditionalFreqDist(brown_tags_words)
cpd_tagwords=nltk.ConditionalProbDist(cfd_tagwords,nltk.MLEProbDist)
print("The probability of an adjective (JJ) being 'new' is", cpd_tagwords["JJ"].prob("new"))
# 形容词（JJ）为“new”的概率是      prob 概率的简写
print("The probability of a verb (VB) being 'duck' is", cpd_tagwords["VB"].prob("duck"))

brown_tags=[tag for (tag,word) in brown_tags_words]
cfd_tags=nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))# count(t{i-1} , ti)
cpd_tags=nltk.ConditionalProbDist(cfd_tags,nltk.MLEProbDist)# P(ti | t{i-1})

#"I LIKE MILK"  tag：PP VB NN
prob_tagsequence=cpd_tags["START"].prob("PP")*cpd_tagwords["PP"].prob("I")* \
                 cpd_tags["PP"].prob("VB")*cpd_tagwords["VB"].prob("want")* \
				 cpd_tags["VB"].prob("NN")*cpd_tagwords["NN"].prob("milk")* \
				 cpd_tags["NN"].prob("END")

print("The probability of the tag sequence 'START PP VB NN END' for 'I like milk' is:", prob_tagsequence)

#维特比实现

distinct_tags=set(brown_tags)
sentence=["I","like","milk"]
sentlen=len(sentence)

viterbi=[]
first_viterbi={}
first_backpointer={}
backpointer=[]

for tag in distinct_tags:
	if tag=="START": continue
	first_viterbi[tag]=cpd_tags["START"].prob(tag)*cpd_tagwords[tag].prob(sentence[0])
	first_backpointer[tag]="START"
	
viterbi.append(first_viterbi)
backpointer.append(first_backpointer)

#currbest=max(first_viterbi,key=lambda tag:first_viterbi[tag])

for wordindex in range(1,len(sentence)):
	this_viterbi={}
	this_backpointer={}
	prev_viterbi=viterbi[-1]
	
	for tag in distinct_tags:
		if tag=="START":
			continue
		best_previous=max(prev_viterbi,key=lambda prevtag:prev_viterbi[prevtag]
								*cpd_tags[prevtag].prob(tag) \
		                                                 *cpd_tagwords[tag].prob(sentence[wordindex]))
		this_viterbi[tag]=prev_viterbi[best_previous] \
						*cpd_tags[best_previous].prob(tag) \
						*cpd_tagwords[tag].prob(sentence[wordindex])
		this_backpointer[tag]=best_previous
	#currbest=max(this_viterbi,key=lambda tag:this_viterbi[tag])
	viterbi.append(this_viterbi)
	backpointer.append(this_backpointer)
	
prev_viterbi=viterbi[-1]
best_previous=max(prev_viterbi,key=lambda prevtag:prev_viterbi[prevtag]*\
						  cpd_tags[prevtag].prob("END"))
prob_tagsequence=prev_viterbi[best_previous]*cpd_tags[best_previous].prob("END")

best_tagsequence = ["END", best_previous]
backpointer.reverse()
 
current_best_tag = best_previous
for bp in backpointer:
    best_tagsequence.append(bp[current_best_tag])
    current_best_tag = bp[current_best_tag]
 
best_tagsequence.reverse()
 
print("The sentence was:"),
for w in sentence: print(w)
print("\n")
print("The best tag sequence is:"),
for t in best_tagsequence: print(t),
print("\n")
print("The probability of the best tag sequence is:", prob_tagsequence)													
