# Part-of-Speech Tagging using Probabilistic Models

### Objective:
Predict the parts of speech for each word in the given sentence using the probability models - Simple, Viterbi(HMM) and Markov Chain Monte Carlo(MCMC).

### Formulation of each problem:

##### Training the dataset - 
To get the probabilities for the word and their parts of speech in the required formats for implementing the probabilistic models, the following four dictionaries are created -

emi_ident    - Contains the overall probabilities of the different parts of speech

word_list    - Contains the words of the train file as the keys and each word has a dictionary of probabilities of different parts of speech for that word -> P(Word/POS)

trans_ident  - It has the transition probabilities of the parts of speech to the next part of speech occuring in the sentence

trans_ident2 - It has the transition probabilities of the parts of speech to the two consecutive parts of speech occuring in the sentence


##### Defining Simple Probability model -
In the simple model, for each word in the sentence the values -> (emission probability of word and part-of-speech) * (Overall probability of part-of-speech) is compared and the part-of-speech having the max value is returned.
Since there are some words in the test file which are not there in the trained set, the part-of-speech having the maximum overall probability is returned.

##### Defining Viterbi(HMM) Proboability model - 
In the Viterbi model, two initial dictionaries having all parts of speech as their keys with their values being list of n zeros where n is the number of words in the sentence.

The two initial dictionaries are -

hmm : Having the max probability value 

pos : Having the part-of-speech having the max probability value

The hmm dictionary first word values are then updated with P(POS)*P(Word/POS).
After this, we loop through the sentence from the second word and the second loop for all the part-of-speech. In these loops, the max probability value is found under each by calculating their transitions and hmm and pos dictionaries are updated.
Upon updating the dictionaries, we start backtracking. The max probability value and the corresponding part-of-speech is found for the last word. Using the pos dictionary, we then backtrack and iteratively update the predicted part-of-speech for each word which is then returned as output.

##### Defining Monte Carlo Markov Chain(MCMC) model - 
In the MCMC model, we first loop through the entire sentence and for each word, we get the probabilities for the corresponding part-of-speech and it is updated in a list. If no probability is available for a part-of-speech, we take the minimum probability and multiply it with a very low value( here 1e-15 ) to define its probability.
The probabilities are then normalised and their sum would be 1. These probability states are updated for every part-of-speech for every word in the prob list-of-lists

After this step, the main step of randomizing samples is implemented. An iteration number is chosen and an empty dictionary having all part-of-speech is initialized to zero. 
We then use the following function -

random.choices(all_pos,weights=prob[i],k=1)

where 

all_pos : All parts of speech

weights : Probabilities of the part-of-speech for each word which is derived from the prob list-of-lists 

k : Number of choices chosen

In the above function, we choose the part-of-speech based on their corresponding probabilities for the word. Upon implementing this, the part-of-speech chosen the maximum number of times after n iterations( here 4500) is iteratively updated in a list which is then returned as output.

##### Defining the log of joint probability P(S,W) - 

For Simple Model,

Log base 10 (emission probability of word and part-of-speech) + Log base 10 (Overall probability of part-of-speech) is taken for each word. 
For the test words not in list of train words, Log base 10 (random value) + Log base 10 (Overall probability of part-of-speech) is considered where the random value is made sure not to exceed 0.5.
Finally, all the values are summed up and returned as the log value of joint probability of simple model.

For Viterbi Model,

We first initialize our value to the Log base 10 (overall probability of the first part-of-speech).
Then, we add the values the same way as done in the simple model. In addition to that, we also add the Log base 10 (transition probability of part-of-speech to the next part-of-speech).
Finally, all the values are summed up and returned as the log value of joint probability of viterbi model.

For the MCMC Model,

We first initialize our value to the Log base 10 (overall probability of the first part-of-speech).
Then, we add the values the same way as done in the viterbi model. In addition to that, we also add the Log base 10 (transition probability of part-of-speech to the two consecutive part-of-speech).
Finally, all the values are summed up and returned as the log value of joint probability of mcmc model. 

### Description of the program:
The program first takes in the train dataset and defined the probabilities of emission and transition for the words and the different parts of speech. The code then computes the different defined probability models - Simple, Viterbi and MCMC models and gets computed as described above.
In addition to this, the program also returns the log of joint probability of P(S,W) for all the three probabilistic models.

### Problems faced and Design decisions:
##### Problems faced -
One of the main problems we faced was the implementation of the models and getting the values for the test words not defined in the train dataset. Initially, we randomly assigned a fixed value but since we were not getting good accuracy, a different approach was taken.
We randomized for all the probabilities we had to consider as described in the code to normalize our values and this made it more accurate. 

##### Design decisions - 
The designs for the Simple and Viterbi were good and did not face any problems but that was not the case for the MCMC model. 

Initially, we got a sample value and from this sample, a gen_sample function was defined which would take this sample and iteratively generate n samples based on their log probability for each word and the different parts of speech. After getting their log probability, the probabilities were cumulatively summed up and each value was compared with a generated random value.

If the cumulative value was greater than the random value, the sample part-of-speech with this new one and the process was repeated. The issues with this was the time taken and due to wrong implementation, the accuracy was not more than 5%.
We then implemented the above algorithm thus reaching ~91% for words and ~36% for sentences.
