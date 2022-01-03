# Defining Packages
import random
import math
import numpy as np
import random
 
# Defining Main Class
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def __init__(self):
        self.emi_ident = {'freq':0}
        self.trans_ident = {}
        self.trans_ident2 = {}
        self.word_list = {}

    def posterior(self, model, sentence, label):
        if model == "Simple":
            return self.posterior_sample(sentence,label)
        elif model == "HMM":
            return self.posterior_hmm_viterbi(sentence,label)
        elif model == "Complex":
            return self.posterior_complex(sentence,label)
        else:
            print("Unknown algo!")

    # Do the training!
    def train(self, data):
        for i in data:
            for l in range(len(i[0])):
                # Updating the Emission Probabilities for word and part-of-speech
                if i[0][l] in self.word_list.keys():
                    self.word_list[i[0][l]]['freq'] += 1
                else:
                    self.word_list[i[0][l]] = {'freq': 1}

                if i[1][l] in self.word_list[i[0][l]].keys(): 
                    self.word_list[i[0][l]][i[1][l]] += 1
                else:
                    self.word_list[i[0][l]][i[1][l]] = 1
                
                # Updating transition probabilities for part-of-speech to next part-of-speech
                if i[1][l] not in self.trans_ident.keys() and l<len(i[0])-1:
                    self.trans_ident[i[1][l]] = {'freq':0}
                    self.trans_ident2[i[1][l]] = {'freq':0}

                if l<len(i[1])-1:
                    if i[1][l+1] in self.trans_ident[i[1][l]]: 
                        self.trans_ident[i[1][l]][i[1][l+1]] += 1
                        self.trans_ident2[i[1][l]][i[1][l+1]]['freq'] += 1
                    else:
                        self.trans_ident[i[1][l]][i[1][l+1]] = 1
                        self.trans_ident2[i[1][l]][i[1][l+1]] = {'freq':0}
                    self.trans_ident[i[1][l]]['freq'] += 1
                    self.trans_ident2[i[1][l]]['freq'] += 1
                
                # Updating transition probabilities for part-of-speech to two consecutive part-of-speech
                if l<len(i[1])-2:
                    if i[1][l+2] in self.trans_ident2[i[1][l]][i[1][l+1]]: 
                        self.trans_ident2[i[1][l]][i[1][l+1]][i[1][l+2]] += 1
                    else:
                        self.trans_ident2[i[1][l]][i[1][l+1]][i[1][l+2]] = 1
                
                # Updating overall probabilities of all part-of-speech
                if i[1][l] in self.emi_ident.keys(): 
                    self.emi_ident[i[1][l]] += 1
                else:
                    self.emi_ident[i[1][l]] = 1
                self.emi_ident['freq'] += 1

        # Updating the Emission values to probabilities
        for i in self.word_list:
            max_num = ['na',-1]
            for j in self.word_list[i]:
                if j!='freq':
                    self.word_list[i][j] = self.word_list[i][j]/self.emi_ident[j]
                    if max_num[1] < self.word_list[i][j]:
                        max_num = [j,self.word_list[i][j]]
            self.word_list[i]['max'] = max_num
        
        # Updating the Transition 1 values to probabilities
        for i in self.trans_ident:
            max_num1 = ['na',-1]
            check = list(self.emi_ident.keys())
            for j in self.trans_ident[i]:
                if j!='freq':
                    self.trans_ident[i][j] = self.trans_ident[i][j]/self.trans_ident[i]['freq']
                    if max_num1[1] < self.trans_ident[i][j]:
                        max_num1 = [j,self.trans_ident[i][j]]
                if j in check:
                    check.pop(check.index(j))
            for k in check:
                self.trans_ident[i][k] = 0
            self.trans_ident[i]['max'] = max_num1
        
        # Updating the Transition 2 values to probabilities
        for i in self.trans_ident2:
            check2 = [i for i in self.emi_ident.keys() if i not in ['freq','max']]
            for j in self.trans_ident2[i]:
                if j!='freq':
                    max_num3 = ['na',-1]
                    check = [i for i in self.emi_ident.keys() if i not in ['freq','max']]
                    for k in self.trans_ident2[i][j]:
                        if k!='freq':
                            if self.trans_ident2[i][j]['freq'] > 0:
                                self.trans_ident2[i][j][k] = self.trans_ident2[i][j][k]/self.trans_ident2[i][j]['freq']
                            if max_num3[1] < self.trans_ident2[i][j][k]:
                                max_num3 = [k,self.trans_ident2[i][j][k]]
                        if k in check:
                            check.pop(check.index(k))
                    for kk in check:
                        self.trans_ident2[i][j][kk] = 0
                    self.trans_ident2[i][j]['max'] = max_num3
                if j in check2:
                    check2.pop(check2.index(j))
            for jj in check2:
                self.trans_ident2[i][jj] = {jjj:0 for jjj in self.emi_ident.keys() if jjj not in ['freq','max']}
        
        # Updating the Overall values to probabilities
        max_num2 = ['na',-1]
        for i in self.emi_ident:
            if i!='freq':
                self.emi_ident[i] = self.emi_ident[i]/self.emi_ident['freq']
                if max_num2[1] < self.emi_ident[i]:
                    max_num2 = [i,self.emi_ident[i]]
        self.emi_ident['max'] = max_num2
    
    # Defining Simple Model 
    def simplified(self, sentence):
        pred_words = []
        self.simple_model = 0
        for i in sentence:
            max_tag = ['na',-1]
            # If Test word not in train dataset, the part-of-speech having max probability value is appended
            if i not in self.word_list.keys():
                pred_words.append(self.emi_ident['max'][0])
            else:
                for j in self.word_list[i]:
                    if j not in ['max','freq']:
                        val = self.word_list[i][j]*self.emi_ident[j]
                        if max_tag[1] < val:
                            max_tag = [j,val]
                # If max_tag does not get updated, the part-of-speech having max probability value is appended
                if max_tag[0] == 'na':
                    pred_words.append(self.emi_ident['max'][0])
                else:
                    pred_words.append(max_tag[0])
        return pred_words

    # Defining log of joint probability P(S,W) for Simple model
    def posterior_sample(self,sentence,label):
        val = 0
        for i in range(len(sentence)):
            if sentence[i] in self.word_list.keys():
                if label[i] in self.word_list[sentence[i]].keys():
                    val += np.log10(self.word_list[sentence[i]][label[i]]) + np.log10(self.emi_ident[label[i]])
                else:
                    # If part-of-speech is not in the emission probability for the given word
                    x = random.random()
                    if x > 0.5:
                        val += np.log10(x-0.5) + np.log10(self.emi_ident[label[i]])
                    else:
                        val += np.log10(x) + np.log10(self.emi_ident[label[i]])
        return val

    # Defining Viterbi(HMM) Model
    def hmm_viterbi(self, sentence):
        pred_words = [""]*len(sentence)
        hmm = {}        
        pos = {}
        
        # Defining the hmm and pos dictionaries
        for i in self.emi_ident.keys():
            if i not in ['max','freq']:
                hmm[i] = [0]*len(sentence)
                pos[i] = [0]*len(sentence)

        # Initializing the hmm dictionary
        for i in hmm.keys():
            if sentence[0] not in self.word_list.keys():
                hmm[i][0] = self.emi_ident[i]
                continue
            if i in self.word_list[sentence[0]].keys():
                hmm[i][0] = self.emi_ident[i]*self.word_list[sentence[0]][i]

        # Looping through all words and all part-of-speech
        for i in range(1,len(sentence)):
            for tag in pos.keys():
                
                # Iteratively getting the max probability value and corresponding part-of-speech along with updation of hmm, pos dictionaries 
                (pos[tag][i], hmm[tag][i]) =  max( [ (s, hmm[s][i-1] * self.trans_ident[s][tag]) for s in pos.keys() ], key=lambda l:l[1] ) 
                if sentence[i] not in self.word_list.keys():
                    hmm[tag][i] *= 1
                else:
                    if tag in self.word_list[sentence[i]].keys():
                        hmm[tag][i] *= self.word_list[sentence[i]][tag]
                    else:
                        hmm[tag][i] *= 0
        
        # Getting the last part-of-speech with max probability value and backtracking to get the other part-of-speech using pos dictionary
        (pred_words[len(sentence)-1],_) = max( [ (s, hmm[s][len(sentence)-1]) for s in hmm.keys() ], key=lambda l:l[1] )
        for i in range(len(sentence)-2, -1, -1):
            pred_words[i] = pos[pred_words[i+1]][i+1]

        return pred_words

    # Defining log of joint probability P(S,W) for Viterbi(HMM) model
    def posterior_hmm_viterbi(self,sentence,label):
        val = np.log10(self.emi_ident[label[0]])
        for i in range(len(sentence)):
            if sentence[i] in self.word_list.keys():
                if label[i] in self.word_list[sentence[i]].keys():
                    val += np.log10(self.word_list[sentence[i]][label[i]])
                else:
                    # If part-of-speech is not in the emission probability for the given word
                    x = random.random()
                    if x > 0.5:
                        val += np.log10(x-0.5)
                    else:
                        val += np.log10(x)
            
            # Summing up the log values of the transition 1 values
            if i > 0:
                val += np.log10(self.trans_ident[label[i-1]][label[i]])
        
        return val

    # Defining Monte Carlo Markov Chain(MCMC) Model 
    def complex_mcmc(self, sentence):
        prob = []
        all_pos = [i for i in self.emi_ident.keys() if i not in ['max','freq']]

        # Iterating through all words and all part-of-speech
        for i in sentence:
            pos_states = []
            for pos in all_pos:

                # Getting the Emission probabilities
                if i in self.word_list.keys():
                    if pos in self.word_list[i].keys():
                        pos_states.append(self.word_list[i][pos])
                    else:
                        pos_states.append(15)
                else:
                    pos_states.append(random.random())
            
            min_val = min(pos_states)

            # Re-initializing values which were not present in emission probabilities to min prob value * 1e-15(low value)
            for j in range(len(pos_states)):
                if pos_states[j] == 15:
                    pos_states[j] = min_val*1e-15

            # Normalizing the probability such that all probabilities sum to 1
            total_pos = sum(pos_states)
            pos_states = [j/total_pos for j in pos_states]
    
            # All probabilities of all part-of-speech for each word is updated in prob
            prob.append(pos_states)

        final_pos = []
        for i in range(len(sentence)):
            # N iterations
            kk=4500
            pos_max = {l:0 for l in all_pos}
            pos_max['max'] = ['na',-1]
            while kk>0:
                kk-=1
                # Randomly choosing part-of-speech along with probability weights
                val = random.choices(all_pos,weights=prob[i],k=1)
                pos_max[val[0]] += 1
                # Updating the part-of-speech having the max selection for the given word
                if pos_max[val[0]] > pos_max['max'][1] and pos_max['max'][0]!=val:
                    pos_max['max'] = [val[0],pos_max[val[0]]]
            
            # Updating the part-of-speech with maximum selections for the given word
            final_pos.append(pos_max['max'][0])

        return final_pos

# Initial Approach taken for MCMC

    # def complex_mcmc(self, sentence):
    #     total_samples = []
    #     total_samples.append(self.hmm_viterbi(sentence))

    #     for i in range(100):
    #         total_samples.append(self.gen_samples(sentence,total_samples[-1]))

    #     final_pos = []
    #     for i in range(len(sentence)):
    #         pos_prob_count = {i:0 for i in self.emi_ident.keys() if i not in ['max','freq']}
    #         pos_prob_count['max'] = ['na',-1]
    #         for s in total_samples:
    #             pos_prob_count[s[i]] += 1
    #             if pos_prob_count['max'][1] < pos_prob_count[s[i]] and s[i]!=pos_prob_count['max'][0]:
    #                 pos_prob_count['max'] = [s[i],pos_prob_count[s[i]]]
    #         final_pos.append(pos_prob_count['max'][0])
    #     return final_pos

    # def gen_samples(self,sentence,sample):
    #     pos = [i for i in self.emi_ident.keys() if i not in ['freq','max']]
    #     for i in range(len(sentence)):
    #         prob_val = [0]*len(pos)
    #         log_val = [0]*len(pos)
    #         for j in range(len(pos)):
    #             sample[i] = pos[j]
    #             log_val[j] = self.posterior_complex(sentence,sample)

    #         min_log = min(log_val)
    #         for j in range(len(log_val)):
    #             log_val[j] -= min_log
    #             prob_val[j] = math.pow(10,log_val[j]) 

    #         total_prob = sum(prob_val)
    #         prob_val = [k/total_prob for k in prob_val]
            
    #         cum_prob_val = 0
    #         rand_val = random.random()
    #         for p in range(len(prob_val)):
    #             cum_prob_val += prob_val[p]
    #             if cum_prob_val >= rand_val:
    #                 # print(f"Index:{i},Len:{len(cum_prob_val)},Final:{cum_prob_val[-1]},Prob:{sum(prob_val)}")
    #                 sample[i] = pos[p]
    #                 break
            
        
    #     return sample

    # Defining log of joint probability P(S,W) for Monte Carlo Markov Chain(MCMC) model
    def posterior_complex(self,sentence,label):
        val = np.log10(self.emi_ident[label[0]])
        for i in range(len(sentence)):
            if sentence[i] in self.word_list.keys():
                if label[i] in self.word_list[sentence[i]].keys():
                    val += np.log10(self.word_list[sentence[i]][label[i]])
                else:
                    # If part-of-speech is not in the emission probability for the given word
                    x = random.random()
                    if x > 0.5:
                        val += np.log10(x-0.5)
                    else:
                        val += np.log10(x)

            # Summing up the log values of the transition 1 values
            if i > 0:
                val += np.log10(self.trans_ident[label[i-1]][label[i]])
            
            # Summing up the log values of the transition 2 values
            if i > 1:
                val += np.log10(self.trans_ident2[label[i-2]][label[i-1]][label[i]])

        return val


    # This solve() method is called by label.py 
    # It should return a list of part-of-speech labelings of the sentence, one part of speech per word.
    
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

