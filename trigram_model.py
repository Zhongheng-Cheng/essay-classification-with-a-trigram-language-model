import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2023 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """

    full_sequence = ['START'] * max(1, (n - 1)) + sequence + ['STOP']
    res = []
    for i in range(len(full_sequence) - n + 1):
        res.append(tuple(full_sequence[i:i + n]))
    return res


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        # Calculate the total number of words including "STOP"s but not "START"s
        self.word_count = sum(self.unigramcounts.values()) - self.unigramcounts[('START',)]
        self.sentence_count = self.unigramcounts[('START', )]


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int)

        for sentence in corpus:
            for i in get_ngrams(sentence, 1):
                self.unigramcounts[i] += 1
            for i in get_ngrams(sentence, 2):
                self.bigramcounts[i] += 1
            for i in get_ngrams(sentence, 3):
                self.trigramcounts[i] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability

        P(v | u, w) = P(u, w, v) / P(u, w)
        => p(v | u, w) = count(u, w, v) / count(u, w)
        """
        if trigram[:-1] == ('START', 'START'):
            # special case for ('START', 'START')
            denominator = self.sentence_count
        else:
            denominator = self.bigramcounts[trigram[:-1]]
            if denominator == 0:
                return 1 / len(self.lexicon)
        return self.trigramcounts[trigram] / denominator

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability

        P(w | u) = P(u, w) / P(u) 
        == (count((u, w)) / count_bigrams)) / (count(u) / count_unigrams_without_start)
        Since count_bigrams == count_unigrams_without_start
        we get P(w | u) = count((u, w)) / count(u)
        """
        if bigram[0] == 'START':
            # special case for ('START')
            denominator = self.sentence_count
        else:
            denominator = self.unigramcounts[bigram[:-1]]

        return self.bigramcounts[bigram] / denominator
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        if unigram == "START":
            return 0.0
        return self.unigramcounts[unigram] / self.word_count

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        from numpy.random import choice
        current_tokens = ('START', 'START')
        sentence = []
        while (not sentence or sentence[-1] != 'STOP') and len(sentence) < t:
            candidate_trigrams = [trigram for trigram in self.trigramcounts.keys() if trigram[:2] == current_tokens]
            candidate_words = [candidate[2] for candidate in candidate_trigrams]
            probabilities = [self.raw_trigram_probability(trigram) for trigram in candidate_trigrams]
            generated_word = choice(candidate_words, size=1, p=probabilities)[0]
            sentence.append(generated_word)
            current_tokens = (current_tokens[1], generated_word)
        return sentence            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return lambda1 * self.raw_trigram_probability(trigram) + \
            lambda2 * self.raw_bigram_probability(trigram[1:]) + \
            lambda3 * self.raw_unigram_probability(trigram[2:])
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        probabilities = [self.smoothed_trigram_probability(trigram) for trigram in trigrams]
        log_prob = sum([math.log2(prob) for prob in probabilities])
        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        log_prob_sum = 0
        word_count = 0
        for sentence in corpus:
            log_prob_sum += self.sentence_logprob(sentence)
            word_count += len(sentence)
        return 2 ** (-log_prob_sum / word_count)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # .. 
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
        
        return 0.0

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # print(get_ngrams(["natural","language","processing"],1))
    # print(get_ngrams(["natural","language","processing"],2))
    # print(get_ngrams(["natural","language","processing"],3))

    # print(model.trigramcounts[('START','START','the')])
    # print(model.bigramcounts[('START','the')])
    # print(model.unigramcounts[('the',)])

    # print(model.raw_trigram_probability(('i', 'told', 'you')))
    # print(model.smoothed_trigram_probability(('i', 'told', 'you')))
    # print(model.raw_bigram_probability(('the', 'biggest')))
    # print(model.raw_bigram_probability(('', )))
    # print(model.generate_sentence())

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

