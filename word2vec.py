import numpy as np
from nltk.corpus import brown
import timeit
import sys
import getopt
import matplotlib.pyplot as plt

epsilon = 1E-8

def sigmoid(param):
    return (1 / (1 + np.exp(-param)))

class Word2vec(object):

    def __init__(self):
        self.dim = 300
        self.words = []
        self.vocab = []
        self.vectors = {}
        self.U = np.empty((0))
        self.V = np.empty((0))
        self.training_examples = [] 
        self.learning_rate = 0.001
        self.neg_samples = 4 
        self.window = 2

    def initialize_params(self):
        self.words = [word.lower() for word in brown.words()[0:10000]];
        punc = ['.', '!', ';', '?']
        punctuation_indices = [i for i,x in enumerate(self.words) if x in punc]
        
        print 'Making word-->vector hash'
        for i in range(0, len(self.words)):
            if i not in punctuation_indices:
                if self.words[i] not in self.vectors.keys():
                    self.vectors[self.words[i]] = [np.random.random_sample((self.dim,1)) - 0.5, np.random.random_sample((self.dim,1)) - 0.5]

        self.vocab = sorted(self.vectors.keys())
        
        print 'Creating training examples'
        punctuation_indices.insert(0, -1)
        for i in range(0,len(punctuation_indices) - 1): 
            index = punctuation_indices[i]
            next_index = punctuation_indices[i + 1]
            self.training_examples.append(self.words[index + 1 : next_index])
        
        print 'Initializing parameters'
        self.U = np.empty((self.dim, 0))
        self.V = np.empty((self.dim, 0))
        for key in self.vocab:
            value = self.vectors.get(key)
            self.U = np.hstack((self.U, value[0]))
            self.V = np.hstack((self.V, value[1]))

        print 'Training Examples: %d' % len(self.training_examples)
        print 'Sample Example: %s' % self.training_examples[10]
        print 'Vocab Size: %d' % len(self.vocab)
        print 'Parameter matrix size: %s' % str(np.shape(self.U))
        print 'Parameter matrix entry statistics\n\tMean: %f\n\tVariance: %f' % (np.mean(self.V), np.var(self.V))
        raw_input('Press enter to begin training')

    def evaluate_loss_global(self):
        print 'Evaluating loss'
        loss = 0
        for e in self.training_examples:
            for i in range(0, len(e)):
                start = max(0, i - self.window)
                end = min(len(e), i + self.window)
                
                for curr in range(start, end):
                    random_samples = np.random.randint(0, len(self.vocab), self.neg_samples)
                    center_word = e[i]
                    outside_word = e[curr]
                    center_vec = self.vectors[center_word][0]
                    outside_vec = self.vectors[outside_word][1]
                    l = np.log(sigmoid(np.transpose(center_vec).dot(outside_vec)))
                    loss = loss + l

                    for r in random_samples:
                        outside_word = self.vocab[r]
                        outside_vec = self.vectors[outside_word][1]
                        l = np.log(1 - sigmoid(np.transpose(center_vec).dot(outside_vec)))
                        loss = loss + l
        
        return loss.flatten().flatten() / len(self.training_examples)

    def evaluate_loss_single(self, example, random_samples):
        loss = 0
        for i in range(0, len(e)):
            start = max(0, i - self.window)
            end = min(len(e), i + self.window)
                
            for curr in range(start, end):
                center_word = e[i]
                outside_word = e[curr]
                center_vec = self.vectors[center_word][0]
                outside_vec = self.vectors[outside_word][1]
                l = np.log(sigmoid(np.transpose(center_vec).dot(outside_vec)))
                loss = loss + l

                for r in random_samples:
                    outside_word = self.vocab[r]
                    outside_vec = self.vectors[outside_word][1]
                    l = np.log(1 - sigmoid(np.transpose(center_vec).dot(outside_vec)))
                    loss = loss + l
      
        return loss

       


    def update_params(self, gradU, gradV):
        old_u = self.U
        old_v = self.V
        self.U = self.U + self.learning_rate * gradU
        self.V = self.V + self.learning_rate * gradV
        for i in range(0, len(self.vocab)):
            curr_word = self.vocab[i]
            self.vectors[curr_word][0] = self.U[:, i]
            self.vectors[curr_word][1] = self.V[:, i]

    
    def calc_grad(self, example, random_samples):
        gradU = np.zeros((self.dim, len(self.vocab)))
        gradV = np.zeros((self.dim, len(self.vocab)))
        for i in range(0, len(example)):
            start = max(0, i - self.window)
            end = min(len(example), i + self.window)
            for j in range(start, end):
                center_word = example[i]
                outside_word = example[j]
                center_vec = self.vectors[center_word][0].flatten()
                outside_vec = self.vectors[outside_word][1].flatten()
                activation = sigmoid(np.transpose(center_vec).dot(outside_vec))
                U_col = self.vocab.index(center_word)
                gradU[:, U_col] = gradU[:, U_col] + (1 - activation) * (outside_vec)
                V_col = self.vocab.index(outside_word)
                gradV[:, V_col] = gradV[:, V_col] + (1 - activation) * (center_vec) 
                for r in random_samples:
                    outside_word = self.vocab[r]
                    outside_vec = self.vectors[outside_word][1].flatten()
                    activation = sigmoid(np.transpose(center_vec).dot(outside_vec))
                    gradV[:, r] = gradV[:, r] + (activation - 1) * (center_vec)

        return (gradU, gradV, random_samples)


if __name__ == '__main__':
    ##Defaults Params##
    nepoch = 1
    ###################

    optlist, args = getopt.getopt(sys.argv[1:], 'n:', ['nepoch=']) 
    for o,a in optlist:
        if o in ('--nepoch', '-n'):
            nepoch = int(a)
        else:
            print 'Unhandled option'
            sys.exit(2)

    loss = []
    w = Word2vec()
    w.initialize_params()
    for i in range(0, nepoch):
        if i % 10 == 0:
            l = w.evaluate_loss_global()
            print 'Loss: %f' % l
        #print 'Parameter matrix entry statistics\n\tMean: %f\n\tVariance: %f' % (np.mean(w.U), np.var(w.U))
        #raw_input('Press enter to begin training')
        loss.append(l)
        for e in w.training_examples:
            random_samples = np.random.randint(0, len(w.vocab), w.neg_samples)
            gradU, gradV, random_samples = w.calc_grad(e, random_samples)
            w.update_params(gradU, gradV)
        
    sample_word = 'administration'
    index = w.vocab.index(sample_word)
    word_vectors = w.U + w.V
    vector = word_vectors[:, index]
    (rows, cols) = np.shape(word_vectors)
    distance = 10000000
    closest_vector = -1
    for i in range(0, cols):
        d = np.linalg.norm(word_vectors[:,i] - vector)
        if d < distance and w.vocab[i] != sample_word:
            distance = np.linalg.norm(word_vectors[:, i] - vector)
            closest_vector = i
    closest_word = w.vocab[closest_vector]
    print 'Closest word to %s is %s' % (sample_word, closest_word)





