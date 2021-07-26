# import all necessary models.

# a library for text generation using markov chains.
import markovify

# sentence tokenizer, part-of-speech-tagger, word tokenizer.
from nltk import sent_tokenize, pos_tag, word_tokenize

# regular expressions.
import re

# random number generator.
import random

# a pattern that leaves only alphabetic sequences 
# including hyphend words.
reg = re.compile('[^-а-яА-я ]')

# overwritten markovify.Text class enabled to use pos-tagging.
class POSifiedText(markovify.Text):

	# a function for splitting a corpus into sentences.
    def sentence_split(self, text):
        sents = sent_tokenize(text)
        return sents
    
	# a function for splitting sentences into tokens.
	# a list of tokens with pos-tags is returned.
    def word_split(self, sentence):
        words = re.split(self.word_split_pattern, sentence)
        words = [ "::".join(tag) for tag in pos_tag(words)]
        return words
	
	# a function for generating new sentences.
    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence

# a function that tries to generate a new sentences 
# that would begin with a word from a previous generated sentence.
def try_w_for_beginning(sent):
    words = word_tokenize(reg.sub(' ', sent.lower()))
    choices = []
    for w in words:
        try:
            try_w = beginning_gen.make_sentence_with_start(w, strict=False, tries=100)
            if try_w is not None:
                choices.append(try_w)
        except KeyError:
            continue
    
	# in case if several sentences were generated,
	# a random one is returned. 
    if len(choices) > 0:
        final = random.choice(choices)
		
	# otherwise a sentence is generated 
	# according to a regular algorithm that chooses a start randomly.
    else:
        final = beginning_gen.make_sentence()
        print('beginning fallback')
    
    return final.capitalize()

# the same is done for a middle part of a tale...
def try_w_for_middle(sent):
    words = word_tokenize(reg.sub(' ', sent.lower()))
    choices = []
    for w in words:
        try:
            try_w = middle_gen.make_sentence_with_start(w, strict=False, tries=100)
            if try_w is not None:
                choices.append(try_w)
        except KeyError:
            continue
        
    if len(choices) > 0:
        final = random.choice(choices)
    else:
        final = middle_gen.make_sentence()
        print('middle fallback')
    
    return final.capitalize()

# ... and the last one.
def try_w_for_ending(sent):
    words = word_tokenize(reg.sub(' ', sent.lower()))
    choices = []
    for w in words:
        try:
            try_w = ending_gen.make_sentence_with_start(w, strict=False, tries=100)
            if try_w is not None:
                choices.append(try_w)
        except KeyError:
            continue
        
    if len(choices) > 0:
        final = random.choice(choices)
    else:
        final = ending_gen.make_sentence()
        print('ending fallback')
    
    return final.capitalize()

# read subcorpora.     
titles =  open('C:\\Users\\kanzl_000\\Desktop\\titles.txt',
            'r', encoding='utf-8').read()   
beginnings = open('C:\\Users\\kanzl_000\\Desktop\\beginnings.txt',
            'r', encoding='utf-8').read()
middles = open('C:\\Users\\kanzl_000\\Desktop\\middles.txt',
            'r', encoding='utf-8').read()
endings = open('C:\\Users\\kanzl_000\\Desktop\\endings.txt',
            'r', encoding='utf-8').read()

# for different models are created. 
title_gen = POSifiedText(titles, state_size=1)
beginning_gen = POSifiedText(beginnings)
middle_gen = POSifiedText(middles)
ending_gen = POSifiedText(endings)

# generate a new tale sentence by sentence.
title = title_gen.make_sentence()
print(title)
beg1 = try_w_for_beginning(title)
print(beg1)
beg2 = try_w_for_beginning(beg1)
print(beg2)
mid1 = try_w_for_middle(beg2)
print(mid1)
mid2 = try_w_for_middle(mid1)
print(mid2)
end1 = try_w_for_ending(mid2)
print(end1)
print(try_w_for_ending(end1))
