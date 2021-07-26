# import all necessary models.

# sentence tokenizer.
from nltk import sent_tokenize
# a library for text generation using markov chains.
import markovify
# a library for natural language processing.
import nltk

# a function to extract titles from a given corpus.
def get_titles(text):
    sents = sent_tokenize(text)
    titles = []
    for s in sents:
        if s.isupper() and s.endswith('.'):
            titles.append(s.strip())
    return titles

# a function to split a corpus into tales by 
# a preset mark 'START'.
def get_tales(text):
    tales = text.split('START')
    return tales

# a function to calculate a third from a total number of 
# sentences in a tale.     
def get_portion(sents, procent=33):
    length = len(sents)
    portion = round((length * procent) / 100)
    return portion

# a function to collect the first third of each tale.
def get_beginning(sents):
    portion = get_portion(sents)
    beginning = sents[:portion]
    return ' '.join(beginning)

# a function to collect the second third of each tale.
def get_middle(sents):
    portion = get_portion(sents)
    middle = sents[portion:portion * 2]
    return ' '.join(middle)

# a function to collect the third third of each tale.
def get_ending(sents):
    portion = get_portion(sents)
    ending = sents[portion * 2:]
    return ' '.join(ending)

# a function to collect all beginnings, middle parts and endings
# of each tale into the sub-corpora.
def get_all(tales, procent=30):
    beginnings = []
    middles = []
    endings = []
    
    for tale in tales:
        if tale != '':
            sents = sent_tokenize(tale.strip())
            sents = sents[2:]
        
            beginning = get_beginning(sents)
            middle = get_middle(sents)
            ending = get_ending(sents)
            
            beginnings.append(beginning)
            middles.append(middle)
            endings.append(ending)
    return beginnings, middles, endings

# write down the corpora.
def write_down(part, name):
    with open('C:\\Users\\kanzl_000\\Desktop\\' + name, 'w',
              encoding='utf-8') as output:
        
        for line in part:
            output.write(line + '\n')
			
# a function to run the whole process.
def overall(text):
    parts = []
    tales = get_tales(text)
    parts.append(get_titles(text))

    for res in get_all(tales):
        parts.append(res)
        
    names = ['titles.txt', 'beginnings.txt', 'middles.txt', 'endings.txt']

    print(len(parts))
    for i, part in enumerate(parts):
        write_down(part, names[i])

# read an input corpus.    
text = open('C:\\Users\\kanzl_000\\Desktop\\tales.txt',
            'r', encoding='utf-8').read()

# run the whole process.
overall(text)
