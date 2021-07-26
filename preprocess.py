import pymorphy2  
from nltk import word_tokenize, sent_tokenize
from collections import Counter
import re

morph = pymorphy2.MorphAnalyzer()
reg = re.compile('[^-а-яА-я ]')

#stopwords = open('C:\\Users\\kanzl_000\\Desktop\\stopwords.txt',
#                 encoding='utf-8').read().split('\n')

#collocations = open('C:\\Users\\kanzl_000\\Desktop\\collocations_lem.txt',
#                 encoding='utf-8').read().split('\n')

def get_tok_sentences(file):
    
    sentences = sent_tokenize(file)

    for i, sent in enumerate(sentences):
        check_hyphens = word_tokenize(reg.sub(' ', sent.lower()))
        sentences[i] = list(filter(('-').__ne__, check_hyphens))

    return sentences

def lemmatize(sentences):

    lemmas = []
    lem_sents = []

    for sent in sentences:
        lemmas = []
        for t in sent:
            lemma = morph.parse(t)[0].normal_form
            lemmas.append(lemma)
        lem_sents.append(lemmas)

    return lem_sents

#def get_collocations(lem_sents, collocations):
#
#    with_collocations = []
#   
#    for i, sent in enumerate(lem_sents):
#        to_str = ' '.join(sent)
#        for c in collocations:
#            if c in to_str and c != '':
#                to_str = to_str.replace(c, c.replace(' ', '_'))
#        with_collocations.append(word_tokenize(to_str))
#
#    return with_collocations

#def remove_stopwords(with_collocations, stopwords):

#    for sentence in with_collocations:
#        for stop in stopwords:
#            if stop in sentence:
#                sentence.remove(stop)
                
#    return with_collocations
    

def stat_dict(lem_sents, tokens_count):
            
    words = sum(sentences, [])

    stat_dict = Counter(words).most_common()
    lines = str(stat_dict).split('),')
    
    with open('C:\\Users\\kanzl_000\\Desktop\\tales_stat_dict.txt', 'w',
              encoding='utf-8') as statistics:
        
        statistics.write('total number of words: ' +
                         str(tokens_count) + '\n')
        
        for line in lines:
            statistics.write(line + '\n')

    return statistics

file = open('C:\\Users\\kanzl_000\\Desktop\\tales.txt', encoding='utf-8').read()

sentences = get_tok_sentences(file)

tokens_count = len(sum(sentences, []))

lem_sents = lemmatize(sentences)

#with_collocations = get_collocations(lem_sents, collocations)

#clean = remove_stopwords(with_collocations, stopwords)

stat_dict = stat_dict(lem_sents, tokens_count)

with open('C:\\Users\\kanzl_000\\Desktop\\tales_lem.txt', 'w', encoding='utf-8') as f:
    for sent in lem_sents:
        to_str = ' '.join(sent)
        f.write(to_str + '\n')
