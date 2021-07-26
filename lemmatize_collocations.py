import pymorphy2  
from nltk import word_tokenize, sent_tokenize

morph = pymorphy2.MorphAnalyzer()
collocations = open('C:\\Users\\kanzl_000\\Desktop\\collocations.txt',
                 encoding='utf-8')

with open('C:\\Users\\kanzl_000\\Desktop\\collocations_lem.txt', 'w',
                 encoding='utf-8') as f:
    
    for line in collocations:
        s = ''
        s2 = ''
        words = word_tokenize(line)
        if len(words) != 2:
            for w in words:
                lemma = morph.parse(w)[0].normal_form
                s += lemma + ' '
                
            f.write(s.strip() + '\n')

            new_s = s.replace('ё', 'е')
            if new_s != s:
                f.write(new_s.strip() + '\n')

        else:
            for w in words:
                lemma = morph.parse(w)[0].normal_form
                s += lemma + ' '

            f.write(s.strip() + '\n')

            new_s = s.replace('ё', 'е')
            if new_s != s:
                f.write(new_s.strip() + '\n')
                
            for w in words[::-1]:
                lemma = morph.parse(w)[0].normal_form
                s2 += lemma + ' '

            f.write(s2.strip() + '\n')

            new_s2 = s2.replace('ё', 'е')
            if new_s2 != s2:
                f.write(new_s2.strip() + '\n')
