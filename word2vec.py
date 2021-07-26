# a library for natural language processing. 
import gensim

# read a prelemmatized corpus line by line.
file = open('C:\\Users\\kanzl_000\\Desktop\\tales_lem.txt',
            encoding='utf-8').read().split('\n')

# split each line into tokens.
for i, line in enumerate(file):
    file[i] = line.split(' ')

# train a model.    
model = gensim.models.Word2Vec(file, min_count=3, size=150,
                               workers=4, window = 10)

# save the model.
model.save('C:\\Users\\kanzl_000\\Desktop\\word2vec_model')

#final size - 9843.
