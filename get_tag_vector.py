# import a library that support multi-dimensional arrays, matrices,
# and high-level mathematical functions.
import numpy as np

# lists of grammatical categories and their values:
# parts of speech, cases, number, genders, animacy,
# spects, persons, tenses, transitivity, voices, moods.
parts_of_speech = ['NOUN', 'ADJF', 'ADJS', 'COMP', 'VERB', 'INFN', 'PRTF',
                'PRTS', 'GRND', 'NUMR', 'ADVB', 'NPRO', 'PRED', 'PREP',
                'CONJ', 'PRCL', 'INTJ']   
cases = ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct', 'voct', 'gen2',
        'acc2', 'loc2']
numbers = ['sing', 'plur']
genders = ['masc', 'femn', 'neut']
animacy = ['anim', 'inan']
aspects = ['perf', 'impf']
persons = ['1per', '2per', '3per']
tenses = ['pres', 'past', 'futr'] 
transitivity = ['tran', 'intr']
voices = ['actv', 'pssv']
moods = ['indc', 'impr']

# a two-dimensional list of all categories and their grammemes.
categories = []
categories.append(parts_of_speech)
categories.append(cases)
categories.append(numbers)
categories.append(genders)
categories.append(animacy)
categories.append(aspects)
categories.append(persons)
categories.append(tenses)
categories.append(transitivity)
categories.append(voices)
categories.append(moods)

# a dictionary of indexed categories.
gram_dict = {}

# indexation.
for i, category in enumerate(categories):
    gram_dict[i] = category

# a function for tagset vectorization.
def get_tag_vector(tag):
    sorted_vectors = get_vectors(tag)
    all_vectors = get_all_vectors(sorted_vectors)
	
	# all vectors are concatenated into one.
    final_vector = np.concatenate(all_vectors)

    return final_vector        

# a function for collecting vectors for each category.
def get_vectors(tag):

    vectors = []
	
	# break a tagset into separate tags.
    tag = ' '.join(str(tag).split(',')).split()
    
	# if a tag belongs to a particular category,
	# get a vector for that category and add it to 
	# the list of vectors for each category to concatenate later.
	# indexation is used to keep the same order of vectors.
	# a sorted list of tuples idx-vector is returned.
    for grammeme in tag:

        if grammeme in gram_dict[0]:
            vector = get_grammeme_vector(grammeme, gram_dict[0])
            vectors.append((0, vector))
            continue
          
        elif grammeme in gram_dict[1]:
            vector = get_grammeme_vector(grammeme, gram_dict[1])
            vectors.append((1, vector))
            continue

        elif grammeme in gram_dict[2]:
            vector = get_grammeme_vector(grammeme, gram_dict[2])
            vectors.append((2, vector))
            continue

        elif grammeme in gram_dict[3]:
            vector = get_grammeme_vector(grammeme, gram_dict[3])
            vectors.append((3, vector))
            continue

        elif grammeme in gram_dict[4]:
            vector = get_grammeme_vector(grammeme, gram_dict[4])
            vectors.append((4, vector))
            continue

        elif grammeme in gram_dict[5]:
            vector = get_grammeme_vector(grammeme, gram_dict[5])
            vectors.append((5, vector))
            continue

        elif grammeme in gram_dict[6]:
            vector = get_grammeme_vector(grammeme, gram_dict[6])
            vectors.append((6, vector))

        elif grammeme in gram_dict[7]:
            vector = get_grammeme_vector(grammeme, gram_dict[7])
            vectors.append((7, vector))
            continue

        elif grammeme in gram_dict[8]:
            vector = get_grammeme_vector(grammeme, gram_dict[8])
            vectors.append((8, vector))
            continue
        
        elif grammeme in gram_dict[9]:
            vector = get_grammeme_vector(grammeme, gram_dict[9])
            vectors.append((9, vector))
            continue

        elif grammeme in gram_dict[10]:
            vector = get_grammeme_vector(grammeme, gram_dict[10])
            vectors.append((10, vector))
            continue
		
    return sorted(vectors, key=lambda x: x[0])

# a function to get a vector for a particular category.
def get_grammeme_vector(grammeme, features):
    
    length = len(features) + 1
    vector = np.zeros((length), dtype=np.int)

    for i, feature in enumerate(features):
        if grammeme == feature:
            vector[i] = 1

    return vector

# a function for filling positions for categories that absent for 
# a current word.
def get_all_vectors(vectors):
    indices = []
    all_vectors = []
    
    for idx in vectors:
        indices.append(idx[0])
    
	# 11 is taken because cases are the second large list after
	# parts of speech which are supposed always to be identified. 
	# if an idx for a category is absent, a vector of the length of
	# that category number of values is created and 1 is placed at
	# the last position.
    for i in range(11):
        if i not in indices:
            length = len(gram_dict[i]) + 1
            vector = np.zeros((length), dtype=np.int)
            vector[length - 1] = 1
            vectors.append((i, vector))

    for pair in sorted(vectors, key=lambda x: x[0]):
       all_vectors.append(pair[1])

    return all_vectors

#tag = 'NOUN,anim,femn sing,nomn'
#print(get_tag_vector(tag), len(get_tag_vector(tag)))
