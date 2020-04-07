import numpy as np
import random 

import time

# funky ReLU
def funkyReLU( x ):
        x[x!=0] = 1
        return x

def array_compare(in_doc, corpus):
    sims = []
    for doc in corpus:
        matches = 0
        for i, val in enumerate(doc):
            if val == in_doc[i]:
                matches += 1
        sims.append(matches/len(doc))
    return sims

def numpy_compare(document, corpus):
    result = np.array(corpus) - np.array(document).transpose()
    return 1 - np.sum(funkyReLU(result), 1)/10000

def compare_time(function, document, corpus):
    start = time.time()
    function(document, corpus)
    end = time.time()
    return end - start

def main():
    num_chars = 50
    doc_size = 10
    num_docs = 100000

    # generation can take a second based on the above numbers... so if it seems like it's frozen, it isn't
    cur = [random.randint(0, num_chars) for _ in range(doc_size)]
    documents = [[random.randint(0, num_chars) for _ in range(doc_size)] for _ in range(num_docs)]

    arr_time = compare_time(array_compare, cur, documents)
    num_time = compare_time(numpy_compare, cur, documents)

    print("Array took:", arr_time)
    print("Numpy took:", num_time)

main()