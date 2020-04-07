import numpy as np
import time

def shingle(n, corpus):
    shingles = list()
    for start in range(0,len(corpus)-1, n-1):
        shingles.append(corpus[start:start+n])
    return shingles

def integerize_shingles(shingles):
    return [hash(shingle) for shingle in shingles]

def numpy_fillna(data):
    """Even out arrays with uneven length in a list."""
    lens = np.array([len(i) for i in data])
    mask = np.arange(lens.max()) < lens[:,None]
    out = np.zeros(mask.shape)
    out[mask] = np.concatenate(data)
    return out

def funkyReLU(x):
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
    result = corpus - document.transpose()
    return 1 - np.sum(funkyReLU(result), 1)/corpus.shape[1]

def compare_time(function, document, corpus):
    start = time.time()
    out = function(document, corpus)
    end = time.time()
    return (end - start, out)

def main():
    corpus_text = [
        "the quickest fox jumped over the lasfjdlk hjsfjf black dog",
        "the quick brown fox jumped over the lazy dog",
        "the quick brown dog jumped over the lazy fox",
        "today is a very good day",
        "fox is another word for attractive older woman",
        "get over it",
        "quick",
        "black",
        "for honor! and destiny!"
    ]
    document_text = "the quick brown fox jumps over the lazy dog" 
    
    corpus = numpy_fillna([integerize_shingles(shingle(2, document)) for document in corpus_text])
    document = numpy_fillna([integerize_shingles(shingle(2, document_text)),corpus[0]])[0]
    
    arr_time = compare_time(array_compare, document, corpus)
    num_time = compare_time(numpy_compare, document, corpus)

    print("Array took:", arr_time[0])
    print("Numpy took:", num_time[0])
    index = np.argmax(num_time[1])
    print("The most similar document is at index", index)
    print("|  \n|  \n|  ", corpus_text[index], "\n|  \n|  \n")
    
main()
