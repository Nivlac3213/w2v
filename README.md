## CS 399 Module 5: ai Exploring Neural Word Embeddings with Python

Word Embedding is a language modeling technique used for mapping words to vectors of real numbers.
It represents words or phrases in vector space with several dimensions.

### Installation

#### Wikipedia 2017

Pre-trained vectors trained on Wikipedia 2017, UMBC webbase corpus, and statmt.org news datasets.
The model contains 300-dimensional vectors for 1 million words. 682 MB uncompressed 2.26 GB.
https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
1st line contains header info (999994 300) .. renaming from .vec to .txt may make things easier.
Sorted most frequent words come 1st needs to be normalized.

#### Google News

Pre-trained vectors trained on a part of the Google News dataset (about 100 billion words).
The model contains 300-dimensional vectors for 3 million words and phrases.
1.65 GB uncompressed 3.64 Gigabytes into txt file: 10.77 GB
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
After conversion from bin into txt. 1st line contains header info (3000000 300).
Sorted most frequent words come 1st needs to be normalized. Conversion takes a long time requires 3rd party lib 'gensim'

#### Stanford’s GloVe

GloVe: Global Vectors for Word Representation https://github.com/stanfordnlp/GloVe
The Common Craw model contains 300-dimensional vectors for 2.2 million words. 2.18 GB uncompressed 5.65
GB. https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip
No header info. Sorted most frequent words come 1st needs to be normalized.

### Preprocessing

Start with loading the model. Consider skipping the 1st row, and normalizing the vectors, using the L2 norm.
Since the model is sorted, when playing with the model, you may not need to load all words,
and instead load only the most frequently used the 100_000 words

### Word Similarity

The **main.py** is Word2Vecs "hello world" example.
'''
king = model.find_word("king")
man = model.find_word("man")
woman = model.find_word("woman")
q = king - man + woman
'''

**wv.py** contains **_Word_** and **_Model_** classes for word vectors.
Those are used to load the model into memory, but does not normalize the vectors.
Loading even a model with just 100_000 words may still take about 5 seconds.

play.py contains two more examples. a_to_b_is_like_c_to what? 

'''
Berlin to Germany is like Paris to France
'''

... and "A list of words associated with 'Sweden' in order of proximity:

* Finland: 0.81715205
* Norway: 0.81073227
* Denmark: 0.80283506
* Iceland: 0.64265937
* Belgium: 0.58728528
* Switzerland: 0.56519071
* Luxembourg: 0.48025945
* France: 0.45033265
