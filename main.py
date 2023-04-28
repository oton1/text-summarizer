import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from heapq import nlargest
from nltk.tokenize import RegexpTokenizer
from deep_translator import GoogleTranslator


def summarize(text, n):
    # tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    tokenizer = RegexpTokenizer(r'\w+|[^ç\w\s]+')
    words = word_tokenize(text.lower())
    
    # remove stopwords
    stopwords_en = stopwords.words('english')
    words = [word for word in words if word not in stopwords_en]
    
    # compute word frequencies
    freq = FreqDist(words)
    
    # compute sentence scores based on word frequencies
    scores = defaultdict(float)
    for i, sentence in enumerate(sentences):
        for word in tokenizer.tokenize(sentence.lower()):
            if word in freq:
                scores[i] += freq[word]
        scores[i] /= len(tokenizer.tokenize(sentence.lower()))
    
    # select the top n sentences based on scores
    idx = nlargest(n, scores, key=scores.get)
    return ' '.join([sentences[i] for i in sorted(idx)])


# read input text from file
with open('trans.txt') as f:
    text = f.read()

# translate to Portuguese if necessary
if 'pt' not in nltk.corpus.stopwords.fileids():
    nltk.download('stopwords')
if 'pt' not in nltk.corpus.cmudict.fileids():
    nltk.download('cmudict')
if 'pt' not in nltk.corpus.wordnet.fileids():
    nltk.download('wordnet')

translate = False  # set this to True to translate the text to Portuguese
if translate:
    translator = GoogleTranslator(source='en', target='pt')
    text = translator.translate(text)

# summarize the text
summary = summarize(text, 2)
print(summary)
