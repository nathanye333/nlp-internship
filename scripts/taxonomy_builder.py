import nltk
nltk.download('punkt')
from collections import Counter
from nltk.util import ngrams
import pandas as pd
# Extract bigrams from remarks
df = pd.read_csv('data/processed/listing_sample.csv')
all_text = ' '.join(df['remarks'].dropna().str.lower())
tokens = nltk.word_tokenize(all_text)
bigrams = list(ngrams(tokens, 2))
freq = Counter(bigrams)
# Top 200 bigrams become taxonomy seed
for bigram, count in freq.most_common(200):
    print(f"{' '.join(bigram)}: {count}")