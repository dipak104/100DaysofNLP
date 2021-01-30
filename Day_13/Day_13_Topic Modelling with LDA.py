import re
import gensim
import pandas as pd
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
from multiprocessing import Process, freeze_support
import os

def run():
    nlp = spacy.load('en_core_web_lg', disable=['parser','ner'])
    stopwords = list(nlp.Defaults.stop_words)
    df = pd.read_json('Day_13/text.json')
    data = df.content.values.tolist()
    def clean_text(text):       # Clean the data
        text = [re.sub('\S*@\S*\s?','', sent) for sent in text]
        text = [re.sub('\s+',' ', sent) for sent in text]
        text = [re.sub("\'","", sent) for sent in text]
        return text
    clean_data = clean_text(data)
    def sent_to_words(sentences):
        for sent in sentences:
            yield(simple_preprocess(str(sent), deacc=True))
    words = list(sent_to_words(clean_data))
    def remove_stopwords(text):
        return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in text]
    def lemmatized(text, postag = ['NOUN', 'AJD', 'VERB']):
        final = []
        for sent in text:
            doc = nlp(" ".join(sent))
            final.append([token.lemma_ for token in doc if token.pos_ in postag])
        return final
    clean_words = remove_stopwords(words)
    lemmatized_data = lemmatized(clean_words)
    id2word = corpora.Dictionary(lemmatized_data)  # Create dictionary
    corp = lemmatized_data      # Create corpus
    corpus = [id2word.doc2bow(text) for text in corp]  # Term document frequency
    # Build the topic model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word = id2word, num_topics=20, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto',per_word_topics=True)
    print('Save lda Model')
    lda_model.save('Day_13/lda.model')
    print('\nLoad LDA model')
    model = gensim.models.LdaModel.load('Day_13/lda.model')
    print(model.print_topics())
    doc = model[corpus]
    # Compute Model Perplexity and coherence score
    print('\nPerplexity: ', model.log_perplexity(corpus))
    # Compute coherence score
    coherence_model = CoherenceModel(model = model, texts=lemmatized_data, dictionary=id2word, coherence='c_v')
    coherence = coherence_model.get_coherence()
    print('\nCoherence Score: ', coherence)
if __name__=="__main__":
    freeze_support()
    Process(target=run).start()