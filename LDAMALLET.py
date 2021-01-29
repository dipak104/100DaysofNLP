
    os.environ.update({'MALLET_HOME':r'C:/Users/dtripathi/Downloads/mallet-2.0.8/mallet-2.0.8/bin'})
    mallet_path = 'C:/Users/dtripathi/Downloads/mallet-2.0.8/mallet-2.0.8/bin/mallet.bat'
    #print(dir(mallet_path))
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)
    print('\nSave Ldamallet model')
    ldamallet.save('Day_13/ldamallet.model')
    print('\nLoad lda Mallet model')
    mallet_model = gensim.models.wrappers.LdaMallet.load('Day_13/ldamallet.model')
    print(mallet_model.show_topics(formatted=True))
    coherence_model_mallet = CoherenceModel(model=mallet_model, texts = lemmatized_data, dictionary=id2word, coherence='c_v')
    coherence_mallet = coherence_model_mallet.get_coherence()
    print('\nCoherence Score : ', coherence_mallet)
