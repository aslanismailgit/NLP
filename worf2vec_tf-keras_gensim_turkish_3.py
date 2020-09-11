import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
import nltk

# corpus is taken from: https://github.com/minsuk-heo/python_tutorial/tree/master/data_science/nlp
# changed to nltk, tf.keras, also in Turkish
# added gensim model

#%%
corpus = {'kral guclu bir adam', 
          'kralice bilge bir kadin', 
          'oglan genc bir adam',
          'kiz genc bir kadin',
          'prens genc bir kral',
          'prenses genc bir kralice',
          'adam guclu', 
          'kadin guzel',
          'prens oglan kral',
          'prenses kiz kralice'}
dfc=pd.DataFrame(corpus)

#%%
stop_words = ['bir']   

txt = dfc[0].str.lower().str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(txt)
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stop_words) 
uniq_words  = {uw for uw in words_except_stop_dist}

rslt = pd.DataFrame(words_except_stop_dist.most_common(len(uniq_words)),
                    columns=['Word', 'Frequency']).set_index('Word')
matplotlib.style.use('ggplot')
rslt.plot.bar(rot=0)


#%% another method
tokens = [x for x in word_tokenize(txt) if x not in stop_words]
" ".join(tokens)

#%%
sentences = [[sent for sent in corp.split() if sent not in stop_words] for corp in corpus]
sentences

#%%  
WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1] : 
            if neighbor != word:
                data.append([word, neighbor])
data

#%%
dfc2 = pd.DataFrame(data, columns = ['input', 'label'])

#%%
from sklearn.preprocessing import LabelBinarizer
#%%
x = dfc2["input"]
y = dfc2["label"]
lb = LabelBinarizer()
xx = lb.fit_transform(x)
yy = lb.transform(y)

#%%
import tensorflow as tf
from tensorflow import keras

#%%
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Input(shape=xx.shape[1]))
model.add(keras.layers.Dense(2, activation="relu"))
model.add(keras.layers.Dense(xx.shape[1], activation="softmax"))

#model.summary()
optimizer = keras.optimizers.SGD(lr=0.05)
model.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
#%%

history = model.fit(xx, yy, epochs=2000)

#%%
#%matplotlib qt
hidden1 = model.layers[1]
W1, b1= hidden1.get_weights()

#%
vectors = (W1*1 + b1*1)
print(vectors)
vectors = np.transpose(vectors)
#%
w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
w2v_df['word'] = lb.classes_
w2v_df = w2v_df[['word', 'x1', 'x2']]
#w2v_df

#%
fig, ax = plt.subplots()

for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word, (x1,x2 ))
plt.plot(w2v_df['x1'],w2v_df['x2'],"r.") 
   
PADDING = 0.70
x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
y_axis_max = np.amax(vectors, axis=0)[1] + PADDING
 
plt.xlim(x_axis_min,x_axis_max)
plt.ylim(y_axis_min,y_axis_max)
plt.rcParams["figure.figsize"] = (7,7)

ax.axhline(y=0, color='b')
ax.axvline(x=0, color='b')
 
plt.show()
#fig.savefig("Turkish_20000_epochs_3.png")
#%% BUILD MODEL USING GENSIM 
import gensim
import pandas as pd
model = gensim.models.Word2Vec(
    sentences,
    size=2,
    window=2,
    min_count=0,
    workers=1,
    iter=100)
#%%
w1 = "kral"
model.wv.most_similar(positive=w1)

w2 = "kralice"
model.wv.similarity(w1,w2)
#%%
words = list(model.wv.vocab)
	
X = model.wv[model.wv.vocab]
w2v_df2 = pd.DataFrame(words, columns = ['word'])
w2v_df2["x1"] = X[:,0]
w2v_df2["x2"] = X[:,1]

#%%
fig, ax = plt.subplots()

for word, x1, x2 in zip(w2v_df2['word'], w2v_df2['x1'], w2v_df2['x2']):
    ax.annotate(word, (x1,x2 ))
plt.plot(w2v_df2['x1'],w2v_df2['x2'],"r.")    
PADDING = 0.30
x_axis_min = np.amin(X, axis=0)[0] - PADDING
y_axis_min = np.amin(X, axis=0)[1] - PADDING
x_axis_max = np.amax(X, axis=0)[0] + PADDING
y_axis_max = np.amax(X, axis=0)[1] + PADDING
 
plt.xlim(x_axis_min,x_axis_max)
plt.ylim(y_axis_min,y_axis_max)
plt.rcParams["figure.figsize"] = (7,7)

ax.axhline(y=0, color='b')
ax.axvline(x=0, color='b')
plt.title("GENSIM Model W2V") 
plt.show()
#fig.savefig("Turkish_20000_epochs_Gensim.png")
