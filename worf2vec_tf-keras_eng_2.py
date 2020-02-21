import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
import nltk

# corpus based on https://github.com/minsuk-heo/python_tutorial/tree/master/data_science/nlp
# changed to nltk, tf.keras

#%%
corpus = ['king is a strong man', 
          'queen is a wise woman', 
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong', 
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']

dfc=pd.DataFrame(corpus)

#%%
stop_words = ['is', 'a', 'will', 'be']

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
sentences = []
for sentence in dfc[0]:
    sent = [w for w in sentence.split() if w not in stop_words] 
    sentences.append(sent)
#    print(sentence, sent)
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
history = model.fit(xx, yy, epochs=20000)

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
#new_p=w2v_df.loc[4,:]+w2v_df.loc[3,:]
#plt.plot(new_p['x1'],new_p['x2'],"bx")
#ax.annotate(new_p[0], (new_p['x1'],new_p['x2'] ))    
#
#new_p2 = (w2v_df[w2v_df['word']=="guclu"].values)+ (w2v_df[w2v_df['word']=="prenses"].values)
#plt.plot(new_p2[0,1],new_p2[0,2],"bx")
#ax.annotate(new_p2[0,0], (new_p2[0,1],new_p2[0,2]))    


plt.show()
fig.savefig("English_20000_epochs_3.png")
