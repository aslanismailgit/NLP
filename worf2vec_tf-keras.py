import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# based on https://github.com/minsuk-heo/python_tutorial/tree/master/data_science/nlp
# changed to tf.keras

#%%
#corpus = ['king is a strong man', 
#          'queen is a wise woman', 
#          'boy is a young man',
#          'girl is a young woman',
#          'prince is a young king',
#          'princess is a young queen',
#          'man is strong', 
#          'woman is pretty',
#          'prince is a boy will be king',
#          'princess is a girl will be queen']
corpus = ['kral guclu bir adam', 
          'karalice bilge bir kadin', 
          'oglan genc bir adam',
          'kiz genc bir kadin',
          'prens genc bir kral',
          'prenses genc bir kralice',
          'adam guclu', 
          'kadin güzel',
          'prens oğlan kral',
          'prenses kiz kralice']
#%%
def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
    
    return results

#%%
corpus = remove_stop_words(corpus)
#%%
words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)
#%%
words = set(words)
#%%
word2int = {}

for i,word in enumerate(words):
    print(i, word)
    word2int[word] = i

#%%
sentences = []
for sentence in corpus:
    print(sentence)
    sentences.append(sentence.split())
#%%    
WINDOW_SIZE = 2

data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1] : 
            if neighbor != word:
                data.append([word, neighbor])

#%%
#%%
for text in corpus:
    print(text)

#%%
df = pd.DataFrame(data, columns = ['input', 'label'])

#%%
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

x = df["input"]
y = df["label"]
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
#%
model = keras.models.Sequential()
model.add(keras.layers.Input(shape=X_train.shape[1:]))
model.add(keras.layers.Dense(2, activation="relu"))
model.add(keras.layers.Dense(X_train.shape[1], activation="softmax"))

#%
#model.summary()
optimizer = keras.optimizers.SGD(lr=0.05)
model.compile(loss="binary_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])
#%
#%%

history = model.fit(xx, yy, epochs=1000)

#%%
hidden1 = model.layers[1]
W1, b1= hidden1.get_weights()

#%
vectors = (W1*1 - b1*1)
print(vectors)
vectors = np.transpose(vectors)
#%
w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
w2v_df['word'] = words
w2v_df = w2v_df[['word', 'x1', 'x2']]
w2v_df

#%
fig, ax = plt.subplots()

for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word, (x1,x2 ))
    
PADDING = 0.20
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


#%%