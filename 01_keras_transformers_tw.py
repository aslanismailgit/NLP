#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#%%
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
#%%
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
#%%
# vocab_size = 20000  # Only consider the top 20k words
# maxlen = 200  # Only consider the first 200 words of each movie review
# (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
# print(len(x_train), "Training sequences")
# print(len(x_val), "Validation sequences")
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

# %%
batch_size = 16
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "tweets/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    shuffle=True,
    seed=1337,
)
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "tweets/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
)
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "tweets/test", batch_size=batch_size
)

print(
    "Number of batches in raw_train_ds: %d"
    % tf.data.experimental.cardinality(raw_train_ds)
)
print(
    "Number of batches in raw_val_ds: %d" 
    % tf.data.experimental.cardinality(raw_val_ds)
)
print(
    "Number of batches in raw_test_ds: %d"
    % tf.data.experimental.cardinality(raw_test_ds)
)
#%%
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re
import string

vocab_size = 20000 #vocab_size
maxlen = 500

def replace_tr(yourText):
    yourText = tf.strings.regex_replace(yourText,"ğ", "g")
    yourText = tf.strings.regex_replace(yourText,"Ğ", "G")
    yourText = tf.strings.regex_replace(yourText,"İ", "i")
    yourText = tf.strings.regex_replace(yourText,"ı", "i")
    yourText = tf.strings.regex_replace(yourText,"Ö", "O")
    yourText = tf.strings.regex_replace(yourText,"ö", "o")
    yourText = tf.strings.regex_replace(yourText,"ü", "u")
    yourText = tf.strings.regex_replace(yourText,"Ü", "g")
    yourText = tf.strings.regex_replace(yourText,"ş", "s")
    yourText = tf.strings.regex_replace(yourText,"Ş", "S")
    yourText = tf.strings.regex_replace(yourText,"ç", "c")
    yourText = tf.strings.regex_replace(yourText,"Ç", "C")
    return yourText
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    remove_turkish_car = replace_tr(lowercase)
    stripped_html = tf.strings.regex_replace(remove_turkish_car, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=maxlen,
)

#%% Let's make a text-only dataset (no labels):
text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)
#%%
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)
# %%
for text_batch, label_batch in train_ds.take(1):
    for i in range(2):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])
#%%
tf.keras.backend.clear_session()
embed_dim = 128  # Embedding size for each token
num_heads = 20  # Number of attention heads
ff_dim = 50  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.1)(x)
# outputs = layers.Dense(2, activation="softmax")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
#%%
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

epochs = 3
history = model.fit(
                    train_ds, 
                    validation_data=val_ds, 
                    epochs=epochs,
                )
# %%
model.evaluate(test_ds)
#%%
