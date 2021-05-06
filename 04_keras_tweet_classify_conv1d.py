#%%
import tensorflow as tf
import numpy as np

print("Version: ", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# %%
batch_size = 32
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "tweets/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
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
# %%
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(1):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])
# %%
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string
import re

# Having looked at our data above, we see that the raw text contains HTML break
# tags of the form '<br />'. These tags will not be removed by the default
# standardizer (which doesn't strip HTML). Because of this, we will need to
# create a custom standardization function.
#%
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
#%%
# translationTable = str.maketrans("ğĞıİöÖüÜşŞçÇ", "gGiIoOuUsScC")
# tr = "ğĞıİöÖüÜşŞçÇ"
# tr_rep = "gGiIoOuUsScC"
# yourText = "Pijamalı Hasta Yağız Şoföre Çabucak Güvendi"
# yourText = yourText.translate(translationTable)
# for t, e in zip(tr, tr_rep):
#     tf.strings.regex_replace(yourText, tf.string(t), tf.string(e))
# yourText

#%%
# Model constants.
max_features = 20000
sequence_length = 500

# Now that we have our custom standardization, we can instantiate our text
# vectorization layer. We are using this layer to normalize, split, and map
# strings to integers, so we set our 'output_mode' to 'int'.
# Note that we're using the default split function,
# and the custom standardization defined above.
# We also set an explicit maximum sequence length, since the CNNs later in our
# model won't support ragged sequences.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# Now that the vocab layer has been created, call `adapt` on a text-only
# dataset to create the vocabulary. You don't have to batch, but for very large
# datasets this means you're not keeping spare copies of the dataset in memory.

# Let's make a text-only dataset (no labels):
text_ds = raw_train_ds.map(lambda x, y: x)
# Let's call `adapt`:
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
#%%
for text_vector, label in train_ds.take(1):
    print(label)
    print(text_vector)
# %%
tf.keras.backend.clear_session()
embedding_dim = 128
from tensorflow.keras import layers

inputs = tf.keras.Input(shape=(None,), dtype="int64")

x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
# x = layers.Conv1D(64, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(64, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.5)(x)

predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

model.compile(loss="binary_crossentropy", 
                optimizer=tf.keras.optimizers.Adam(), 
                metrics=["accuracy"])
model.summary()
# %%
epochs = 10

# Fit the model using the train and test datasets.
model.fit(train_ds, validation_data=val_ds, epochs=epochs)
# %%
model.evaluate(test_ds)
# 2160/(2533+2160)
# %%
# A string input
inputs = tf.keras.Input(shape=(1,), dtype="string")
# Turn strings into vocab indices
indices = vectorize_layer(inputs)
# Turn vocab indices into predictions
outputs = model(indices)

# Our end to end model
end_to_end_model = tf.keras.Model(inputs, outputs)
end_to_end_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Test it with `raw_test_ds`, which yields raw strings
end_to_end_model.evaluate(raw_test_ds)
# %%
