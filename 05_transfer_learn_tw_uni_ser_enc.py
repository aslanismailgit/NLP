#%%
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text
print("Version: ", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# %%
batch_size = 16
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
#%%
train_ds = raw_train_ds.cache().prefetch(buffer_size=5)
val_ds = raw_val_ds.cache().prefetch(buffer_size=5)
test_ds = raw_test_ds.cache().prefetch(buffer_size=5)
# %%
for text_batch, label_batch in train_ds.take(1):
    for i in range(2):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])

#%%
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text
print("Version: ", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

model_link = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
model = hub.KerasLayer(model_link, trainable=False)

#%%
embeddings = model(["ali gel."])
print(embeddings.shape)  #(4,128)

#%%
tf.keras.backend.clear_session()
from tensorflow.keras import layers
inputs = tf.keras.Input(shape=(),dtype=tf.string)

embeddings = model(inputs)
x = layers.Dropout(0.9)([embeddings])

# x = layers.Conv1D(64, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model_big = tf.keras.Model(inputs, predictions)
model_big.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model_big.summary()
#%%
epochs = 3
model_big.fit(train_ds, 
                validation_data=val_ds, 
                epochs=epochs)





#%%
