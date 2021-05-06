#%%
import tensorflow as tf
import numpy as np
import transformers

# import tensorflow_hub as hub
# import tensorflow_text as text
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
from transformers import BertTokenizer, glue_convert_examples_to_features
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# train_ds = glue_convert_examples_to_features(raw_train_ds, tokenizer, max_length=128, task='mrpc')
# train_ds = train_ds.shuffle(100).batch(32).repeat(2)
#%%
sample = next(iter(train_ds))
sample
#%%
max_length = 100

text = ["ali gel"]
text = sample[0][0]
def encode_text(text, label):
    # text = tf.expand_dims(text, -1)
    
    encoded = tokenizer.batch_encode_plus(
                str(text),
                add_special_tokens=True,
                max_length=max_length,
                return_attention_mask=True,
                return_token_type_ids=True,
                pad_to_max_length=True,
                return_tensors="tf",
            )
    return encoded, label
    # input_ids = np.array(encoded["input_ids"], dtype="int32")
    # attention_masks = np.array(encoded["attention_mask"], dtype="int32")
    # token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")
    # return [input_ids, attention_masks, token_type_ids], label

#%%
train_ds = raw_train_ds.map(encode_text)
val_ds = raw_val_ds.map(encode_text)
test_ds = raw_test_ds.map(encode_text)
#%%
for encoded, label in train_ds.take(1):
    print(encoded)
    print(label)
sequence_output = bert_model(encoded)[0]
#%%
tf.keras.backend.clear_session()
input_ids = tf.keras.layers.Input(
    shape=(max_length,), dtype=tf.int32, name="input_ids"
)
# Attention masks indicates to the model which tokens should be attended to.
attention_masks = tf.keras.layers.Input(
    shape=(max_length,), dtype=tf.int32, name="attention_masks"
)
# Token type ids are binary masks identifying different sequences in the model.
token_type_ids = tf.keras.layers.Input(
    shape=(max_length,), dtype=tf.int32, name="token_type_ids"
)
# Loading pretrained BERT model.
bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
# Freeze the BERT model to reuse the pretrained features without modifying them.
bert_model.trainable = False

sequence_output = bert_model(
    input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
)[0]
#%
# Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
bi_lstm = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64, return_sequences=True)
)(sequence_output)
# Applying hybrid pooling approach to bi_lstm sequence output.
avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
concat = tf.keras.layers.concatenate([avg_pool, max_pool])
dropout = tf.keras.layers.Dropout(0.3)(concat)                          
output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout)
model = tf.keras.models.Model(
    inputs=[input_ids, attention_masks, token_type_ids], outputs=output
)

model.summary()
#%%
model.compile(loss="binary_crossentropy", 
                optimizer=tf.keras.optimizers.Adam(), 
                metrics=["accuracy"])
#%%
epochs = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)

#%%
#%%
from transformers import TFBertForSequenceClassification
# model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

#%%

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)
model.fit(train_ds, epochs=2)






















#%%
#%%
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
