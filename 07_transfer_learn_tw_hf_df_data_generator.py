#%%
import numpy as np
import pandas as pd 
import transformers
import tensorflow as tf

# import tensorflow_hub as hub
# import tensorflow_text as text
print("Version: ", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# %%
max_length = 128  # Maximum length of input sentence to the model.
batch_size = 32
epochs = 2
# "bert-base-uncased"
model_name = "savasy/bert-base-turkish-sentiment-cased"
# Labels in our dataset.
# labels = ["1", "entailment", "neutral"]

# %%
main_df = pd.read_csv("../data/stock_tw/text_label_hg.csv")
main_df.reset_index(inplace=True)
msk = np.random.rand(len(main_df)) < 0.7
train_df = main_df[msk]
val_test_df = main_df[~msk]
msk = np.random.rand(len(val_test_df)) < 0.7
val_df = val_test_df[msk]
test_df = val_test_df[~msk]
train_df.shape, val_df.shape, test_df.shape
#%%
y_train = train_df["label"] #tf.keras.utils.to_categorical(train_df.label, num_classes=2)
y_val = val_df["label"] #tf.keras.utils.to_categorical(val_df.label, num_classes=2)
y_test = test_df["label"] #tf.keras.utils.to_categorical(test_df.label, num_classes=2)
# #%%
# tokenizer = transformers.BertTokenizer.from_pretrained(
#             "bert-base-uncased", do_lower_case=True)
# #%%
# # tw_text = train_df.loc[1:3, "Text"]
# tw_text = val_df[["Text"]].values[:3].astype("str")
# # for t in tw_text:
# #     print("------------")
# #     print(t[0])
# ls = [t[0] for t in tw_text.tolist()]
# ls
# #%%
# tw_text = [[t for t in item] for item in tw_text]
# print(tw_text)
# type(tw_text)
# #%%
# tw_text = ["come here", "go there"]
# type(tw_text)
# #%%
# encoded = tokenizer.batch_encode_plus(
#             [t[0] for t in tw_text.tolist()],
#             add_special_tokens=True,
#             max_length=max_length,
#             return_attention_mask=True,
#             return_token_type_ids=True,
#             pad_to_max_length=True,
#             return_tensors="tf",
#         )
# encoded
# %%
class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        tw_text,
        label,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.tw_text = tw_text
        self.label = label
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "savasy/bert-base-turkish-sentiment-cased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.tw_text))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.tw_text) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        tw_text = self.tw_text[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        # print("\ntw_text type is : ", type(tw_text))
        encoded = self.tokenizer.batch_encode_plus(
            [t[0] for t in tw_text.tolist()],
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )
        # print("\ntw_text type is : ", type(tw_text))
        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")
        # print("\ninput_ids : ", input_ids.shape)
        # print("attention_masks : ", attention_masks.shape)
        # print("token_type_ids : ", token_type_ids.shape)
        # print("\indexes type : ", type(indexes))
        # print("\indexes : ", indexes)
        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            label = np.array(self.label.iloc[indexes], dtype="int32")
            # print("label : ", label.shape)
            return [input_ids, attention_masks, token_type_ids], label
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)
#%%
# ind =  [51,  976,  184, 1001]
# print(type((ind)))
# y_train.iloc[ind]
#%%
train_data = BertSemanticDataGenerator(
    train_df[["Text"]].values.astype("str"),
    y_train,
    batch_size=batch_size,
    shuffle=True,
)
val_data = BertSemanticDataGenerator(
    val_df[["Text"]].values.astype("str"),
    y_val,
    batch_size=batch_size,
    shuffle=True,
)
#%%
sample = next(iter(val_data))
sample
#%%
for encoded in val_data.__getitem__(1):
    print(encoded)
    # print(label)
# sequence_output = bert_model(encoded)[0]
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
bert_model = transformers.TFBertModel.from_pretrained(model_name, from_pt=True)
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
epochs = 3
history = model.fit(
    train_data,
    validation_data=val_data,
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
