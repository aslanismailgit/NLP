#%%
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import time

#%%
sim_df = pd.read_csv("./data/sim_df.csv")
print(sim_df.to_string())


#%%  ------------  similarity calculation   ------------
# from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, pipeline

# model_name = "savasy/bert-base-turkish-sentiment-cased"
# # This model only exists in PyTorch, so we use the `from_pt` flag to import that model in TensorFlow.
# model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# sa = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
# #%%
# text = "normal"
# print("\n", text)
# pred = sa(text)
# print(pred[0]["label"], pred[0]["score"])
# #%%
# def get_sentiment(text):
#     sent = sa(text)
#     return [sent[0]["label"], sent[0]["score"]]
# text = "bugün hava çok süper"
# get_sentiment(text)

# #%%
#%%
df = pd.read_csv("./data/tweet_with_labels_sentim.csv")
df.dropna(inplace=True)
df.reset_index(inplace=True)
# df["text_len"] = df["Text"].apply(lambda x: len(x))
# df.drop(df["Text"].loc[df["text_len"]<10].index, inplace=True)
df.shape
# %%
start = time.time()
# df_small = df.iloc[0:10,:].copy()
# df_small['sentiment'], df_small["sent_score"] = zip(*df_small.apply(lambda row : get_sentiment(row['Text']),axis=1))

# df['sentiment'], df["sent_score"] = zip(*df.apply(lambda row : get_sentiment(row['Text']),axis=1))
# df.to_csv("./data/tweet_with_labels_sentim.csv", index=False)
print(f"total time passed {time.time()-start:.2f}")

#%%
df_tweet = pd.read_csv("./data/tweets_table_nisan.csv")
df_tweet["Timestamp"] = pd.to_datetime(df_tweet["Timestamp"])
df_tweet['Text'] = df_tweet['Text'].astype('str')
df_tweet = df_tweet.sort_values(["Timestamp"], ascending=False)
df_tweet['Text'].isnull().sum()
df_tweet = df_tweet.sort_values("Timestamp")
df_tweet.dropna(axis=0, inplace=True)

# %%
merged = pd.merge(df, df_tweet, how="inner",
                        left_on=['Text'], right_on=['Text'])
merged.drop(["index", "id", "Timestamp_x", "Timestamp_y", "Comments", "Likes", "Retweets"], axis=1, inplace=True)

#%%
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

i = 1
handles = merged.value_counts("Handle").index
# sim_df = pd.DataFrame(columns = ["Handle", "Similarity"])
sim_df = merged.value_counts("Handle").to_frame('Counts')
sim_df.reset_index(inplace=True)
for i, handles_temp in enumerate(handles):
    df_temp = merged.loc[merged["Handle"].astype("str") == handles_temp]
    X = df_temp.loc[:,["dif"]].values
    Y = df_temp.loc[:,["sent_score"]].values
    sim_score = cos_sim(np.transpose(X),np.transpose(Y))
    sim_df.loc[i, 'Handle'] = handles_temp
    sim_df.loc[i, 'Similarity'] = sim_score[0][0]
    # print(sim_df)
sim_df
# %%
plt.plot(sim_df.index, sim_df.Similarity.values)
# %%
sim_df["Sim_abs"] = abs(sim_df.Similarity.values)
sim_df.drop(sim_df.loc[sim_df.Counts<100].index, inplace=True)
sim_df.sort_values("Sim_abs", inplace=True, ascending=False)
sim_df
# %%
sim_df.Counts.values
# %%
plt.plot(sim_df.Counts.values[10:])
# %%
print(sim_df.to_string())
sim_df.to_csv("./data/sim_df.csv")
# %%
