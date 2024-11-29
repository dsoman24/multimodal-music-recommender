import sqlite3
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import pickle

class LyricsDataset(Dataset):
    def __init__(self, song_ids, texts, tokenizer, max_length=512):
        self.song_ids = song_ids
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        song_id = self.song_ids[idx]
        tokens = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        tokens = {key: val.squeeze(0) for key, val in tokens.items()}  # Remove batch dimension
        return song_id, tokens

class LyricsProvider:

    def __init__(self, lyrics_pivot) -> None:
        self.lyrics_df = None
        self.lyrics_pivot = lyrics_pivot
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  
        
        
    def get_roberta_embeddings(self, lyrics_pivot, batch_size=8):
        '''
        Converts each row (song) in pivot table to a text format and generates embeddings for each song using RoBERTa.
        Processes songs in batches to improve efficiency.
        '''
        song_ids, texts = self.get_embedding_texts()
        dataset = LyricsDataset(song_ids, texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        song_embeddings = []
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating Embeddings"):
                batch_song_ids = batch[0]
                batch_tokens = {key: val.to(self.device) for key, val in batch[1].items()}

                outputs = self.model(**batch_tokens)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token embeddings
                
                # Collect results
                for song_id, embedding in zip(batch_song_ids, embeddings):
                    song_embeddings.append({'song_id': song_id, 'embedding': embedding})


        embeddings_df = pd.DataFrame(song_embeddings)
        return embeddings_df

    def get_embedding_texts(self):
        '''
        Given text, returns CLS embedding using RoBERTa
        '''
        texts, song_ids = [], []
        for _, row in self.lyrics_pivot.iterrows():
            song_id = row['song_id']
            lyrics_text = " ".join(
                [f"{word}:{count}" for word, count in row.drop('song_id').items() if count > 0]
            )
            texts.append(lyrics_text)
            song_ids.append(song_id)
        return song_ids, texts

    def kmeans_cluster(self, embeddings_rb ,n_clusters=5, random_state=42):
        
        embeddings = np.vstack(embeddings_rb['embedding'].values)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(embeddings)

        embeddings_rb['cluster'] = clusters

        silhouette = None
        calinski_score = None

        if len(set(clusters)) > 1 and -1 not in set(clusters):
            silhouette = silhouette_score(embeddings, clusters, metric='cosine')
            calinski_score = calinski_harabasz_score(embeddings, clusters)
        print(f"DBSCAN Completed. Clusters: {len(set(clusters))}")
        if silhouette:
            print(f"Silhouette Score: {silhouette:.4f}")
        if calinski_score:
            print(f"Calinski-Harabasz Index: {calinski_score:.4f}")
        return embeddings_rb, silhouette, calinski_score


    def visualize_cluster(self, clustered_df):
        cluster_emb = np.vstack(clustered_df['embedding'].values)
        clusters = clustered_df['cluster']
        
        num_samples = cluster_emb.shape[0]
        perplexity = min(30, num_samples // 3)

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(cluster_emb)

        plt.figure(figsize=(10, 6))
        for cluster_label in set(clusters):
            cluster_pt = embeddings_2d[clustered_df['cluster'] == cluster_label]
            plt.scatter(cluster_pt[:, 0], cluster_pt[:, 1], label=f"Cluster {cluster_label}")
        plt.title("KMeans Clustering of Lyrics Embeddings")
        plt.legend()
        plt.show()


    def get_tfidf_embeddings(self, max_features=20):
        '''
        Generates TF-IDF embeddings for each song based on the pivoted lyrics data.
        '''

        all_texts = []
        song_ids = []

        for _, row in self.lyrics_pivot.iterrows():
            song_id = row['song_id']


            lyrics_text = " ".join(
                [f"{word} " * int(count) for word, count in row.drop('song_id').items() if count > 1]
            )

            all_texts.append(lyrics_text)
            song_ids.append(song_id)


        #
        # vectorizer = TfidfVectorizer(max_features=500)
        vectorizer = TfidfVectorizer(max_features=max_features) 

        tfidf_matrix = vectorizer.fit_transform(all_texts)

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=song_ids, columns=vectorizer.get_feature_names_out())
        tfidf_df.reset_index(inplace=True)
        tfidf_df.rename(columns={'index': 'song_id'}, inplace=True)

        print("TF-IDF embeddings generated.")
        self.lyrics_df = tfidf_df


    def embeddings_to_pkl(self, file_path, embeddings):
        direc = os.path.dirname(file_path)
        
        if not os.path.exists(direc):
            os.makedirs(direc)
            print(f"Directory created")
        else:
            print(f"Directory exists")
        
        if os.path.exists(file_path):
            print(f"Embeddings already stored in pkl file.")
        else:
            with open(file_path, 'wb') as pkl_file:
                pickle.dump(embeddings, pkl_file)
            print(f"Embeddings Saved to '{file_path}'")
