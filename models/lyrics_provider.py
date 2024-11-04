import sqlite3
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from tqdm import tqdm

class LyricsProvider:

    def __init__(self, lyrics_pivot) -> None:
        self.lyrics_df = None
        self.lyrics_pivot = lyrics_pivot
        # self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # self.model = RobertaModel.from_pretrained("roberta-base")

    # def get_roberta_embeddings(self, batch_size=8):
    #     '''
    #     Converts each row (song) in pivot table to a text format and generates embeddings for each song using RoBERTa.
    #     Processes songs in batches to improve efficiency.
    #     '''
    #     song_embeddings = []
    #     all_texts = []
    #     song_ids = []

    #     for _, row in self.lyrics_pivot.iterrows():
    #         song_id = row['song_id']
            
    #         # Convert the word count data into a text format (e.g., repeating words based on their count)
    #         lyrics_text = " ".join(
    #             [f"{word} " * int(count) for word, count in row.drop('song_id').items() if count > 1]
    #         )
            
    #         # Append text and ID for batching
    #         all_texts.append(lyrics_text)
    #         song_ids.append(song_id)

    #     # Process in batches
    #     for i in tqdm(range(0, len(all_texts), batch_size), desc="Generating Embeddings in Batches"):
    #         batch_texts = all_texts[i:i+batch_size]
    #         batch_ids = song_ids[i:i+batch_size]
            
    #         inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
    #         with torch.no_grad():
    #             outputs = self.model(**inputs)
            
    #         batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Get CLS token embeddings
            
    #         for song_id, embedding in zip(batch_ids, batch_embeddings):
    #             song_embeddings.append({'song_id': song_id, 'embedding': embedding})

    #     embeddings_df = pd.DataFrame(song_embeddings)
    #     return embeddings_df

    # def get_roberta_embedding_for_text(self, text):
    #     '''
    #     Given text, returns CLS embedding using RoBERTa
    #     '''
    #     inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #     cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    #     return cls_embedding

    def get_tfidf_embeddings(self, lyrics_pivot, max_features=20):
        '''
        Generates TF-IDF embeddings for each song based on the pivoted lyrics data.
        '''
        
        all_texts = []
        song_ids = []

        for _, row in lyrics_pivot.iterrows():
            song_id = row['song_id']
            
            
            lyrics_text = " ".join(
                [f"{word} " * int(count) for word, count in row.drop('song_id').items() if count > 1]
            )
            
            all_texts.append(lyrics_text)
            song_ids.append(song_id)

        # Initialize the TF-IDF vectorizer
        # vectorizer = TfidfVectorizer(max_features=500)  # Limit features to improve efficiency
        vectorizer = TfidfVectorizer(max_features=max_features)  # Less features to correctly concatanate and fuse the features
        
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=song_ids, columns=vectorizer.get_feature_names_out())
        tfidf_df.reset_index(inplace=True)
        tfidf_df.rename(columns={'index': 'song_id'}, inplace=True)

        print("TF-IDF embeddings generated.")
        return tfidf_df

