"""
- user_data_df has user_id, song_id, count
- NEED: mapping of song_id to prediction vector output for test data
- in the user data df only include rows where track_id exists in the test set


"""


import numpy as np
import pandas as pd


class Recommender:

    def __init__(self, user_data_df, predictions_df, debug=False):
        self.debug = debug
        self.user_data_df = user_data_df
        self.predictions_df = predictions_df
        # filter out records that are not in the test dataframe
        self.user_data_df = self.user_data_df[self.user_data_df['song_id'].isin(self.predictions_df['song_id'])]

    def _print_debug(self, message):
        if self.debug:
            print(message)

    def _songs_not_listened_to(self, user):
        """
        Returns a the set of song ids that the user has not listened to.
        """
        listened_to = set(self.user_data_df[self.user_data_df['user_id'] == user]['song_id'])
        return set(self.predictions_df['song_id']) - listened_to

    def _generate_fingerprint(self, user):
        """
        Weighted average prediction vector for the user.
        """
        user_data = self.user_data_df[self.user_data_df['user_id'] == user]
        user_data = user_data.merge(self.predictions_df, on='song_id')
        weighted_prediction = None
        total_count = 0
        for index, row in user_data.iterrows():
            prediction = np.array(row['prediction'])
            count = row['play_count']
            if total_count == 0:
                weighted_prediction = prediction * count
            else:
                weighted_prediction += prediction * count
            total_count += count
        if total_count == 0:
            return None
        return weighted_prediction / total_count

    def recommend_songs(self, user, n=10):
        """
        Returns a table of n song ids to recommend to the user.
        """
        not_listened_to = self._songs_not_listened_to(user)
        predictions_not_listened_to = self.predictions_df[self.predictions_df['song_id'].isin(not_listened_to)]
        fingerprint = self._generate_fingerprint(user)
        if fingerprint is None:
            self._print_debug(f"No fingerprint available for user {user}.")
            return None
        self._print_debug(f"Fingerprint for user {user}: {fingerprint}")
        predictions_not_listened_to['similarity'] = predictions_not_listened_to['prediction'].apply(lambda x: np.dot(np.array(x), fingerprint))
        predictions_not_listened_to = predictions_not_listened_to.sort_values('similarity', ascending=False)
        return predictions_not_listened_to.head(n)

    def recommend_users(self, user, n=10):
        """
        Returns a table of n user ids to recommend to the user.
        """
        user_fingerprints = {}
        user_fingerprint = None
        for uid in self.user_data_df['user_id'].unique():
            if uid == user:
                user_fingerprint = self._generate_fingerprint(uid)
            else:
                fingerprint = self._generate_fingerprint(uid)
                if fingerprint is not None:
                    user_fingerprints[uid] = fingerprint
        if user_fingerprint is None:
            self._print_debug(f"No fingerprint available for user {user}.")
            return None
        self._print_debug(f"Fingerprint for user {user}: {user_fingerprint}")
        similarities = []
        for uid, fingerprint in user_fingerprints.items():
            similarity = np.dot(user_fingerprint, fingerprint)
            similarities.append({'user_id': uid, 'similarity': similarity})
        similarities_df = pd.DataFrame(similarities)
        similarities_df = similarities_df.sort_values('similarity', ascending=False).head(n)
        return similarities_df
