"""
- user_data_df has user_id, song_id, count
- NEED: mapping of song_id to prediction vector output for test data
- in the user data df only include rows where track_id exists in the test set


"""


import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


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
        if user_data.empty:
            return None
        counts = user_data['play_count'].values.reshape(-1, 1)
        predictions = np.stack(user_data['prediction'].values)
        weighted_predictions = predictions * counts
        total_count = counts.sum()
        return weighted_predictions.sum(axis=0) / total_count

    def recommend(self, user, n=10, omit_listened_to=False):
        """
        Returns a table of n song ids to recommend to the user.
        """
        if omit_listened_to:
            not_listened_to = self._songs_not_listened_to(user)
            predictions = self.predictions_df[self.predictions_df['song_id'].isin(not_listened_to)]
        else:
            predictions = self.predictions_df
        fingerprint = self._generate_fingerprint(user)
        if fingerprint is None:
            self._print_debug(f"No fingerprint available for user {user}.")
            return None
        self._print_debug(f"Fingerprint for user {user}: {fingerprint}")
        predictions['similarity'] = predictions['prediction'].apply(lambda x: np.dot(np.array(x), fingerprint))
        predictions = predictions.sort_values('similarity', ascending=False)
        return predictions.head(n)

    def similar_users(self, user, n=10):
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


class RecommenderEvaluator:
    """
    This class evaluates a recommender system using baseline metrics:
    - Mean Average Precision (mAP)
    - Area under ROC curve (AUC)
    """

    def __init__(self, recommender):
        self.recommender = recommender

    def evaluate(self, n=500):
        mAP = self._mean_average_precision(n=n)
        # AUC = self._area_under_roc_curve(n=n)
        # return {'mAP': mAP, 'AUC': AUC}
        return {'mAP': mAP}

    def _mean_average_precision(self, n=500):
        users = self.recommender.user_data_df['user_id'].unique()
        average_precisions = []
        for user in tqdm(users, desc=f"Computing mAP with top {n} recommendations per user"):
            recommended = self.recommender.recommend(user, n, omit_listened_to=False)['song_id'].tolist()
            relevant = set(self.recommender.user_data_df[self.recommender.user_data_df['user_id'] == user]['song_id'])
            if not relevant:
                continue
            score = 0.0
            num_hits = 0
            for i, song_id in enumerate(recommended):
                if song_id in relevant:
                    num_hits += 1
                    score += num_hits / (i + 1)
            if num_hits > 0:
                average_precisions.append(score / min(len(relevant), 10))
        return np.mean(average_precisions) if average_precisions else 0.0

    # def _area_under_roc_curve(self, n=500):
    #     users = self.recommender.user_data_df['user_id'].unique()
    #     auc_scores = []
    #     for user in tqdm(users, desc=f"Computing AUC with top {n} recommendations per user"):
    #         recommended = self.recommender.recommend(user, n, omit_listened_to=False)
    #         if recommended is None:
    #             continue
    #         relevant = set(self.recommender.user_data_df[self.recommender.user_data_df['user_id'] == user]['song_id'])
    #         all_songs = self.recommender.predictions_df['song_id']
    #         y_true = all_songs.isin(relevant).astype(int)
    #         y_scores = self.recommender.predictions_df['prediction'].apply(lambda x: np.dot(np.array(x), self.recommender._generate_fingerprint(user) if self.recommender._generate_fingerprint(user) is not None else np.zeros_like(x)))
    #         if len(set(y_true)) < 2:
    #             continue
    #         auc = roc_auc_score(y_true, y_scores)
    #         auc_scores.append(auc)
    #     return np.mean(auc_scores) if auc_scores else 0.0
