import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

recipes = pd.read_csv('./data/RAW_recipes.csv')
interactions = pd.read_csv('./data/RAW_interactions.csv')

print(recipes['name'][0])

ratings = interactions[['user_id', 'recipe_id', 'rating']].copy()

user_enc = LabelEncoder()
item_enc = LabelEncoder()

ratings['user_idx'] = user_enc.fit_transform(ratings['user_id'])
ratings['item_idx'] = item_enc.fit_transform(ratings['recipe_id'])

num_users = ratings['user_idx'].nunique()
num_items = ratings['item_idx'].nunique()

sparse_matrix = csr_matrix((ratings['rating'].astype(float), (ratings['user_idx'], ratings['item_idx'])), shape=(num_users, num_items))

item_similarity = cosine_similarity(sparse_matrix.T, dense_output=False)

id_to_name = dict(zip(recipes['id'], recipes['name']))

def recommend(recipe_id, n_recipes=5):
    recipe_idx = item_enc.transform([recipe_id])[0]

    scores = item_similarity[recipe_idx].toarray().flatten()
    similar_idx = scores.argsort()[::-1][1:n_recipes+1]
    similar_id = item_enc.inverse_transform(similar_idx)

    recipe_names = [(id, id_to_name.get(id, 'Unknown')) for id in similar_id]
    return recipe_names

print(recommend(137739))


