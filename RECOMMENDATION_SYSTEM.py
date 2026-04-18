# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds

# =========================================
# 2. LOAD DATASET
# =========================================
# Expected format: user_id, item_id, rating

try:
    data = pd.read_csv("ratings.csv")
except:
    # Sample dataset (fallback)
    data = pd.DataFrame({
        "user_id": [1,1,1,2,2,3,3,4,4,5],
        "item_id": [1,2,3,1,3,2,3,1,2,3],
        "rating":  [5,4,3,4,2,5,4,3,5,4]
    })

print("Dataset Preview:")
print(data.head())

# =========================================
# 3. CREATE USER-ITEM MATRIX
# =========================================
user_item_matrix = data.pivot_table(
    index='user_id',
    columns='item_id',
    values='rating'
).fillna(0)

print("\nUser-Item Matrix:")
print(user_item_matrix)

# Convert to numpy matrix
matrix = user_item_matrix.values

# =========================================
# 4. NORMALIZE DATA (MEAN CENTERING)
# =========================================
user_ratings_mean = np.mean(matrix, axis=1)
matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

# =========================================
# 5. APPLY SVD (MATRIX FACTORIZATION)
# =========================================
U, sigma, Vt = svds(matrix_demeaned, k=2)

sigma = np.diag(sigma)

# Reconstruct matrix
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

pred_df = pd.DataFrame(predicted_ratings,
                       columns=user_item_matrix.columns,
                       index=user_item_matrix.index)

print("\nPredicted Ratings:")
print(pred_df)

# =========================================
# 6. RECOMMENDATION FUNCTION
# =========================================
def recommend_items(user_id, num_recommendations=2):
    
    user_row_number = user_id - 1
    
    sorted_user_predictions = pred_df.iloc[user_row_number].sort_values(ascending=False)
    
    user_data = data[data.user_id == user_id]
    user_items = user_data['item_id'].tolist()
    
    recommendations = []
    
    for item in sorted_user_predictions.index:
        if item not in user_items:
            recommendations.append(item)
        if len(recommendations) == num_recommendations:
            break
    
    return recommendations

# =========================================
# 7. TEST RECOMMENDATION
# =========================================
print("\nRecommendations:")
for user in user_item_matrix.index:
    print(f"User {user} -> Recommended Items:", recommend_items(user))

# =========================================
# 8. EVALUATION (RMSE)
# =========================================
# Flatten original and predicted values
actual = matrix.flatten()
predicted = predicted_ratings.flatten()

rmse = np.sqrt(mean_squared_error(actual, predicted))
print("\nRMSE:", rmse)

# =========================================
# 9. OPTIONAL: SAVE MODEL OUTPUT
# =========================================
pred_df.to_csv("predicted_ratings.csv")

print("\nRecommendation system executed successfully!")
