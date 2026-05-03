# 1. Features ki list banaein (6 features)
    features = [[length, dots, hyphens, at_symbol, is_http, digits]]

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# 1. Data load karein (Ensure 'data.csv' is in the same folder)
df = pd.read_csv('data.csv')

# 2. Features aur Target set karein
# Hum 6 features use kar rahe hain: length, dots, hyphens, at_symbol, is_http, digits
X = df[['length', 'dots', 'hyphens', 'at_symbol', 'is_http', 'digits']]
y = df['label']

# 3. Model Train karein
model = RandomForestClassifier()
model.fit(X, y)

# 4. Model ko save karein
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model Successfully Updated with 6 Features!")