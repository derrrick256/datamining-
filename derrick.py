# MUWULYA DERRICK

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# PART A: DATA PREPARATION

# Creating transaction dataset
data = {
    'Transaction_ID': [1,2,3,4,5,6,7,8,9,10],
    'Items': [
        "Bread, Milk, Eggs",
        "Bread, Butter",
        "Milk, Diapers, Beer",
        "Bread, Milk, Butter",
        "Milk, Diapers, Bread",
        "Beer, Diapers",
        "Bread, Milk, Eggs, Butter",
        "Eggs, Milk",
        "Bread, Diapers, Beer",
        "Milk, Butter"
    ]
}

# Loading into DataFrame
df = pd.DataFrame(data)

print("\ ORIGINAL TRANSACTION DATASET")
print(df)

# Converting item strings into lists
df['Items'] = df['Items'].apply(lambda x: [item.strip() for item in x.split(',')])

# Geting unique items
all_items = sorted({item for sublist in df['Items'] for item in sublist})

# One-hot encode the transactions
encoded_df = pd.DataFrame(0, index=df['Transaction_ID'], columns=all_items)

for tid, items in zip(df['Transaction_ID'], df['Items']):
    encoded_df.loc[tid, items] = 1

print("\n ONE-HOT ENCODED DATASET")
print(encoded_df)

# PART B: APRIORI ALGORITHM
# Applying Apriori
frequent_itemsets = apriori(
    encoded_df,
    min_support=0.2,
    use_colnames=True
)

print("\n FREQUENT ITEMSETS (Support ≥ 0.2)")
print(frequent_itemsets)

# Generating association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

# Selecting required columns
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

print("\n--- ASSOCIATION RULES (Confidence ≥ 0.5) ---")
print(rules)


