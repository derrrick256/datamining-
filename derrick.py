
# FULL ASSOCIATION RULE MINING PIPELINE (APRiori)
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
# PART A: DATA PREPARATION
# Dataset
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
# Loading
df = pd.DataFrame(data)
# Converting
df['Items'] = df['Items'].apply(lambda x: x.split(', '))
# Create list of all unique items
all_items = sorted(list({item for sublist in df['Items'] for item in sublist}))
# One-hot encoding
encoded_df = pd.DataFrame(0, index=df.Transaction_ID, columns=all_items)
for idx, items in zip(df.Transaction_ID, df['Items']):
    encoded_df.loc[idx, items] = 1
print("\n--- ONE HOT ENCODED DATA ---")
print(encoded_df)
# PART B: APRIORI ALGORITHM
# Apply Apriori
frequent_itemsets = apriori(encoded_df, min_support=0.2, use_colnames=True)
print("\n--- FREQUENT ITEMSETS (Support >= 0.2) ---")
print(frequent_itemsets)
# Generate association rules
rules = association_rules(frequent_itemsets, 
                          metric="confidence", 
                          min_threshold=0.5)

rules = rules[['antecedents','consequents','support','confidence','lift']]

print("\n--- ASSOCIATION RULES (Conf >= 0.5) ---")
print(rules)
# PART C: INTERPRETATION (TOP RULES BY LIFT)
# Sort rules by lift (descending)
strongest_rules = rules.sort_values(by='lift', ascending=False).head(3)
print("\n--- TOP 3 STRONGEST RULES (BY LIFT) ---")
print(strongest_rules)