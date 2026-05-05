import pandas as np
import pandas as pd
import matplotlib.pyplot as plt

# load train csv
train_df = pd.read_csv('dataset/drugsComTrain_raw.csv')

# load test csv
test_df = pd.read_csv('dataset/drugsComTest_raw.csv')

print(f'TRAIN DATASET LENGTH: {len(train_df)} \nTEST DATASET LENGTH: {len(test_df)}')

print('TRAIN DATASET:')
train_df.dtypes

print('TEST DATASET:')
test_df.dtypes

print('TRAIN DATASET:') 
train_df.head(5)

print(f'TEST DATASET:')
test_df.head(5)

print(f"TRAIN UNIQUE IDs: {len(train_df['uniqueID'].unique())}/{len(train_df)}")
print(f"TEST UNIQUE IDs: {len(test_df['uniqueID'].unique())}/{len(test_df)}")

test_ids_in_train = train_df[train_df['uniqueID'].isin(test_df['uniqueID'])]
print(f"Num TEST ID's IN TRAIN: {len(test_ids_in_train)}")


# concatenate the dfs
dataset = pd.concat([train_df, test_df], axis = 0)

# save new df as csv
dataset.to_csv("dataset/drugsCOM_raw.csv")

print(f"LEN DATASET: {len(dataset)}")

print(f"Dataset shape: {dataset.shape}\n")
print("Missing values per column:")
print(dataset.isnull().sum())
print(f"\nTotal missing: {dataset.isnull().sum().sum()}")
print(f"Missing (%) per column:")
print((dataset.isnull().sum() / len(dataset) * 100).round(2))

# plot rating distribution
scores = dataset['rating']

plt.figure(figsize=(10, 5))
bars = plt.bar(scores.unique(), scores.value_counts(), color='steelblue', edgecolor='black')

plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Rating Distribution')

# add num samples for each val
for bar in bars: 
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
             f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9)

plt.xticks(scores.unique())
plt.tight_layout()
plt.show()