import pandas as pd

# Load datasets
df = pd.read_csv("dataset/drugsCOM_raw.csv")

print(f"\nCombined dataframe shape: {df.shape}")

df.head()

# Count words for review
review_lengths = df['review'].apply(lambda x: len(x.split()))
review_mean_lenth = review_lengths.mean()
review_quantile_95 = review_lengths.quantile(0.95)
review_max_length = review_lengths.max()

print(f"Average length: {review_mean_lenth:.2f}")
print(f"95th percentile: {review_quantile_95:.2f}")
print(f"Max length: {review_max_length}")

# Add a feature to capture the number of words of the review
df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))

# Filter to mantain only reviews <= 256 words 
df_filtered = df[df['review_length'] <= 256].copy()

# Verify how many samples remain
print(f"Original samples: {len(df)}")
print(f"Samples after filtering: {len(df_filtered)}")
print(f"Deleted samples: {len(df) - len(df_filtered)}")

df_filtered['rating'].value_counts()