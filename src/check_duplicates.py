from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / '..' / 'impulsive-detector-main' / 'impulsive-detector-main' / 'data' / '03_final_data' / '04_Merged_labeled_transaction.csv'

if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH)
    print('rows:', len(df))
    dup_rows = df.duplicated().sum()
    print('exact duplicate full rows:', dup_rows)
    if dup_rows > 0:
        print('example duplicates:')
        print(df[df.duplicated(keep=False)].head(5).to_string(index=False))

    train, test = train_test_split(df, test_size=0.25, stratify=df['label'], random_state=42)
    print('train size:', len(train), 'test size:', len(test))

    merged = pd.merge(train.assign(_flag='train'), test.assign(_flag='test'), how='inner', on=list(df.columns))
    print('duplicate full-row overlap train/test:', len(merged))
    if len(merged) > 0:
        print('example overlapping rows:')
        print(merged.head(5).to_string(index=False))

    key_cols = ['timestamp', 'category', 'amount', 'metode_pembayaran', 'source', 'hour', 'day_of_week']
    train_keys = train[key_cols].drop_duplicates()
    test_keys = test[key_cols].drop_duplicates()
    overlap = pd.merge(train_keys, test_keys, on=key_cols)
    print('key overlap count:', len(overlap))
    if len(overlap) > 0:
        print('example key overlap:')
        print(overlap.head(5).to_string(index=False))
