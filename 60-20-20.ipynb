# initialize training sets

cross_validation_ratio = 0.4

train_mask = np.random.rand(len(df)) > cross_validation_ratio
val_mask = np.random.rand(len(df[~train_mask])) > 0.5

train_df = df[train_mask]
val_df = (df[~train_mask])[val_mask]
test_df = (df[~train_mask])[~val_mask]

print("Training: {} values, {}% of dataset.".format(len(train_df),len(train_df)/len(df)))
print("Validation: {} values, {}% of dataset.".format(len(val_df),len(val_df)/len(df)))
print("Testing: {} values, {}% of dataset.".format(len(test_df),len(test_df)/len(df)))

x_train = train_df['tokenized_txt'].values
x_val = val_df['tokenized_txt'].values
y_train = train_df['y_expected'].values
y_val = val_df['y_expected'].values

train_batch_mask = np.random.rand(len(train_df)) > 0.9
val_batch_mask = np.random.rand(len(val_df)) > 0.9

# create batches for quicker testing

#x_train = (train_df[train_batch_mask])['tokenized_txt'].values
#x_val = (val_df[val_batch_mask])['tokenized_txt'].values
#y_train = (train_df[train_batch_mask])['y_expected'].values
#y_val = (val_df[val_batch_mask])['y_expected'].values

#print("Training Batch: {} values.".format(len(x_train)))
#print("Validation Batch: {} values.".format(len(x_val)))

# NOTE: Pandas' dataframe.values occasionally returns entries with NaN values, even when the entries had good numerical values in the actual dataframe.
# I don't have a fix for this, so when iterating through entries for training, use the following at the beginning of each loop to skip over NaN values:
#
# if x_train[i][0][0] != x_train[i][0][0]:
#     continue
#
# This is what solved the issues I've been having.
