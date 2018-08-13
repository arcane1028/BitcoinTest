import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# define csv header
block_hash_header = ["blockID", "block_hash", "block_timestamp", "n_txs"]
transaction_header = ["txID", "blockID", "n_inputs", "n_outputs"]
transaction_in_header = ["txID", "seq", "prev_txID",
                         "prev_output_seq", "addrID", "sum"]
transaction_out_header = ["txID", "seq", "addrID", "sum"]

# read csv
block_hash = pd.read_csv("data/bh.CSV", names=block_hash_header)
print("read block_hash",len(block_hash.index))
transaction = pd.read_csv("data/tx.CSV", names=transaction_header)
print("read transaction", len(transaction.index))
transaction_in = pd.read_csv("data/txin.csv", names=transaction_in_header)
print("read transaction_in", len(transaction_in.index))
transaction_out = pd.read_csv("data/txout.csv", names=transaction_out_header)
print("read transaction_out", len(transaction_out.index))

"""
print(block_hash)
print(transaction)
print(transaction_in)
print(transaction_out)
"""

transaction_in = transaction_in.drop(columns=['prev_txID', 'prev_output_seq'])

# merge
result = pd.merge(transaction, block_hash)
print(result.head())

del block_hash, transaction

# drop block_hash
result = result.drop(columns=['block_hash'])
print(result.head())

# merge
result_in = pd.merge(result, transaction_in)
del transaction_in

print("==result_in==")
print(result_in.head())
print("is nan(false) : ", np.any(np.isnan(result_in)))  # false
print("is finite(true) : ", np.all(np.isfinite(result_in)))  # true
print("size : ", len(result_in.index))

result_in.to_csv("data_in.csv", encoding='utf-8', index=False)
del result_in

# merge
result_out = pd.merge(result, transaction_out)
del transaction_out

print("==result_out==")
print(result_out.head())
print("is nan(false) : ", np.any(np.isnan(result_out)))  # false
print("is finite(true) : ", np.all(np.isfinite(result_out)))  # true
print("size : ", len(result_out.index))

result_out.to_csv("data_out.csv", encoding='utf-8', index=False)
del result_out
print("==End==")

"""
print(np.any(np.isnan(result)))  # false
print(np.all(np.isfinite(result))) # true
del transaction_in

result = pd.merge(result, transaction_out, how='right')
print(result.head())

del transaction_out

result.to_csv("data_in.csv", encoding='utf-8')

"""
