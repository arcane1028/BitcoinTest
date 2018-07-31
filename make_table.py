
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# define csv header
block_hash_header = ["blockID", "block_hash", "block_timestamp", "n_txs"]
transaction_header = ["txID", "blockID", "n_inputs", "n_outputs"]
transaction_in_header = ["txID", "input_seq", "prev_txID",
                         "prev_output_seq", "addrID", "sum"]
transaction_out_header = ["txID", "output_seq", "addrID", "sum"]

# read csv
block_hash = pd.read_csv("data/bh.CSV", names=block_hash_header)
print("read block_hash")
transaction = pd.read_csv("data/tx.CSV", names=transaction_header)
print("read transaction")
transaction_in = pd.read_csv("data/txin.csv", names=transaction_in_header)
print("read transaction_in")
transaction_out = pd.read_csv("data/txout.csv", names=transaction_out_header)
print("read transaction_out")

"""
print(block_hash)
print(transaction)
print(transaction_in)
print(transaction_out)
"""
# merge
result = pd.merge(transaction, block_hash)
print(result.head())

del block_hash, transaction

# drop block_hash
result = result.drop(columns=['block_hash'])
print(result.head())

# merge
result = pd.merge(result, transaction_in, how='right')
print(result.head())

del transaction_in

result = pd.merge(result, transaction_out, how='right')
print(result.head())

del transaction_out

result.to_csv("data.csv", encoding='utf-8')
