from SchwabClient import authenticate_client
from JournalEngine import JournalEngine
import json
import os
from datetime import date

def dump_sample_transactions(limit=3):
    try:
        client = authenticate_client()
    except Exception as e:
        print('Unable to create client from token file: ', e)
        return
    try:
        je = JournalEngine(client)
        # Use date objects per Schwab client's expected parameters
        resp = client.get_transactions(
            account_hash=je.account_hash,
            start_date=date(2025, 8, 30),
            end_date=date(2025, 11, 28),
        )
        data = resp.json()
    except Exception as e:
        print('Error fetching transactions: ', e)
        return

    if isinstance(data, list):
        tx_list = data
    elif isinstance(data, dict):
        print('returned dict keys:', list(data.keys()))
        # If there's any list under the top-level keys, pick the first list
        tx_list = []
        for v in data.values():
            if isinstance(v, list):
                tx_list = v
                break
    else:
        print('unrecognized payload type: ', type(data))
        return

    print(f"Found {len(tx_list)} transactions; dumping first {min(limit, len(tx_list))} (sanitized):")
    for idx, entry in enumerate(tx_list[:limit]):
        # redact obvious sensitive fields
        e = dict(entry)
        for k in list(e.keys()):
            if 'account' in k.lower() or 'hash' in k.lower() or 'token' in k.lower():
                e[k] = 'REDACTED'
        print(f"--- Entry {idx+1} ---")
        print(json.dumps(e, indent=2, default=str))

if __name__ == '__main__':
    dump_sample_transactions(limit=3)
from SchwabClient import authenticate_client
from JournalEngine import JournalEngine
import json
import os

def dump_sample_transactions(limit=3):
    try:
        client = authenticate_client()
    except Exception as e:
        print('Unable to create client from token file: ', e)
        return
    try:
        je = JournalEngine(client)
        resp = client.get_transactions(account_hash=je.account_hash, start_date='2025-08-30', end_date='2025-11-28')
        data = resp.json()
    except Exception as e:
        print('Error fetching transactions: ', e)
        return

    if isinstance(data, list):
        tx_list = data
    elif isinstance(data, dict):
        print('returned dict keys:', list(data.keys()))
        # If there's any list under the top-level keys, pick the first list
        tx_list = []
        for v in data.values():
            if isinstance(v, list):
                tx_list = v
                break
    else:
        print('unrecognized payload type: ', type(data))
        return

    print(f"Found {len(tx_list)} transactions; dumping first {min(limit, len(tx_list))} (sanitized):")
    for idx, entry in enumerate(tx_list[:limit]):
        # redact obvious sensitive fields
        e = dict(entry)
        for k in list(e.keys()):
            if 'account' in k.lower() or 'hash' in k.lower() or 'token' in k.lower():
                e[k] = 'REDACTED'
        print(f"--- Entry {idx+1} ---")
        print(json.dumps(e, indent=2, default=str))

if __name__ == '__main__':
    dump_sample_transactions(limit=3)

from SchwabClient import SchwabClient
from JournalEngine import JournalEngine
client = SchwabClient.load_from_token('schwab_token.json')  # adapt to your client import pattern
je = JournalEngine(client)
resp = client.get_transactions(account_hash=je.account_hash, start_date='2025-08-30', end_date='2025-11-28')
data = resp.json()
# If the root is list or dict, normalize to list:
if isinstance(data, list):
    tx_list = data
elif isinstance(data, dict):
    # print possible keys and the first 3 entries if any exist
    print('returned dict keys:', list(data.keys()))
    # print a snippet
    for k,v in data.items():
        if isinstance(v, list):
            print('sample entries for', k)
            for entry in v[:3]:
                print(entry)
else:
    print('unrecognized payload', type(data))
PY