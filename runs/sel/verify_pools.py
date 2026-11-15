"""Check pool_s23 against the frozen POOL_S23_MD5.json before any group is generated (audit 11-02). Exit 2 on mismatch.
usage: python3 verify_pools.py [qid_file.json ...]   (default: every frozen question)"""
import hashlib, json, sys
H = json.load(open("POOL_S23_MD5.json"))
ids = [q["question_id"] for f in sys.argv[1:] for q in json.load(open(f))] if len(sys.argv) > 1 else list(H)
bad = [i for i in ids if i not in H or hashlib.md5(open(f"pool_s23/{i}.json", "rb").read()).hexdigest() != H[i]]
print("verify_pools:", len(ids), "checked,", len(bad), "mismatched", bad[:5]); sys.exit(2 if bad else 0)
