"""Compare judge wordings on known positives and negatives."""
import sys
from gen import Pool
rp = Pool("Qwen3.8-27B-FP8@18301"); gp = Pool("gemma-4-31b-it@18321")
resp = "Based on the chats, here are items:\n1. Treating NaN values as critical failures that stop the pipeline.\n2. Mapping API calls to GL accounts.\n3. The specific error code mentioned is **429** (Too Many Requests)."
cases = [("I decided that the data validator must treat any 'Not a Number' values as critical failures rather than warnings.", True), ("429 errors", True), ("The General Ledger now syncs with the source system in hourly batches.", False), ("The office moved to Denver.", False)]
V = {"old": "Question: what were they\nResponse: {r}\nRequirement: the response must contain this fact, in any wording: {x}\nIs the requirement met? Answer YES or NO only.",
     "fact_last": "Response:\n{r}\n\nFact: {x}\n\nDoes the response state this fact or its main point, in any wording? Answer YES or NO.",
     "yesno_lower": "Here is a response.\n{r}\n\nDoes it mention this? {x}\nAnswer yes or no."}
for n, p in (("27B", rp), ("gemma", gp)):
    for v, t in V.items():
        for mt in (8, 40):
            print(n, v, mt, [(p.chat(t.format(r=resp, x=x), max_tokens=mt, temperature=0)[0][:30], want) for x, want in cases])
