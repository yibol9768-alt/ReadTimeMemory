"""Seed-only JSON/retry protocol over the stable world3 model transports."""
import copy
import json
import re

from world3 import ModelFamilies


def parse_response(text, task):
    """Accept a complete object/list with fences or surrounding prose.

    Never repair incomplete JSON or recover a nested object from a truncated
    parent: doing so could discard the other facts or a judge's abstention.
    """
    text = re.sub(r'^\s*<think>.*?</think>\s*', '', text, flags=re.S)
    decoder = json.JSONDecoder()
    for m in re.finditer(r'[\[{]', text):
        prefix = text[:m.start()]
        # Opening delimiters inside a broken parent are not independent replies.
        if '{' in prefix or '[' in prefix: continue
        try: value, _ = decoder.raw_decode(text[m.start():])
        except ValueError: continue
        if isinstance(value, list):
            if task == 'render': return {'messages': value}
            if task == 'seed_plan': return {'facts': value}
            if len(value) == 1 and isinstance(value[0], dict): return value[0]
        if isinstance(value, dict):
            if task == 'render' and 'id' in value: return {'messages': [value]}
            return value
    raise ValueError('no complete JSON object/list')


class SeedModels(ModelFamilies):
    """Keep transport and family identity; isolate seed schemas from world edits."""
    def ask(self, family, instructions, payload, validator=None, row_validator=None, failure_collector=None):
        family %= 3
        request = copy.deepcopy(payload)
        failures = []
        for attempt in range(3):
            if failures: request['retry_feedback'] = failures[-1]
            with self.lock: self.calls[f'family{family}:{payload["task"]}'] += 1
            try:
                raw = self.pools[family].chat(instructions+'\nPAYLOAD\n'+json.dumps(request),
                    max_tokens=6000, temperature=.6 if payload['task']=='render' else 0)[0]
                result = parse_response(raw, payload['task'])
                reasons = []
                if payload['task'] == 'render':
                    specs = payload['messages']; rows = result.get('messages')
                    if len(specs)!=1: raise ValueError('seed renderer requires exactly one message per call')
                    if not isinstance(rows,list) or len(rows)!=1 or not isinstance(rows[0],dict) or rows[0].get('id')!=specs[0]['id']:
                        raise ValueError('return exactly one JSON message with the requested id')
                    if row_validator and not row_validator(rows[0],specs[0]):
                        reasons = rows[0].pop('_validation_failures',[]) or ['message validation rejected']
                if reasons: raise ValueError('; '.join(reasons))
                if validator and not validator(result): raise ValueError('schema or semantic validation rejected response')
                return result
            except (ValueError,TypeError,KeyError,AttributeError,OSError) as exc:
                reason = str(exc)
                failures.append(dict(attempt=attempt+1,reasons=[reason],instruction='Fix these concrete failures and return complete JSON only.'))
                if failure_collector is not None:
                    for spec in payload.get('messages',[]): failure_collector.setdefault(spec['id'],[]).append(failures[-1])
        raise ValueError(f'{payload["task"]} exhausted 3 attempts: {failures[-1]["reasons"]}')
