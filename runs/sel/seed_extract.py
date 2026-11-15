"""Offline seed skeleton extraction. JSONL needs only the standard library.

Parquet input imports pandas lazily; zip input streams Blog Corpus members.
Raw names are used transiently during cleaning and are not written to skeletons.
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import glob
import hashlib
import html
import json
from pathlib import Path
import random
import re
import zipfile

SOURCES = {
    'irc_ubuntu': ('group', 'CC BY 4.0', 'jkkummerfeld/irc_disentangle'),
    'enron': ('group', 'Enron research corpus; content rights not cleared for redistribution', 'corbt/enron-emails'),
    'dailydialog': ('dyad', 'CC BY-NC-SA 4.0; attribution and share-alike apply', 'DailyDialog'),
    'oasst2_val_en': ('assistant', 'Apache-2.0', 'OpenAssistant/oasst2'),
    'wildchat': ('assistant', 'ODC-BY; individual content rights not implied', 'WildChat-1M'),
    'blogs': ('record', 'Non-commercial research only; redistribution not granted', 'Blog Authorship Corpus'),
}
# Names are deliberately unrelated to corpus nicknames and author filenames.
NAMES = ('Ari', 'Bela', 'Cleo', 'Davi', 'Esme', 'Farah', 'Galen', 'Hana',
         'Inez', 'Jules', 'Kiran', 'Luca', 'Mira', 'Nora', 'Omar', 'Pia')
EMAIL = re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
URL = re.compile(r'(?:https?://|www\.)\S+', re.I)
PHONE = re.compile(r'(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-])\d{3}[ .-]\d{4}(?!\w)')
COMMON = set('the and you that this with have what for are was not can how it is to of in on my me we a i do please thanks hello hi yes no okay good right well really sounds great morning evening monday tuesday wednesday thursday friday saturday sunday january february march april may june july august september october november december ubuntu linux windows english assistant user dear best regards sincerely subject from sent reply forwarded christmas internet'.split())


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def split_for(source, conversation):
    n = int(sha([source, str(conversation)])[:8], 16) % 100
    return 'train' if n < 80 else 'dev' if n < 90 else 'audit'


def is_english(text, language=None):
    if language and str(language).casefold() not in ('en', 'english', 'en-us', 'en-gb'):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < .93:
        return False
    words = re.findall(r'[A-Za-z]+', text.lower())
    # Conservative, dependency-free heuristic, including short IRC replies.
    return bool(set(words) & COMMON) or len(words) <= 3


def clean_text(text):
    text = html.unescape(str(text)).replace('\x00', '')
    text = URL.sub('[URL]', text)
    text = EMAIL.sub('[EMAIL]', text)
    text = PHONE.sub('[PHONE]', text)
    return re.sub(r'\s+', ' ', text).strip()


def strip_email(body):
    out = []
    for line in str(body).splitlines():
        if re.match(r'\s*(?:[-_]{2,}.*(?:Original|Forwarded|$)|On .+wrote:|From:|Sent:|Begin forwarded|--\s*$|Best,?\s*$|Regards,?\s*$|Sincerely,?\s*$)', line, re.I):
            break
        if not line.lstrip().startswith('>'):
            out.append(line)
    return '\n'.join(out).strip()


def normalized_subject(subject):
    return re.sub(r'\s+', ' ', re.sub(r'^(?:(?:re|fw|fwd)\s*:\s*)+', '', str(subject).strip(), flags=re.I)).casefold()


def addresses(value):
    if isinstance(value, str):
        return EMAIL.findall(value) or ([value.strip()] if value.strip() else [])
    return [a for v in (value or []) for a in addresses(v) if a]


def conversations(source, rows):
    """Yield (source conversation identity, turns, identity alias groups)."""
    if source == 'irc_ubuntu':
        days = collections.defaultdict(list)
        for row in rows:
            m = re.match(r'\[([^]]+)\]\s*<([^>]+)>\s*(.*)', str(row.get('raw', '')))
            if m:
                days[str(row.get('date', 'unknown'))].append(dict(tid=str(row['id']), speaker=m[2], text=m[3],
                    time=str(row.get('date', ''))+'T'+m[1], reply_to=[str(x) for x in row.get('connections', [])]))
        for date, turns in sorted(days.items()):
            # Sort within a day, never bridge midnight or a missing source row.
            turns.sort(key=lambda x: int(x['tid']))
            chunks = []; current = []
            for t in turns:
                if current and int(t['tid']) != int(current[-1]['tid'])+1:
                    chunks.append(current); current = []
                current.append(t)
            if current: chunks.append(current)
            for ts in chunks:
                yield date, ts, {t['speaker']: [t['speaker']] for t in turns}
    elif source == 'enron':
        threads = collections.defaultdict(list)
        for row in rows:
            sub = normalized_subject(row.get('subject', ''))
            if sub: threads[sub].append(row)
        for sub, emails in sorted(threads.items()):
            emails.sort(key=lambda x: str(x.get('date', '')))
            # Reject broadcast threads before anonymization or window selection.
            recipients = {a.casefold() for r in emails for field in ('to','cc','bcc') for a in addresses(r.get(field)) if a}
            if len(recipients) > 10: continue
            unique = {}; aliases = {}
            for r in emails:
                sender = addresses(r.get('from'))
                if not sender: continue
                body = strip_email(r.get('body', ''))
                if not body: continue
                key = sha([sender[0], body])
                unique.setdefault(key, (r, sender[0], body))
                for a in sender+addresses(r.get('to'))+addresses(r.get('cc'))+addresses(r.get('bcc')):
                    local = a.split('@')[0]
                    aliases[a] = [a, local, local.replace('.', ' ')]
                    aliases[a] += [p for p in re.split(r'[._-]', local) if len(p) > 2 and p.lower() not in COMMON]
            if len(unique) < 3 or len({s for _, s, _ in unique.values()}) < 2: continue
            turns = [dict(tid=sha([sub, r.get('message_id'), s, body])[:16], speaker=s, text=body,
                          time=str(r.get('date', '')), recipients=list(dict.fromkeys(addresses(r.get('to'))+addresses(r.get('cc'))+addresses(r.get('bcc'))))[:6])
                     for r, s, body in unique.values()]
            yield sub, turns, aliases
    elif source == 'dailydialog':
        for row in rows:
            dialog = row.get('dialog', [])
            yield sha(dialog), [dict(tid=str(i), speaker='A' if i%2 == 0 else 'B', text=t) for i,t in enumerate(dialog)], {'A': [], 'B': []}
    elif source == 'wildchat':
        for row in rows:
            if str(row.get('language', 'English')).lower() not in ('en', 'english'): continue
            conv = row.get('conversation', [])
            ts = [dict(tid=str(i), speaker='Assistant' if t.get('role') == 'assistant' else 'User',
                       role=t.get('role'), text=t.get('content', ''), time=str(t.get('timestamp', '')))
                  for i,t in enumerate(conv) if t.get('role') in ('user', 'assistant')]
            yield str(row.get('conversation_id') or row.get('id') or sha(conv)), ts, {'User': [], 'Assistant': []}
    elif source == 'oasst2_val_en':
        trees = collections.defaultdict(dict)
        for row in rows:
            trees[str(row['message_tree_id'])][str(row['message_id'])] = row
        for tree, nodes in sorted(trees.items()):
            parents = {str(r.get('parent_id')) for r in nodes.values()}
            for leaf in sorted(set(nodes)-parents):
                path = []; seen = set(); node = nodes[leaf]
                while node and str(node['message_id']) not in seen:
                    mid = str(node['message_id']); seen.add(mid); path.append(node)
                    node = nodes.get(str(node.get('parent_id')))
                path.reverse()
                if not path or path[0].get('parent_id') is not None: continue
                if any(r.get('lang', 'en') != 'en' for r in path): continue
                ts = [dict(tid=str(r['message_id']), speaker='Assistant' if r['role']=='assistant' else 'User',
                           role='assistant' if r['role']=='assistant' else 'user', text=r.get('text',''), reply_to=[str(r['parent_id'])] if r.get('parent_id') else []) for r in path]
                yield tree, ts, {'User': [], 'Assistant': []}
    elif source == 'blogs':
        for row in rows:
            xml = row.get('xml', '')
            posts = re.findall(r'<date>\s*(.*?)\s*</date>\s*<post>(.*?)</post>', xml, flags=re.S|re.I)
            author = Path(row.get('file', 'unknown')).name.split('.')[0]
            ts = [dict(tid=str(i), speaker='Author', text=re.sub(r'<[^>]+>', ' ', text), time=date.strip()) for i,(date,text) in enumerate(posts)]
            yield author, ts, {'Author': []}
    else:
        raise ValueError('unknown source '+source)


def rename_all(turns, aliases, seed=0):
    """Rename detected people, never arbitrary capitalized words.

    Detection is deliberately limited to speaker aliases, email locals, explicit
    @mentions, and names in greetings/sign-offs. One simultaneous mapping covers
    text and metadata; a finite natural-name pool fails closed on exhaustion.
    """
    rng = random.Random(seed)
    surnames = ('Hayes', 'Morgan', 'Patel', 'Reed', 'Silva', 'Brooks', 'Chen',
                'Rivera', 'Bennett', 'Foster', 'Price', 'Walker', 'Ellis', 'Park',
                'Santos', 'Wells', 'Clarke', 'Lane', 'Shaw', 'Wong')
    bank = list(NAMES); rng.shuffle(bank)
    extended = [first+' '+last for first in NAMES for last in surnames]
    rng.shuffle(extended); bank += extended
    mapping = {}; identities = {}; ambiguous = set(); used = {'Assistant'}
    def allocate():
        while bank:
            name = bank.pop(0)
            if name not in used:
                used.add(name); return name
        raise ValueError('natural name pool exhausted')
    def alias(name, pseudo):
        if not name or name.casefold() in COMMON or len(name)<2: return
        key = name.casefold()
        if key in mapping and mapping[key] != pseudo: ambiguous.add(key)
        else: mapping[key] = pseudo
    people = dict(aliases)
    for t in turns:
        people.setdefault(t['speaker'], [t['speaker']])
        for a in t.get('recipients', []): people.setdefault(a, [a])
    text = '\n'.join(t['text'] for t in turns)
    for a in EMAIL.findall(text): people.setdefault(a, [a])
    for person, names in people.items():
        pseudo = 'Assistant' if person == 'Assistant' else allocate()
        identities[person] = pseudo
        detected = list(names)
        if person not in ('User', 'Author', 'A', 'B', 'Assistant'): detected.append(person)
        if EMAIL.fullmatch(person):
            local = person.split('@')[0]
            detected += [local, local.replace('.', ' ')]
            detected += [part for part in re.split(r'[._-]',local) if len(part)>2]
        for name in detected: alias(name, pseudo)
    mentions = re.findall(r'@([A-Za-z][\w.-]*)', EMAIL.sub('',text))
    greetings = re.findall(r'(?:^|[\n.!?]\s*)(?:Hi|Hello|Dear|Hey)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)(?=[,!:\n])', text)
    signoffs = re.findall(r'(?:Thanks|Thank you|Best|Regards|Cheers|Sincerely)[,!]\s*\n?\s*([A-Z][a-z]+(?: [A-Z][a-z]+)?)(?=\s*(?:[.!]|$|\n))',text)
    for candidate in mentions+greetings+signoffs:
        if candidate.casefold() not in COMMON | {'everyone','team','all','folks','there'} and candidate.casefold() not in mapping:
            alias(candidate, allocate())
    # Ambiguous first names still refer to a person, but cannot safely resolve.
    for key in ambiguous: mapping[key] = allocate()
    pattern = re.compile(r'(?<![\w])(?:'+'|'.join(re.escape(x) for x in sorted(mapping,key=len,reverse=True))+r')(?![\w])', re.I) if mapping else None
    def replace(value):
        return pattern.sub(lambda m:mapping.get(m[0].casefold(), '[PERSON]'),str(value)) if pattern else str(value)
    out = []
    for t in turns:
        item = dict(t, speaker=identities[t['speaker']], text=replace(t['text']))
        if 'recipients' in item:
            item['recipients'] = [identities[x] for x in item['recipients'][:6]]
        out.append(item)
    leftovers = [m[0] for t in out for m in (pattern.finditer(t['text']) if pattern else []) if m[0] not in used]
    if leftovers: raise ValueError('leftover-name scan failed')
    return out, dict(identities=len(identities), aliases=len(mapping), ambiguous_aliases=len(ambiguous), leftover_count=0,
                     mapping_hash=sha(mapping), method='speaker_email_mention_greeting_signoff_only')


def length_samples(stats, carrier, source_turns):
    """Accept natural_stats reports or an explicit {carrier: {turn_chars: [...]}}."""
    benchmarks = {'group': 'ev', 'dyad': 'loc', 'assistant': 'lme', 'record': 'ep'}
    # natural_stats.by_carrier is SYNTHETIC; by_benchmark contains the real pools.
    item = (stats or {}).get('by_benchmark', {}).get(benchmarks[carrier], {})
    if not item:
        item = (stats or {}).get(carrier, {})
    if isinstance(item.get('turn_chars'), list) and item['turn_chars']:
        return item['turn_chars'], 'natural_stats.turn_chars'
    qs = [item.get('chars_'+q) for q in ('p10','p50','p90')]
    if all(isinstance(v,(int,float)) and v>0 for v in qs):
        # Candidate round totals: divide assistant totals across two sides.
        return [max(40,int(v/(2 if carrier=='assistant' else 1))) for v in qs], 'natural_stats.quantiles'
    return [len(clean_text(t['text'])) for t in source_turns if t['text']] or [300], 'source_empirical_fallback'


def extract(source, rows, version='pre-extracted-jsonl', stats=None, seed=31, window_turns=32, min_turns=6, limit=None, license_text=None):
    carrier, license_name, repository = SOURCES[source]
    output = []; seen = set()
    for conv, ts, aliases in conversations(source, rows):
        # Rename once over the complete conversation so overlapping windows,
        # quotations and mentions share the same pseudonyms.
        renamed_ts, audit = rename_all(ts, aliases, int(sha([source,conv])[:8],16))
        renamed_by_tid = {t['tid']:t for t in renamed_ts}
        # Keep contiguous English runs; never silently join across removed turns.
        runs = []; run = []
        for t in ts:
            if is_english(t['text']) and clean_text(t['text']): run.append(t)
            else:
                if run: runs.append(run)
                run = []
        if run: runs.append(run)
        for run in runs:
            for offset in range(0,len(run),window_turns):
                fragment = run[offset:offset+window_turns]
                if len(fragment) < (3 if source=='enron' else min_turns): continue
                if carrier in ('dyad','assistant','group') and len({t['speaker'] for t in fragment}) < 2: continue
                skey = sha([source,conv,[t['tid'] for t in fragment]])
                if skey in seen: continue
                seen.add(skey); rng = random.Random(f'{seed}:{skey}')
                caps, method = length_samples(stats,carrier,fragment)
                renamed = [dict(renamed_by_tid[t['tid']]) for t in fragment]
                for t in renamed:
                    cap = max(40, int(rng.choice(caps)))
                    t['text'] = clean_text(t['text'])[:cap].rsplit(' ',1)[0] if len(clean_text(t['text']))>cap else clean_text(t['text'])
                    t['length_cap'] = cap
                    t['reply_to'] = [str(x) for x in t.get('reply_to',[]) if str(x)!=t['tid']]
                output.append(dict(skel_id='sk_'+skey[:20], source=source, source_conv_id=sha([source,conv]),
                    carrier=carrier, split=split_for(source,conv), turns=renamed,
                    source_version=version, license=license_name, license_text=license_text,
                    license_text_status='supplied_local_text' if license_text else 'not_vendored; verify before scale/release',
                    source_repository=repository, rename_audit=audit, extraction_seed=seed,
                    length_sampling=method, modifications=['English-run filter','consistent pseudonyms','PII placeholders','sampled length caps'],
                    public_release_allowed=False))
                if limit and len(output)>=limit: return output
    return output


def read_rows(paths):
    for pattern in paths:
        for name in sorted(glob.glob(str(Path(pattern).expanduser()))):
            path = Path(name)
            if path.suffix == '.parquet':
                import pandas as pd
                # pandas/Numpy scalars and array columns become ordinary JSON types.
                for row in json.loads(pd.read_parquet(path).to_json(orient='records')): yield row
            elif path.suffix == '.zip':
                with zipfile.ZipFile(path) as z:
                    for member in sorted(z.namelist()):
                        if member.lower().endswith('.xml'):
                            yield dict(file=member,xml=z.read(member).decode('utf-8',errors='replace'))
            else:
                with path.open() as f:
                    for line in f:
                        if line.strip(): yield json.loads(line)


def write_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tmp.replace(path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=list(SOURCES),required=True)
    p.add_argument('--input',nargs='+',required=True); p.add_argument('--out',required=True)
    p.add_argument('--source-version',required=True,help='Corpus revision, shard manifest/version or local snapshot digest')
    p.add_argument('--license-file'); p.add_argument('--natural-stats')
    p.add_argument('--seed',type=int,default=31); p.add_argument('--window-turns',type=int,default=32)
    p.add_argument('--min-turns',type=int,default=6); p.add_argument('--limit',type=int)
    a=p.parse_args(argv)
    if a.window_turns<3 or a.min_turns<3: p.error('window/min turns must be >=3')
    rows=extract(a.source,read_rows(a.input),a.source_version,
                 json.loads(Path(a.natural_stats).read_text()) if a.natural_stats else None,
                 a.seed,a.window_turns,a.min_turns,a.limit,
                 Path(a.license_file).read_text() if a.license_file else None)
    write_jsonl(a.out,rows); print(json.dumps(dict(skeletons=len(rows),source=a.source,out=a.out)))


if __name__=='__main__': main()
