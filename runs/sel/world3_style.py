"""Label-blind carrier prompts, fact anchors and offline length matching.

Mock prose is a plumbing fixture, never evidence of real model naturalness.
"""
import hashlib
import math
import random
import re
import unicodedata
from functools import lru_cache

ROUND_CHARS = {'group': (330, 390), 'dyad': (270, 335),
               'assistant': (2770, 2870), 'record': (2000, 2670)}
ASSISTANT_PROMPT = ('Could you help me read through these notes? Please keep the dates, '
                    'people and qualifications clear as you explain them in everyday language. '
                    'I am trying to understand the wording before I ask anything more specific about it.')
COMMENTARY = (
    'I would read the wording carefully before drawing a conclusion from it. ',
    'The scope of a statement matters as much as the words used to describe it. ',
    'It helps to distinguish an explanation of the entry from a change to the entry itself. ',
    'A careful reading can keep the subject of the statement separate from the person reporting it. ',
    'For that reason, I would return to the original phrasing when checking an interpretation. ',
    'An explanation should make the wording easier to follow without extending what it says. ',
    'The same care applies when reading a date, a qualification, or a description of a decision. ',
    'Keeping those parts in view makes it easier to follow the passage as a whole. ',
)
SHORT = ('I am reading this with care. ', 'It helps to keep the words in view. ',
         'I can follow the note as it is. ', 'Let us read each part in turn. ')
TOPICS = {
    'software': 'release notes', 'hospital': 'ward notes', 'travel': 'travel plans',
    'laboratory': 'lab notebook', 'restaurant': 'service notes', 'school': 'club discussion',
}
FILLER_OPEN = ('Could we slow down a little with the {topic}?',
    'Thanks for keeping the {topic} readable.', 'How would you phrase this part of the {topic}?',
    'Can we take the {topic} one point at a time?', 'I am still reading through the {topic}.',
    'Would a little more space help with the {topic}?', 'Let me catch up with the {topic}.',
    'I appreciate the care with the {topic}.', 'Could someone talk me through the {topic}?',
    'I would like a moment to think about the {topic}.')
FILLER_CLOSE = ('There is no rush to reply.', 'I am happy to listen for a bit.',
    'A plain explanation would help.', 'Please keep going when you are ready.',
    'A quick question might be easier than a long explanation.', 'I can wait while everyone reads.',
    'Thanks for bearing with me.', 'It helps to hear another voice.',
    'Could we keep the wording simple?', 'Let us leave room for questions.')

# Explicit alternatives, never a fuzzy stemmer. Keep state vocabularies disjoint:
# a minimal-state intervention must not become an equivalent anchor.
STATUS_SYNONYMS = {
    'proposed': ('proposed', 'suggested', 'floated'),
    'approved': ('approved', 'signed off', 'greenlit'),
    'vetoed': ('vetoed', 'blocked', 'rejected'),
    'executed': ('executed', 'ran', 'run', 'passed through'),
    'rolled_back': ('rolled back', 'rolled it back', 'reverted'),
    'draft': ('draft', 'drafted'), 'rostered': ('rostered', 'put on the roster'),
    'off_shift': ('off shift', 'taken off the shift'),
    'held': ('held', 'on hold'), 'ticketed': ('ticketed', 'tickets issued'),
    'cancelled': ('cancelled', 'canceled', 'called off'), 'expired': ('expired', 'lapsed'),
    'received': ('received', 'arrived'), 'screened': ('screened', 'screening completed'),
    'released': ('released', 'cleared for release'),
    'quarantined': ('quarantined', 'placed in quarantine'),
    'ordered': ('ordered', 'order placed'), 'stocked': ('stocked', 'put into stock'),
    'consumed': ('consumed', 'used up'), 'recalled': ('recalled', 'recall issued'),
    'nominated': ('nominated', 'put forward'), 'elected': ('elected', 'voted in'),
    'active': ('active', 'serving in the role'), 'resigned': ('resigned', 'stepped down'),
    'built': ('built', 'build completed'), 'tested': ('tested', 'tests completed'),
    'published': ('published', 'released', 'shipped', 'went out'),
    'yanked': ('yanked', 'pulled', 'withdrawn'), 'failed': ('failed', 'did not pass'),
}
SOCIAL_TURNS = ('+1', 'Thanks @{name}.', 'On it.', 'Following along.',
                'Thanks for the note.', 'Got it.', 'Reading now.', 'Appreciate it.')


@lru_cache(maxsize=16384)
def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    # Join hot-fix as well as hot_fix/hot fix, but keep word boundaries elsewhere.
    text = re.sub(r'hot[\s_\-‐‑–—]*fix', 'hotfix', text)
    text = text.replace('_', ' ')
    text = re.sub(r'[^\w]+', ' ', text).strip()
    # Deliberately small inflection table, not fuzzy substring matching.
    forms = {'published':'publish', 'publishes':'publish', 'publishing':'publish',
             'executed':'execute', 'executes':'execute', 'executing':'execute',
             'approved':'approve', 'approves':'approve', 'approving':'approve',
             'proposed':'propose', 'proposes':'propose', 'proposing':'propose',
             'vetoes':'veto', 'vetoed':'veto', 'yanked':'yank',
             'versions':'version', 'dependencies':'dependency', 'services':'service',
             'reviews':'review', 'updates':'update', 'constraints':'constraint',
             'components':'component', 'tests':'test', 'policies':'policy'}
    return ' '.join(forms.get(word, word) for word in text.split())


def anchors_for(f):
    # Qualifier binding belongs to the independent semantic verifiers.
    spans = [f[k] for k in ('entity', 'value', 'status')]
    policy = f.get('natural_surface')
    if isinstance(policy, dict) and policy.get('holder_anchor'):
        spans.append(f['holder'])
    return list(dict.fromkeys(spans))


def signature_for(f):
    # Stronger local leakage guard; reduced required anchors alone cannot
    # distinguish two versions with the same entity, value and state.
    spans=[f[k] for k in ('entity','attr','value','status','event_time')]
    spans.append(f"{f['holder']}'s {f['scope']}")
    spans += [f'{k} was {v}' for k,v in sorted(f['condition'].items())]
    if not f['condition']:
        # Absence of a qualifier cannot be detected by positive span matching.
        # Leave implicit universal assertions to the semantic verifiers.
        spans.append('in all settings')
    policy=f.get('natural_surface')
    if not isinstance(policy,dict) or policy.get('update'):
        spans.append(f['proposal'])
    return spans



@lru_cache(maxsize=65536)
def anchor_present(span, text):
    normalized = ' ' + normalize(text) + ' '
    date = re.fullmatch(r'(\d{4})-(\d\d)-(\d\d)', span)
    if date:
        import calendar
        year, month, day = map(int, date.groups())
        if not 1 <= month <= 12 or not 1 <= day <= 31:
            return False
        # A supplied conflicting year must not match the yearless alternative.
        variants = [span, f'{year}/{month:02}/{day:02}']
        for name in (calendar.month_name[month], calendar.month_abbr[month]):
            for suffix in ('', 'st', 'nd', 'rd', 'th'):
                d = f'{day}{suffix}'
                for body in (f'{name} {d}', f'{d} {name}'):
                    pattern = r'(?<!\w)' + re.escape(normalize(body)) + r'(?: (\d{4}))?(?!\w)'
                    for match in re.finditer(pattern, normalize(text)):
                        if match[1] is None or match[1] == str(year):
                            return True
        return any(' ' + normalize(v) + ' ' in normalized for v in variants)
    variants = STATUS_SYNONYMS.get(span, (span,))
    return any(' ' + normalize(variant) + ' ' in normalized for variant in variants)


def anchors_present(spans, text):
    return all(anchor_present(span, text) for span in spans)


def leaks_anchors(forbidden, text):
    # Attribute/date/status words shared by two assigned assertions must not
    # manufacture a third assertion merely by pooling the entire transcript.
    # The semantic verifier checks their binding; deterministic leakage checks
    # apply to each paragraph/turn. Prompts keep each atom in one such unit.
    units=[re.sub(r'\b(?:it )?(?:replaced|superseded)\b[^.!?\n]*[.!?]?', '', s, flags=re.I)
           for s in text.splitlines() if s.strip()]
    return any(anchors_present(a,unit) for unit in units for a in forbidden)


@lru_cache(maxsize=32768)
def collapse_whitespace(text):
    return re.sub(r'\s+',' ',text).strip()


def atom_present(atom, text):
    if 'evidence_spans' in atom:
        collapsed=collapse_whitespace(text)
        return any(collapse_whitespace(span) in collapsed for span in atom['evidence_spans'])
    if atom['statement'] in text:
        return True
    return bool(atom.get('anchors')) and anchors_present(atom['anchors'],text)


def prefix(m):
    return f"[{m['channel']}/{m['thread']}] {m['speaker']}: "


def mock_prose(start, target, key, short=False):
    rng=random.Random(hashlib.sha256(key.encode()).hexdigest())
    text=start.rstrip()
    if text:text+=' '
    bank=SHORT if short else COMMENTARY
    while len(text)<target:text+=rng.choice(bank)
    end=text[:target].rstrip()
    if len(text)>target and end and end[-1] not in '.!?':
        end=end.rsplit(' ',1)[0].rstrip(' ,;:')+'.'
    return end.ljust(target)


def configure(w, case, messages):
    rng=random.Random(f"carrier-length-v5:{w['seed']}:{w['world']}:{case['id']}")
    carrier=case['carrier'];low,high=ROUND_CHARS[carrier]
    desired=rng.randint(low,high)
    if carrier in ('group','dyad'):
        desired=max(desired,max(len(prefix(m))+len(m['core'])+160 for m in messages)+18)
    if carrier=='assistant':
        desired=max(desired,max((len(prefix(m))+len(m['core'])+1950 for m in messages if m['speaker']!='Assistant'),default=desired))
    byid={f['id']:f for f in w['facts']}
    for i,m in enumerate(messages):
        m['length_contract']='carrier_v8'
        m['round_chars']=desired
        m['anchor_sets']={fid:anchors_for(byid[fid]) for fid in m['atoms']}
        # Include forbidden signatures, never other atoms' statements, in local
        # validation only. They are not sent to the renderer.
        m['draft']=m['core'] or filler(w,m,int(m['id'][1:]))
        if m.get('meaningful_confirmation'):
            m['draft']+=' I approve this decision and commit to carrying it out as recorded.' if m['confirmation_kind']=='approval_commitment' else ' I confirm that this decision was carried out as recorded.'
        if carrier=='assistant' and m['speaker']!='Assistant':
            m['text_chars']=max(len(m['draft'])+120,250)
            m['reply_chars']=max(1800,desired-18-len(prefix(m))-m['text_chars'])
        else:
            overhead=18+len(prefix(m))
            if m['speaker']=='Assistant':overhead+=len(ASSISTANT_PROMPT)+len(f"[{m['channel']}/{m['thread']}] ")
            m['text_chars']=max(len(m['draft'])+65,desired-overhead)
        m['max_chars']=math.ceil(2.5*m['text_chars'])-1
        m['user_speaker']=case['asker']
        m['participants']=list(w['team'][:2] if carrier=='dyad' else w['team'])


def filler(w,m,i):
    """Combinatorial, topic-aware questions and social turns, no ledger claims."""
    topic=TOPICS[w['domain']]
    # Sentence combinations are unique up to 10,000 empty rounds per world.
    a=i%10;b=(i//10)%10;c=(i//100)%10;d=(i//1000)%10
    return (' '.join([FILLER_OPEN[a].format(topic=topic),FILLER_CLOSE[b],
                      FILLER_OPEN[c].format(topic='wording'),FILLER_CLOSE[d]]))


def mock_message(m):
    if m.get('render_side'):
        return dict(id=m['id'], text=mock_prose(m.get('draft',''),m['text_chars'],m.get('style_key',m['id'])))
    if not m.get('length_contract'):
        return {'id':m['id'],'text':m['draft']}
    carrier=m['channel'];short=carrier not in ('record','assistant')
    start=m['draft'];names=m.get('participants',[m['speaker']])
    if carrier=='record':
        start=f"{m['speaker']} noted the following in the notebook. "+start
    row={'id':m['id'],'text':mock_prose(start,max(m['text_chars'],len(start)+10),m['id'],short)}
    if 'reply_chars' in m:row['assistant_reply']=mock_prose('',m['reply_chars'],m['id']+':reply')
    if m.get('participants') and carrier != 'record':
        if carrier == 'assistant':
            user=m.get('user_speaker',names[0]) if m['speaker']=='Assistant' else m['speaker']
            row['turns']=[dict(speaker=user,text=ASSISTANT_PROMPT if m['speaker']=='Assistant' else row['text']),
                          dict(speaker='Assistant',text=row['text'] if m['speaker']=='Assistant' else row['assistant_reply'])]
        else:
            others=[n for n in names if n!=m['speaker']][:1 if carrier=='dyad' else 2]
            row['turns']=[dict(speaker=m['speaker'],text=row['text'])]+[
                dict(speaker=n,text='Thanks, I am reading along.') for n in others]
            assemble_turns(row,m)
            overhead=len(row['text'])-len(row['turns'][0]['text'])
            row['turns'][0]['text']=mock_prose(start,max(len(start)+10,m['text_chars']-overhead),m['id'],short)
            assemble_turns(row,m)
    return row


def assemble_turns(row, m):
    """Format structured speech ourselves; prose is never parsed for names."""
    turns = row.get('turns')
    if not isinstance(turns, list):
        return
    if not all(isinstance(t, dict) and isinstance(t.get('speaker'), str)
               and isinstance(t.get('text'), str) and t['text'].strip() for t in turns):
        return
    carrier = 'group' if m['channel'] not in ('dyad','record','assistant') else m['channel']
    if carrier in ('group', 'dyad'):
        names = m.get('participants', [])
        lines = []
        for i, turn in enumerate(turns):
            mention = ''
            if carrier == 'group' and names:
                other = next((n for n in names if n != turn['speaker']), names[0])
                mention = f'@{other} '
            lines.append(f"{turn['speaker']}: {mention}{turn['text']}")
        row['text'] = '\n'.join(lines)
        if carrier == 'group':
            row['text'] = f"[thread {m.get('thread','discussion')}] " + row['text']
    elif carrier == 'assistant' and len(turns) == 2:
        if m['speaker'] == 'Assistant':
            row['user_text'], row['text'] = turns[0]['text'], turns[1]['text']
        else:
            row['text'], row['assistant_reply'] = turns[0]['text'], turns[1]['text']


def message_failures(row, m, forbidden=()):
    assemble_turns(row, m)
    reasons = []
    text = row.get('text')
    if not isinstance(text, str) or not text.strip():
        return ['missing text']
    if m.get('anchor_sets') is not None:
        reasons += [f"missing anchor {span!r}" for spans in m['anchor_sets'].values()
                    for span in spans if not anchor_present(span, text)]
        if leaks_anchors(forbidden, text): reasons.append('unassigned fact signature')
    elif m.get('core') and m['core'] not in text:
        reasons.append('missing source core')
    if not m.get('length_contract'):
        if len(text) > m.get('max_chars',1000): reasons.append('text too long')
        return reasons
    carrier = m['channel']
    if carrier != 'record' and m.get('participants'):
        turns = row.get('turns')
        if not isinstance(turns, list) or not all(isinstance(t,dict) and
            isinstance(t.get('speaker'),str) and isinstance(t.get('text'),str) and t['text'].strip() for t in turns):
            reasons.append('need JSON turns with speaker and nonempty text')
        else:
            speakers = {t['speaker'] for t in turns}
            if carrier == 'assistant':
                user = m.get('user_speaker',m['participants'][0]) if m['speaker']=='Assistant' else m['speaker']
                if [t['speaker'] for t in turns] != [user,'Assistant']:
                    reasons.append(f'need user {user!r} then Assistant (2 speakers)')
            elif carrier == 'dyad':
                if speakers != set(m['participants']): reasons.append('need exactly 2 participant speakers')
            elif not speakers <= set(m['participants']) or len(speakers) < 2:
                reasons.append('need at least 2 participant speakers')
            if carrier != 'assistant' and turns and turns[0]['speaker'] != m['speaker']:
                reasons.append('first turn must use annotated speaker')
            if m.get('core'):
                assertion='\n'.join(t['speaker']+': '+t['text'] for t in turns if t['speaker']==m['speaker'])
                if not all(anchors_present(spans,assertion) for spans in m.get('anchor_sets',{}).values()):
                    reasons.append('annotated speaker must assert the fact')
    low = 1
    high = math.ceil(2.5*m['text_chars'])-1
    if not low <= len(text) <= high:
        reasons.append(f'text length {len(text)} outside {low}..{high}')
    if m.get('core') and len(text.split()) < 15:
        reasons.append('fact-bearing text needs at least 15 words')
    if 'reply_chars' in m:
        reply = row.get('assistant_reply')
        lo = 1
        hi = math.ceil(2.5*m['reply_chars'])-1
        if not isinstance(reply,str) or not lo <= len(reply) <= hi:
            reasons.append(f'assistant reply length outside {lo}..{hi}')
        # Three anchor words can occur in a discussion of wording without
        # asserting their full binding. Reject literal source copies here; the
        # semantic verifiers still reject copied/paraphrased ledger assertions.
        elif any(collapse_whitespace(line) in collapse_whitespace(reply)
                 for line in m.get('core','').splitlines() if line.strip()) or leaks_anchors(forbidden,reply):
            reasons.append('assistant reply asserts ledger facts')
    return reasons


def augment_group(row, m):
    """Label-blind social padding; never put factual content in new mouths."""
    turns=row.get('turns')
    if not isinstance(turns,list):return
    speakers={t['speaker'] for t in turns}
    row['rendered_speaker_count']=len(speakers)
    seed=int(hashlib.sha256(m['id'].encode()).hexdigest()[:8],16)
    desired=min(len(m.get('participants',[])), (2,3,4)[seed % 3])
    for name in m.get('participants',[]):
        if len(speakers)>=desired:break
        if name in speakers:continue
        text=SOCIAL_TURNS[(seed+len(speakers))%len(SOCIAL_TURNS)].format(name=m['speaker'])
        turns.append(dict(speaker=name,text=text,program_added=True));speakers.add(name)
    assemble_turns(row,m)


def valid_message(row,m,forbidden=()):
    return not message_failures(row,m,forbidden)


def render_instructions(carrier):
    form = {
        'group': 'Return {id,turns:[{speaker,text}]} with at least 2 named participants from participants. Put the annotated speaker first; only that person may assert the supplied facts. Other turns ask or acknowledge. Names, @mentions and thread markers are formatted by the program.',
        'dyad': 'Return {id,turns:[{speaker,text}]} using exactly the two participants. Put the annotated speaker first; the other may ask or acknowledge without adding facts. The program formats names.',
        'assistant': 'Return {id,text} for the requested side only. User and assistant sides are requested separately and assembled by the program in that order. Preserve facts only on the annotated side; a fact-free reply must not repeat names, values or status anchors.',
        'record': 'Return {id,text}. Write a third-person notebook entry in 1-3 coherent paragraphs, referring to people by name or pronouns.',
    }[carrier]
    return ('Render conversational memory excerpts as a JSON list, or {"messages":[...]}. '+form+
        ' The core is a source of meaning. Paraphrase naturally while preserving its attribution, scope, dates, states and any supplied conditions, replacements and links. '
        'Required anchors tolerate natural date forms, case, underscores, hyphens and spaces and simple inflections. Status anchors accept any phrase in status_alternatives, such as ran for executed or reverted for rolled_back. Other qualifiers are checked semantically. '
        'A parenthesized date after an entity identifies a distinct version by its first entry date; preserve that identity naturally, such as the January 6 entry. Keep each source atom in its own paragraph or turn. Do not mention IDs, schema fields, labels or character budgets in prose. '
        'Use natural length: short chat turns, a paragraph or two for records, and a normal helpful assistant reply. Character targets are program-side matching budgets, not minimum lengths. Fact-bearing messages need at least 15 words; empty-core social turns can be brief. '
        'Empty-core turns may ask or acknowledge but assert no ledger facts, decisions or commitments. Do not echo neighbouring facts or invent reasons, names, dates or values. '
        'Preserve annotated meaningful confirmations. Start with substantive content so evidence survives shortening.')


def sentence_trim(text, ceiling, minimum=0, required=()):
    """Only remove whole trailing sentences, and never remove a fact anchor."""
    if len(text)<=ceiling:return text
    ends=[m.end() for m in re.finditer(r'[.!?](?=\s|$)',text)]
    for end in reversed(ends):
        if minimum<=end<=ceiling and anchors_present(required,text[:end]):return text[:end]
    return text


def match_lengths(w):
    """Label-blind sentence trimming towards each case's common target.

    Empirical positive/negative distribution matching happens on final groups,
    where tokenizer/formatter effects are visible (world3_adapter).
    """
    for m in w['rounds']:
        if not m.get('length_contract') or m.get('render_error'):continue
        if m.get('turns'):continue
        original=m['text'];original_reply=m.get('assistant_reply')
        required=[a for spans in m.get('anchor_sets',{}).values() for a in spans]
        m['text']=sentence_trim(m['text'],m['text_chars'],1,required)
        if isinstance(m.get('assistant_reply'),str) and 'reply_chars' in m:
            m['assistant_reply']=sentence_trim(m['assistant_reply'],m['reply_chars'],1)
        if not valid_message(m,m):
            # Sentence trimming must preserve carrier turns as well as anchors
            # and the formatter's minimum word count.
            m['text']=original
            if original_reply is not None:m['assistant_reply']=original_reply


def filler_keys(row, message):
    # Program-generated names, mentions and thread IDs must not make identical
    # social speech look unique across excerpts.
    if row.get('turns') and message['channel'] not in ('record','assistant'):
        return {normalize(' '.join(turn['text'] for turn in row['turns'] if not turn.get('program_added')))}
    return {normalize(text) for text in (row.get('text',''),row.get('assistant_reply','')) if text}


def dedupe_fillers(w):
    """Duplicate empty-core outputs are isolated; never silently invent real prose."""
    seen=set();duplicates=[]
    for m in w['rounds']:
        if m['core'] or m.get('render_error'):continue
        keys=filler_keys(m,m)
        if keys & seen:
            m.update(render_error='duplicate no-new-fact filler',ok=False)
            duplicates.append(m['id'])
        else:seen.update(keys)
    return duplicates
