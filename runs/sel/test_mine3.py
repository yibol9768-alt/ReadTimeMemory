"""Offline stage, adversarial-label, source-offset and CLI tests for mine3."""
import contextlib
import copy
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import mine3 as m

PAD = ' ' + ' '.join(['This is unrelated background about daily work and reading.'] * 5)
TOK = m.ByteTokenizer()


def fixture_material(source='wildchat'):
    texts = [('alice', 'My editor is Vim.'), ('bob', 'My editor is Nano.'),
             ('alice', 'My editor is now Emacs.'), ('bob', "alice's editor is Emacs."),
             ('alice', 'My theme is dark.'), ('alice', 'My lunch is soup.'),
             ('bob', 'The weather is pleasant today.')]
    turns = [dict(id=str(i), tid=str(i), speaker=speaker, text=text + PAD,
                  time=f'2024-01-{i + 1:02d}T12:00:00+00:00', reply_to=[],
                  source_ref=dict(fixture=True, row=i), role='user' if speaker == 'alice' else 'assistant')
             for i, (speaker, text) in enumerate(texts)]
    raw = dict(source=source, unit='fixture-unit', source_digest=m.digest(turns), turns=turns,
               aliases={'alice': ['alice'], 'bob': ['bob']})
    return m.freeze_material(raw, TOK)


def extended_mock(family, prompt, p):
    raw, usage = m.mock_transport(family, prompt, p)
    result = json.loads(raw)
    if p['task'] == 'extract':
        for s in p['chunk']['segments']:
            for text, topic, attr, val in [('My theme is dark.', 'software', 'theme', 'dark'),
                                           ('My lunch is soup.', 'food', 'lunch', 'soup')]:
                if text in s['text']:
                    a = s['start'] + s['text'].index(text)
                    result['facts'].append(dict(holder=s['speaker'], topic=topic, attribute=attr, value=val,
                                                kind='preference', segment_id=s['segment_id'], span=text,
                                                span_start=a, span_end=a + len(text), time=s['time'],
                                                reported_by=s['speaker'], about_self=True, status='asserted'))
    return json.dumps(result), usage


def pipeline_parts(transport=extended_mock):
    material = fixture_material()
    models = m.Models(transport=transport)
    mined = m.mine(material, models, TOK, 1000, 120)
    cases = m.structures(mined)
    return material, models, mined, cases


def case_for(mined, operation):
    # Seed rotation makes all four operations reachable while retaining <=3.
    for seed in range(100):
        for case in m.structures(mined, seed):
            if case['qtype'] == operation:
                return case
    raise AssertionError('missing operation: ' + operation)


class MaterialsTests(unittest.TestCase):
    def test_blog_latin1_date_sort_and_xml_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'blogs.zip'
            raw = '<Blog><date>02,January,2004</date><post>My café &amp; tea is here.</post>' \
                  '<date>31,December,2003</date><post>I read <b>books</b> today.</post></Blog>'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('123.author.xml', raw.encode('latin-1'))
            rows = list(m.blog_materials([path]))
            material = m.freeze_material(rows[0][0], TOK)
            self.assertEqual([s['id'] for s in material['segments']], ['1', '0'])
            self.assertIn('café & tea', material['segments'][1]['text'])
            s = material['segments'][0]
            a = s['text'].index('books')
            intervals = m.source_intervals(s, a, a + 5)
            self.assertEqual(''.join(raw[a:b] for a, b in intervals), 'books')
            s = material['segments'][1]
            a = s['text'].index('&')
            self.assertEqual([raw[a:b] for a, b in m.source_intervals(s, a, a + 1)], ['&amp;'])

    def test_bad_blog_date_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'blogs.zip'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('author.xml', '<date>unknown</date><post>I am here.</post>')
            self.assertEqual(list(m.blog_materials([path])), [(None, 'invalid_blog_date')])

    def test_blog_minimum_is_thirty_posts(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / 'blogs.zip'
            with zipfile.ZipFile(path, 'w') as z:
                for name, count in [('short', 29), ('long', 30)]:
                    z.writestr(name + '.xml', ''.join(f'<date>{i:02d},January,2004</date>'
                                                       f'<post>I read {name} book number {i} today.</post>'
                                                       for i in range(1, count + 1)))
            rows, flow = m.materials(dict(blogs=str(path), wildchat='', irc_ubuntu=''),
                                    dict(blogs=1, wildchat=0, irc_ubuntu=0), TOK)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['source_segment_count'], 30)
            self.assertEqual(flow['flow']['blogs']['drops']['too_few_segments'], 1)

    def test_arrow_datetimes_are_frozen_as_json_not_lost_or_fabricated(self):
        pq = types.ModuleType('pyarrow.parquet')
        pa = types.ModuleType('pyarrow')
        pa.parquet = pq
        row = dict(timestamp=dt.datetime(2024, 1, 2, 3, 4, tzinfo=dt.timezone.utc),
                   conversation=[dict(role='user', content='Hello.', timestamp=dt.datetime(2024, 1, 2, 3, 5))])
        pq.ParquetFile = lambda path: types.SimpleNamespace(iter_batches=lambda batch_size: [
            types.SimpleNamespace(to_pylist=lambda: [row])])
        with patch.dict('sys.modules', {'pyarrow': pa, 'pyarrow.parquet': pq}):
            parsed = list(m.corpus_rows([Path('/fake/shard.parquet')]))[0]
        self.assertIsInstance(parsed['timestamp'], str)
        self.assertEqual(m.timestamp(parsed['conversation'][0]['timestamp']), '2024-01-02T03:05:00+00:00')

    def test_native_irc_ascii_and_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '2004-12-25.ascii.txt'
            path.write_text('[10:00] <alice> I am here.\n[10:01] <bob> Yes, hello.\n')
            path.with_name('2004-12-25.annotation.txt').write_text('1 0 -\n')
            rows = list(m.irc_materials([path]))
            self.assertEqual(rows[0][0]['turns'][1]['reply_to'], ['0'])
            self.assertEqual(len(rows[0][0]['turns']), 2)

    def test_long_irc_period_bounds_identity_allocation_before_renaming(self):
        # The existing name bank has 336 identities. Only the bounded material
        # should allocate names, not the other hundreds in the channel-day.
        turns = [dict(id=str(i), tid=str(i), speaker='nick' + str(i), text='I read my book today. ' * 4,
                      time='2024-01-01T01:00:00+00:00', reply_to=[], source_ref={}) for i in range(400)]
        raw = dict(source='irc_ubuntu', unit='channel-day', source_digest='long', turns=turns,
                   aliases={t['speaker']: [t['speaker']] for t in turns})
        frozen = m.freeze_material(raw, TOK, cap=500)
        self.assertLessEqual(frozen['source_tokens'], 500)
        self.assertLessEqual(frozen['sanitation']['identities'], 10)
        self.assertEqual(frozen['source_segment_count'], 400)

    def test_sanitize_once_consistent_names_pii_and_exact_literal_mapping(self):
        original = 'Hello alice! I see bob at https://site.test/x and x@example.org, +1 212-555-1234. &amp; Café'
        turns = [dict(id='0', tid='0', speaker='alice', text=original, raw_start=100),
                 dict(id='1', tid='1', speaker='bob', text='@alice said bob is here.')]
        with patch.object(m.sx, 'rename_all', wraps=m.sx.rename_all) as rename:
            out, audit = m.sanitize(turns, {'alice': ['alice'], 'bob': ['bob']}, 42)
        self.assertEqual(rename.call_count, 1)
        text = out[0]['text']
        self.assertIn(out[0]['speaker'], out[1]['text'])
        self.assertIn(out[1]['speaker'], text)
        self.assertNotRegex(' '.join(t['text'] for t in out), r'alice|bob|site\.test|example\.org|212')
        self.assertIn('[URL]', text)
        self.assertIn('[PHONE]', text)
        a = text.index('Café')
        self.assertEqual(m.source_intervals(out[0], a, a + 4), [[100 + original.index('Café'), 100 + len(original)]])
        self.assertEqual(audit['leftover_count'], 0)
        for t in out:
            self.assertEqual(sum(r['end'] - r['start'] for r in t['offset_map']), len(t['text']))

    def test_timestamp_sources_and_wildchat_no_cross_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'wild.jsonl'
            rows = [dict(conversation_hash='c1', timestamp=1700000000000, language='English',
                         conversation=[dict(role='user', content='I am here.', timestamp=1700000001000),
                                       dict(role='assistant', content='Hello there.')]),
                    dict(conversation_hash='c2', timestamp=1700000100000, language='English',
                         conversation=[dict(role='user', content='I am still here.')]),
                    dict(language='French', conversation=[])]
            m.write_rows(path, rows)
            parsed = list(m.wildchat_materials([path]))
            self.assertEqual(parsed[0][0]['unit'], 'c1')
            self.assertEqual(parsed[1][0]['unit'], 'c2')
            self.assertEqual(parsed[0][0]['turns'][1]['time_source'], 'conversation_timestamp')
            self.assertEqual(parsed[0][0]['turns'][0]['time'], m.timestamp(1700000001000))
            self.assertEqual(parsed[-1][1], 'non_english')

    def test_irc_channel_day_contiguous_gaps_and_reply_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'irc.jsonl'
            rows = [dict(id=i, date=day, channel=ch, raw=raw, connections=[0])
                    for i, day, ch, raw in [(0, '2004-12-25', 'a', '[10:00] <alice> I am here.'),
                                            (1, '2004-12-25', 'a', '[10:01] <bob> I see alice.'),
                                            (3, '2004-12-25', 'a', '[10:03] <alice> Hello again.'),
                                            (4, '2004-12-25', 'a', '*** disconnected'),
                                            (5, '2004-12-25', 'a', '[10:05] <bob> Hello.'),
                                            (0, '2004-12-26', 'a', '[10:00] <alice> I am here.'),
                                            (0, '2004-12-25', 'b', '[10:00] <alice> Hello.')]]
            m.write_rows(path, rows)
            parsed = [r for r, _ in m.irc_materials([path])]
            self.assertEqual(len(parsed), 3)
            first = parsed[0]
            self.assertEqual([t['id'] for t in first['turns']], ['0', '1', '3', '5'])
            self.assertEqual(first['turns'][1]['reply_to'], ['0'])
            later = copy.deepcopy(first)
            later['turns'] = first['turns'] * 2
            later['source_digest'] = 'different-period'
            self.assertEqual(m.freeze_material(first, TOK)['split'], m.freeze_material(later, TOK)['split'])

    def test_seeded_selection_minimums_and_duplicate_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'wild.jsonl'
            rows = [dict(id=str(n), language='English', timestamp=1700000000000,
                         conversation=[dict(role='user' if i % 2 == 0 else 'assistant',
                                            content=f'I have project {n}, message {i}.') for i in range(turns)])
                    for n, turns in enumerate([30, 33, 35, 29])]
            rows.append(dict(rows[0], id='duplicate-other-id'))
            m.write_rows(path, rows)
            paths = dict(blogs='', wildchat=str(path), irc_ubuntu='')
            counts = dict(blogs=0, wildchat=2, irc_ubuntu=0)
            with contextlib.redirect_stdout(io.StringIO()):
                one, flow = m.materials(paths, counts, TOK)
                two, _ = m.materials(paths, counts, TOK)
            self.assertEqual(one, two)
            self.assertEqual(flow['flow']['wildchat']['eligible'], 3)
            self.assertEqual(flow['flow']['wildchat']['drops']['duplicate'], 1)
            self.assertEqual(len(one), 2)
            self.assertTrue(all(len(x['segments']) >= 30 for x in one))

    def test_cap_is_contiguous_and_source_text_immutable(self):
        material = fixture_material()
        with self.assertRaisesRegex(ValueError, 'mutated'):
            bad = copy.deepcopy(material)
            bad['segments'][0]['text'] += 'extra'
            m.check_material(bad)
        raw = dict(source='blogs', unit='author', source_digest='fixed', aliases={'Author': []},
                   turns=[dict(id=str(i), tid=str(i), speaker='Author', text='I am here. ' * 8,
                               time=m.timestamp(f'2004-01-{i + 1:02d}'), reply_to=[], source_ref={}) for i in range(20)])
        capped = m.freeze_material(raw, TOK, cap=400)
        self.assertLessEqual(capped['source_tokens'], 400)
        ids = [int(s['id']) for s in capped['segments']]
        self.assertEqual(ids, list(range(ids[0], ids[-1] + 1)))
        self.assertTrue(all(s['text'] == 'I am here. ' * 7 + 'I am here.' for s in capped['segments']))
        with self.assertRaises(ValueError):
            m.freeze_material(raw, TOK, cap=16001)


    def test_irc_windows_ignore_gaps_and_unparsed_lines_and_balance_tail(self):
        rows = [dict(id=i * 3, channel='a', date='2004-12-25',
                     raw=f'[{i // 60:02d}:{i % 60:02d}] <alice> I use my computer.', connections=[])
                for i in range(801)]
        rows.insert(100, dict(id=299, channel='a', date='2004-12-25', raw='*** disconnected'))
        rows.append(dict(id=0, channel='b', date='2004-12-25', raw='[00:00] <bob> I am here.'))
        rows.append(dict(id=0, channel='a', date='2004-12-26', raw='[00:00] <alice> I am back.'))
        with patch.object(m, 'corpus_rows', return_value=iter(reversed(rows))):
            parsed = [r for r, _ in m.irc_materials([])]
        day = [r for r in parsed if r['unit'] == 'a:2004-12-25']
        self.assertEqual([len(r['turns']) for r in day], [401, 400])
        ids = [int(t['id']) for r in day for t in r['turns']]
        self.assertEqual(ids, list(range(0, 801 * 3, 3)))
        self.assertEqual(len(parsed), 4)

    def test_wildchat_prefers_user_changes_and_rejects_short_or_assistant_only(self):
        def conversation(unit, n, personal=True, changes=False):
            turns = []
            for i in range(n):
                text = 'Please explain the theory of computer science.'
                if personal and i % 2 == 0:
                    text = 'My project uses Python for a web application.'
                    if changes and i == 2:
                        text = 'I have now switched my project to Rust instead.'
                if not personal and i % 2:
                    text = 'I have now switched my project to Rust instead.'
                turns.append(dict(role='user' if i % 2 == 0 else 'assistant', content=text))
            return dict(id=unit, language='English', timestamp=1700000000000, conversation=turns)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / 'wild.jsonl'
            rows = [conversation('long', 100), conversation('changed', 30, changes=True),
                    conversation('short', 29, changes=True), conversation('assistant-only', 40, personal=False)]
            m.write_rows(path, rows)
            frozen, flow = m.materials(dict(blogs='', wildchat=str(path), irc_ubuntu=''),
                                       dict(blogs=0, wildchat=1, irc_ubuntu=0), TOK)
        self.assertEqual(len(frozen), 1)
        self.assertEqual(frozen[0]['split_unit'], m.digest(['wildchat', 'changed']))
        self.assertEqual(frozen[0]['source_segment_count'], 30)
        self.assertEqual(flow['flow']['wildchat']['drops']['too_few_segments'], 1)
        self.assertEqual(flow['flow']['wildchat']['drops']['no_user_personal_facts'], 1)
        self.assertEqual(len({s['source_ref']['conversation'] for s in frozen[0]['segments']}), 1)


class MiningTests(unittest.TestCase):
    def test_complete_objects_salvaged_only_from_extraction_array(self):
        fact = dict(value='a } and [ inside a string', nested=dict(complete=True))
        raw = '{"facts":[' + json.dumps(fact) + ',{"value":"unfinished'
        self.assertEqual(m.extraction_response(raw), dict(facts=[fact], truncated=True))
        self.assertEqual(m.extraction_response('[' + json.dumps(fact) + ']'), dict(facts=[fact]))
        self.assertEqual(m.extraction_response('{"facts":[]}'), dict(facts=[]))
        for raw in ('{"facts":[{"nested":{"value":"inner"}', '{"other":[{"value":"not a fact"}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                m.extraction_response(raw)

    def test_extraction_failure_splits_chunk_and_preserves_offsets(self):
        sizes = []

        def transport(family, prompt, p):
            if p['task'] == 'extract':
                self.assertEqual(p['max_facts'], 12)
                size = p['chunk']['end'] - p['chunk']['start']
                sizes.append(size)
                if size > 500:
                    return '{"facts":[', dict(prompt_tokens=10, completion_tokens=1)
            return extended_mock(family, prompt, p)
        _, models, mined, _ = pipeline_parts(transport)
        self.assertEqual(len(mined['facts']), 6)
        self.assertTrue(any(e['error'] for e in mined['extraction']))
        self.assertTrue(any(e['depth'] > 0 for e in mined['extraction']))
        self.assertLess(min(sizes), max(sizes))
        failures = [e for e in models.records if not e['ok']]
        self.assertTrue(failures)
        self.assertTrue(all(e['attempt'] == 1 and e['usage'] for e in failures))

    def test_truncated_extraction_salvages_and_retries_smaller(self):
        def transport(family, prompt, p):
            raw, usage = extended_mock(family, prompt, p)
            if p['task'] == 'extract' and p['chunk']['end'] - p['chunk']['start'] > 700:
                rows = json.loads(raw)['facts']
                if rows:
                    return '{"facts":[' + json.dumps(rows[0]) + ',{"holder":', usage
            return raw, usage
        _, models, mined, _ = pipeline_parts(transport)
        self.assertEqual(len(mined['facts']), 6)
        self.assertTrue(any(e.get('salvaged') for e in models.records))
        self.assertTrue(any(e['depth'] > 0 for e in mined['extraction']))
        self.assertEqual(len({f['id'] for f in mined['facts']}), 6)

    def test_extraction_http_output_budget_is_higher(self):
        captured = []

        def urlopen(request, timeout):
            captured.append(json.loads(request.data))
            response = unittest.mock.MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(
                dict(choices=[dict(message=dict(content='{"facts":[]}'))], usage={})).encode()
            return response
        models = m.Models(['qwen@1', 'gemma@2', 'gpt-oss@3'])
        with patch.object(m.urllib.request, 'urlopen', side_effect=urlopen):
            models.request(0, m.EXTRACT, dict(task='extract'))
            models.request(0, 'confirm', dict(task='confirm'))
        self.assertGreater(captured[0]['max_tokens'], 6000)
        self.assertEqual(captured[1]['max_tokens'], 6000)
        self.assertIn('at most 12', m.EXTRACT)

    def test_canonicalization_payload_is_bounded_by_holder_and_items(self):
        material, _, mined, _ = pipeline_parts()
        base = mined['facts'][0]
        facts = [dict(base, id=f'f{i}', attribute=f'attribute {i:03d}', confirmation={'reason': 'x' * 50000},
                      span='x' * 50000, scope='s' * 50000, holder=f'holder{i % 2}') for i in range(130)]
        batches = []

        def transport(family, prompt, p):
            if p['task'] == 'canonicalize':
                batches.append(p)
                self.assertLessEqual(len(p['facts']), 24)
                self.assertTrue(all(f['holder'] == p['holder'] for f in p['facts']))
                self.assertNotIn('confirmation', p['facts'][0])
                self.assertLessEqual(len(json.dumps(p['facts'])), 12000)
                return json.dumps(dict(merges=[])), dict(prompt_tokens=1, completion_tokens=1)
            return extended_mock(family, prompt, p)
        m.canonicalize(facts, material, m.Models(transport=transport))
        self.assertGreater(len(batches), 2)
        self.assertEqual({f['attribute'] for p in batches for f in p['facts']}, {f['attribute'] for f in facts})

    def test_replacement_prefilter_and_positive_only_second_family(self):
        material, _, mined, _ = pipeline_parts()
        old = next(f for f in mined['facts'] if f['value'] == 'Vim')
        new = next(f for f in mined['facts'] if f['value'] == 'Emacs')
        self.assertTrue(m.replacement_candidate(old, new, 'wildchat'))
        for changes in (dict(kind='event'), dict(status='proposal'), dict(value=old['value']),
                        dict(holder='someone else'), dict(canonical_attribute='theme')):
            with self.subTest(changes=changes):
                self.assertFalse(m.replacement_candidate(old, dict(new, **changes), 'wildchat'))
        self.assertFalse(m.replacement_candidate(dict(old, condition='at home'), dict(new, condition='at work'), 'wildchat'))
        self.assertFalse(m.replacement_candidate(old, dict(new, time=old['time']), 'blogs'))
        seen = []

        def transport(family, prompt, p):
            raw, usage = extended_mock(family, prompt, p)
            if p['task'] == 'replacement':
                seen.append((family, p))
                self.assertEqual(len(p['context']), 2)
                self.assertNotIn('source_intervals', p['old'])
                self.assertNotIn(material['segments'][-1]['text'], [s['text'] for s in p['context']])
                result = json.loads(raw)
                result['replacement'] = False
                return json.dumps(result), usage
            return raw, usage
        _, _, rejected, _ = pipeline_parts(transport)
        self.assertEqual([f for f, _ in seen], [0])
        self.assertFalse(rejected['updates'])
        self.assertEqual(len(mined['replacement_audit'][0]['votes']), 2)

    def test_replacement_context_has_one_neighbor_sentence_each_side(self):
        material = fixture_material()
        s = material['segments'][0]
        s['text'] = 'Far before. Neighbor before. My editor is Vim. Neighbor after. Far after.'
        a = s['text'].index('My editor')
        fact = dict(segment_id=s['id'], span='My editor is Vim.', span_start=a, span_end=a + len('My editor is Vim.'))
        context = m.span_context(material, fact)
        self.assertEqual(context['text'].strip(), 'Neighbor before. My editor is Vim. Neighbor after.')
        self.assertEqual(context['text'][context['span_start']:context['span_start'] + len(fact['span'])], fact['span'])

    def test_chunk_overlap_exact_offsets_and_dedup(self):
        material, models, mined, _ = pipeline_parts()
        chunks = list(m.chunks(material, TOK, 700, 150))
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(a['end'] > b['start'] for a, b in zip(chunks, chunks[1:])))
        for chunk in chunks:
            self.assertLessEqual(m.ntokens(m.material_text(material['segments'])[chunk['start']:chunk['end']], TOK), 700)
            for p in chunk['segments']:
                s = next(s for s in material['segments'] if s['id'] == p['segment_id'])
                self.assertEqual(s['text'][p['start']:p['end']], p['text'])
        self.assertEqual(len(mined['facts']), 6)
        self.assertEqual(len({f['id'] for f in mined['facts']}), 6)
        self.assertEqual(len(mined['updates']), 1)
        self.assertEqual(len(mined['relays']), 1)
        self.assertEqual(mined['facts'][0]['reported_by'], material['segments'][0]['speaker'])

    def test_invalid_literal_wrong_holder_and_confirmation_rejected(self):
        def transport(family, prompt, p):
            raw, usage = extended_mock(family, prompt, p)
            r = json.loads(raw)
            if p['task'] == 'extract' and r['facts']:
                fake = dict(r['facts'][0], span='invented fact')
                wrong = dict(r['facts'][0], holder='unknown holder')
                r['facts'] += [fake, wrong]
            if p['task'] == 'confirm' and p['fact']['value'] == 'Nano':
                r['holder_correct'] = False
            return json.dumps(r), usage
        _, _, mined, _ = pipeline_parts(transport)
        self.assertTrue(any('nonliteral' in f['reason'] for f in mined['rejected_facts']))
        self.assertTrue(any('holder differs' in f['reason'] for f in mined['rejected_facts']))
        self.assertFalse(any(f['value'] == 'Nano' for f in mined['facts']))

    def test_unique_literal_derives_offsets_but_ambiguous_quotes_fail_closed(self):
        material = fixture_material()
        chunk = next(m.chunks(material, TOK, 16000, 100))
        raw, _ = extended_mock(0, '', dict(task='extract', chunk=chunk))
        row = json.loads(raw)['facts'][0]
        row.update(span_start=9999, span_end=10001, time='2024-01-01T12:00:00Z')
        fact = m.validate_fact(row, material, chunk)
        self.assertEqual(fact['span_start'], 0)
        self.assertEqual(fact['time'], material['segments'][0]['time'])
        self.assertTrue(fact['offsets_derived_from_unique_literal'])
        repeated = copy.deepcopy(material)
        repeated['segments'][0]['text'] = row['span'] + ' ' + row['span']
        repeated['segments'][0]['clean_end'] = len(repeated['segments'][0]['text'])
        duplicate_chunk = dict(segments=[dict(segment_id='0', start=0, end=len(repeated['segments'][0]['text']),
                                             text=repeated['segments'][0]['text'])])
        with self.assertRaisesRegex(ValueError, 'ambiguous literal'):
            m.validate_fact(row, repeated, duplicate_chunk)

    def test_proposals_and_rejected_replacements_do_not_make_current(self):
        def transport(family, prompt, p):
            raw, usage = extended_mock(family, prompt, p)
            r = json.loads(raw)
            if p['task'] == 'replacement':
                r.update(replacement=False, same_event_scope=False, reason='different events')
            return json.dumps(r), usage
        _, _, mined, cases = pipeline_parts(transport)
        self.assertEqual(mined['updates'], [])
        self.assertTrue(all(c['qtype'] == 'count' for c in cases))

        def proposals(family, prompt, p):
            raw, usage = extended_mock(family, prompt, p)
            r = json.loads(raw)
            if p['task'] == 'extract':
                for f in r['facts']:
                    if f['value'] == 'Emacs':
                        f['status'] = 'proposal'
            return json.dumps(r), usage
        _, _, mined, _ = pipeline_parts(proposals)
        self.assertEqual(mined['updates'], [])
        self.assertEqual(mined['relays'], [])

    def test_canonicalization_needs_independent_confirmation(self):
        material, _, mined, _ = pipeline_parts()
        facts = copy.deepcopy(mined['facts'])
        editor = next(f for f in facts if f['value'] == 'Emacs')
        editor['attribute'] = 'text editor'

        def canonical(family, prompt, p):
            if p['task'] == 'canonicalize':
                return json.dumps(dict(merges=[['editor', 'text editor']])), dict(prompt_tokens=1, completion_tokens=1)
            if p['task'] == 'canonical_confirm':
                return json.dumps(dict(same_slot=False, reason='uncertain')), dict(prompt_tokens=1, completion_tokens=1)
            return extended_mock(family, prompt, p)
        audit = m.canonicalize(facts, material, m.Models(transport=canonical))
        self.assertTrue(audit)
        self.assertEqual(editor['canonical_attribute'], 'text editor')
        facts2 = copy.deepcopy(facts)

        def confirmed(family, prompt, p):
            if p['task'] == 'canonical_confirm':
                return json.dumps(dict(same_slot=True, reason='exact synonym')), dict(prompt_tokens=1, completion_tokens=1)
            return canonical(family, prompt, p)
        m.canonicalize(facts2, material, m.Models(transport=confirmed))
        self.assertEqual(next(f for f in facts2 if f['id'] == editor['id'])['canonical_attribute'], 'editor')

    def test_ambiguous_latest_or_nonexhaustive_set_blocks_structures(self):
        _, _, mined, _ = pipeline_parts()
        value_set = mined['value_sets'][0]
        value_set['result']['latest_id'] = None
        self.assertEqual([c['qtype'] for c in m.structures(mined)], ['count'])
        value_set['result']['exhaustive'] = False
        self.assertEqual(m.structures(mined), [])


class QuestionTests(unittest.TestCase):
    def test_natural_question_prompt_and_template_retry(self):
        _, _, mined, _ = pipeline_parts()
        case = case_for(mined, 'current')
        holder = case['intent']['holder']
        old = f'Which editor was explicitly reported for {holder} in the record?'
        self.assertIn('template language', m.question_failures(old, case))
        self.assertIn('template language', m.question_failures('What was the effective value?', case))
        natural = f'As of January 7, 2024, which editor does {holder} use?'
        self.assertEqual(m.question_failures(natural, case), [])
        calls = []

        def transport(family, prompt, p):
            if p['task'] == 'question':
                calls.append(p)
                self.assertIn('what does Mira do for work?', prompt)
                return json.dumps(dict(question=old if len(calls) == 1 else natural)), dict(prompt_tokens=1, completion_tokens=1)
            return extended_mock(family, prompt, p)
        written = m.questions(case, m.Models(transport=transport))
        self.assertTrue(written['ok'])
        self.assertEqual(written['question'], natural)
        self.assertEqual([v['family'] for v in written['attempts'][-1]['votes']], [0, 2])

    def test_temporal_questions_are_preferred(self):
        _, _, mined, _ = pipeline_parts()
        self.assertEqual({c['qtype'] for c in m.structures(mined, 31)}, {'current', 'history', 'compare'})

    def test_count_scans_all_segments_and_independently_confirms_before_writer(self):
        material, _, mined, _ = pipeline_parts()
        case = case_for(mined, 'count')
        calls = []

        def transport(family, prompt, p):
            calls.append((family, copy.deepcopy(p)))
            if p['task'] in ('count_scan', 'count_confirm'):
                self.assertEqual(p['context'], m.source_view(material))
            return extended_mock(family, prompt, p)
        written = m.questions(case, m.Models(transport=transport))
        self.assertTrue(written['ok'])
        self.assertEqual([(f, p['task']) for f, p in calls[:3]], [(0, 'count_scan'), (2, 'count_confirm'), (0, 'question')])
        self.assertTrue(written['count_check']['ok'])

    def test_incomplete_count_gold_or_uncertainty_drops_count_before_writing(self):
        _, _, mined, _ = pipeline_parts()
        case = case_for(mined, 'count')
        # The overlooked value is outside every mined fact and after 3000 chars.
        case['count_context'][-1]['text'] += ' Background.' * 400 + f" {case['intent']['holder']}'s editor is Nano."
        calls = []

        def transport(family, prompt, p):
            calls.append(p['task'])
            return extended_mock(family, prompt, p)
        written = m.questions(case, m.Models(transport=transport))
        self.assertFalse(written['ok'])
        self.assertEqual(written['drop_reason'], 'count_incomplete')
        self.assertEqual(calls, ['count_scan', 'count_confirm'])
        self.assertEqual(len(written['count_check']['votes'][0]['result']['values']), 3)
        for uncertain_task in ('count_scan', 'count_confirm'):
            calls.clear()

            def uncertain(family, prompt, p):
                calls.append(p['task'])
                raw, usage = extended_mock(family, prompt, p)
                result = json.loads(raw)
                if p['task'] == uncertain_task:
                    result['exhaustive'] = False
                return json.dumps(result), usage
            failed = m.questions(case_for(mined, 'count'), m.Models(transport=uncertain))
            self.assertEqual(failed['drop_reason'], 'count_incomplete')
            self.assertNotIn('question', calls)
        self.assertEqual(len(m.structures(mined, 31)), 3)

    def test_at_most_three_distinct_structures_and_compare_two_values(self):
        _, _, mined, _ = pipeline_parts()
        seen = set()
        for seed in range(10):
            cases = m.structures(mined, seed)
            self.assertLessEqual(len(cases), 3)
            self.assertEqual(len({c['qtype'] for c in cases}), len(cases))
            seen.update(c['qtype'] for c in cases)
        self.assertEqual(seen, {'current', 'history', 'count', 'compare'})
        compare = case_for(mined, 'compare')
        self.assertEqual(compare['answer'], 'Vim -> Emacs')
        self.assertEqual(len(compare['components']), 2)
        self.assertEqual(case_for(mined, 'count')['answer'], '2')

    def test_two_family_meaning_judge_retry_and_no_answer_leak(self):
        _, _, mined, _ = pipeline_parts()
        case = case_for(mined, 'current')
        seen = []

        def transport(family, prompt, p):
            seen.append((family, copy.deepcopy(p)))
            raw, usage = extended_mock(family, prompt, p)
            r = json.loads(raw)
            if p['task'] == 'intent' and family == 2 and sum(t['task'] == 'question' for _, t in seen) == 1:
                r = dict(faithful=False, reason='cutoff missing')
            return json.dumps(r), usage
        written = m.questions(case, m.Models(transport=transport))
        self.assertTrue(written['ok'])
        self.assertEqual(len(written['attempts']), 2)
        self.assertEqual([v['family'] for v in written['attempts'][-1]['votes']], [0, 2])
        self.assertIn('cutoff missing', written['attempts'][0]['failures'])
        self.assertTrue(any(p.get('retry_feedback') for _, p in seen if p['task'] == 'question'))
        self.assertEqual(m.question_failures('Is it Emacs?', case), ['answer value leaked'])

    def test_bad_questions_are_persisted_as_failed_attempts(self):
        _, _, mined, _ = pipeline_parts()

        def transport(family, prompt, p):
            if p['task'] == 'question':
                return json.dumps(dict(question='Is it Emacs?')), dict(prompt_tokens=1, completion_tokens=1)
            return extended_mock(family, prompt, p)
        failed = m.questions(case_for(mined, 'current'), m.Models(transport=transport))
        self.assertFalse(failed['ok'])
        self.assertEqual(len(failed['attempts']), 3)
        self.assertEqual(failed['drop_reason'], 'question_intent')


class VerificationTests(unittest.TestCase):
    def test_semantic_control_grading_accepts_paraphrases_for_all_temporal_operations(self):
        for operation, expected, response, values in (
                ('current', 'Knee damage', 'an injury to his knee', ['Knee damage']),
                ('history', 'swollen knee', 'his knee was swollen', ['swollen knee']),
                ('compare', 'swollen knee -> Knee damage',
                 'Initially his knee was swollen; later he had an injury to the knee.', ['swollen knee', 'Knee damage']),
                ('compare', 'Vim -> Emacs', 'Emacs is the current editor; previously it was Vim.', ['Vim', 'Emacs'])):
            with self.subTest(operation=operation):
                case = dict(material_id='m', question_id='q', question='What happened to his knee?',
                            qtype=operation, answer=expected, components=[dict(value=v) for v in values])

                def transport(family, prompt, p):
                    if p['task'] == 'answer':
                        self.assertNotIn('expected', p)
                        return json.dumps(dict(answer=response, abstain=False)), dict(prompt_tokens=1, completion_tokens=1)
                    self.assertEqual(p['task'], 'answer_grade')
                    self.assertIn('Grade MEANING', prompt)
                    return json.dumps(dict(correct=True, reason='same meaning')), dict(prompt_tokens=1, completion_tokens=1)
                votes = m.answer_control(case, ['full gold'], m.Models(transport=transport), 'positive')
                self.assertTrue(all(v['correct'] for v in votes))

    def test_positive_readers_receive_full_gold_beyond_character_and_token_caps(self):
        material = copy.deepcopy(self.material)
        material['segments'][2]['text'] += ' Extra background.' * 250 + ' GOLD_TAIL_MARKER.'
        position = 0
        for s in material['segments']:
            s.update(clean_start=position, clean_end=position + len(s['text']), text_sha256=m.digest(s['text']))
            position = s['clean_end'] + 1
        material['immutable_sha256'] = m.digest({k: v for k, v in material.items() if k != 'immutable_sha256'})
        mined = dict(self.mined, material=material)
        seen = []

        def transport(family, prompt, p):
            if p['task'] == 'answer' and p['control'] == 'positive':
                seen.append(p)
                self.assertTrue(any(material['segments'][2]['text'] in doc for doc in p['documents']))
                self.assertTrue(any('GOLD_TAIL_MARKER' in doc for doc in p['documents']))
            return extended_mock(family, prompt, p)
        verified = m.verify(self.current, mined, m.Models(transport=transport), TOK, 1000)
        self.assertTrue(verified['accepted'])
        self.assertEqual(len(seen), 2)
        self.assertNotIn('GOLD_TAIL_MARKER', ''.join(verified['visible_documents']))
        self.assertTrue(all(m.ntokens(doc[:-len(m.SFX)], TOK) <= 1000 for doc in verified['visible_documents']))
        self.assertIn('2', verified['positive_control_segment_ids'])

    def setUp(self):
        self.material, self.models, self.mined, _ = pipeline_parts()
        self.current = m.questions(case_for(self.mined, 'current'), self.models)

    def test_every_visible_candidate_and_native_negative_types(self):
        c = m.verify(self.current, self.mined, self.models, TOK, 1000)
        self.assertTrue(c['accepted'], c.get('drop_reason'))
        self.assertEqual(len(c['judgments']), len(self.material['segments']))
        self.assertEqual(c['gold_indices'], [2, 3])
        self.assertEqual(c['kinds'], ['minimal_state_diff', 'wrong_person', 'pos', 'pos',
                                      'same_topic_wrong_attribute', 'same_speaker_other_topic', 'random'])
        self.assertIn('3', c['removed_segment_ids'])
        self.assertTrue(all(v['correct'] for v in c['controls']['positive']))
        self.assertFalse(any(v['correct'] for v in c['controls']['evidence_removed']))
        # Relayed SAME-holder evidence is positive, never wrong-person.
        self.assertEqual(c['negative_types'][3], [])

    def test_support_normalization_quotes_and_third_vote(self):
        seen = []

        def transport(family, prompt, p):
            if p['task'] == 'support':
                seen.append((family, p['index']))
            raw, usage = extended_mock(family, prompt, p)
            r = json.loads(raw)
            if p['task'] == 'support' and p['index'] == 2:
                if family == 2:
                    r.update(support='none', components=[], quotes={})
                else:
                    r['support'] = 'partial'  # one component => normalized full
            return json.dumps(r), usage
        c = m.verify(self.current, self.mined, m.Models(transport=transport), TOK, 1000)
        self.assertTrue(c['accepted'])
        self.assertEqual(c['judgments'][2]['result']['support'], 'full')
        self.assertIn((1, 2), seen)
        self.assertFalse(c['judgments'][2]['primary_agreement'])
        doc = 'Alice said “yes”\n today.'
        self.assertEqual(m.literal_quote('"yes" today.', doc), '“yes”\n today.')
        with self.assertRaisesRegex(ValueError, 'not visible'):
            m.normalize_support(dict(support='full', components=['x'], quotes={'x': 'absent'}, reason='bad'), [{'id': 'x'}], doc)

    def test_unresolved_candidate_drops_whole_question_never_default_negative(self):
        def transport(family, prompt, p):
            if p['task'] == 'support' and p['index'] == 6:
                if family == 0:
                    return json.dumps(dict(support='none', components=[], quotes={}, reason='none')), dict(prompt_tokens=1, completion_tokens=1)
                return '{"broken":', dict(prompt_tokens=1, completion_tokens=1)
            return extended_mock(family, prompt, p)
        c = m.verify(self.current, self.mined, m.Models(transport=transport), TOK, 1000)
        self.assertFalse(c['accepted'])
        self.assertEqual(c['drop_reason'], 'unresolved_candidate')
        self.assertIsNone(c['judgments'][6]['result'])

    def test_reader_never_gets_reference_values_or_extracted_intent(self):
        calls = []

        def transport(family, prompt, p):
            if p['task'] == 'answer':
                calls.append(p)
                self.assertNotIn('expected', p)
                self.assertNotIn('intent', p)
                self.assertNotIn('components', p)
                self.assertNotIn('Vim', p['question'])
            return extended_mock(family, prompt, p)
        m.verify(self.current, self.mined, m.Models(transport=transport), TOK, 1000)
        self.assertEqual(len(calls), 4)
        for p in calls:
            if p['control'] == 'evidence_removed':
                self.assertFalse(any('Emacs' in doc for doc in p['documents']))

    def test_positive_failure_and_answerable_removed_control_drop(self):
        for control, expected in [('positive', 'positive_control'), ('evidence_removed', 'evidence_removed_answerable')]:
            with self.subTest(control=control):
                def transport(family, prompt, p):
                    if p['task'] == 'answer' and p['control'] == control:
                        return json.dumps(dict(answer='' if control == 'positive' else 'Emacs',
                                               abstain=control == 'positive', reason='forced')), dict(prompt_tokens=1, completion_tokens=1)
                    return extended_mock(family, prompt, p)
                c = m.verify(self.current, self.mined, m.Models(transport=transport), TOK, 1000)
                self.assertFalse(c['accepted'])
                self.assertEqual(c['drop_reason'], expected)

    def test_compare_changed_alone_is_incorrect_even_with_bad_grader(self):
        case = m.questions(case_for(self.mined, 'compare'), self.models)

        def transport(family, prompt, p):
            if p['task'] == 'answer':
                return json.dumps(dict(answer='changed', abstain=False)), dict(prompt_tokens=1, completion_tokens=1)
            if p['task'] == 'answer_grade':
                return json.dumps(dict(correct=True, reason='bad grader')), dict(prompt_tokens=1, completion_tokens=1)
            return extended_mock(family, prompt, p)
        c = m.verify(case, self.mined, m.Models(transport=transport), TOK, 1000)
        self.assertEqual(c['drop_reason'], 'positive_control')

    def test_comparison_reversed_values_do_not_pass(self):
        case = m.questions(case_for(self.mined, 'compare'), self.models)

        def transport(family, prompt, p):
            if p['task'] == 'answer':
                return json.dumps(dict(answer='Emacs -> Vim', abstain=False)), dict(prompt_tokens=1, completion_tokens=1)
            if p['task'] == 'answer_grade':
                return json.dumps(dict(correct=True, reason='bad grader')), dict(prompt_tokens=1, completion_tokens=1)
            return extended_mock(family, prompt, p)
        c = m.verify(case, self.mined, m.Models(transport=transport), TOK, 1000)
        self.assertEqual(c['drop_reason'], 'positive_control')

    def test_inherited_context_is_protected_and_removed(self):
        changed = copy.deepcopy(self.material)
        changed['segments'][3]['text'] = 'Yes, exactly.'
        s = changed['segments'][3]
        s.update(text_sha256=m.digest(s['text']), offset_map=m.compact_map([(i, i + 1) for i in range(len(s['text']))]))
        changed['immutable_sha256'] = m.digest({k: v for k, v in changed.items() if k != 'immutable_sha256'})
        mined = dict(self.mined, material=changed)
        docs, deps = m.serialize(changed, TOK, 1000)
        self.assertIn('Emacs', docs[3])
        self.assertIn('2', deps[3])
        c = m.verify(self.current, mined, self.models, TOK, 1000)
        self.assertIn(3, c['gold_indices'])
        self.assertIn('3', c['removed_segment_ids'])

    def test_exact_consumed_token_truncation_not_full_segment(self):
        docs, _ = m.serialize(self.material, TOK, 80)
        for doc in docs:
            self.assertTrue(doc.endswith(m.SFX))
            self.assertLessEqual(m.ntokens(doc[:-len(m.SFX)], TOK), 80)
        c = m.verify(self.current, self.mined, self.models, TOK, 10)
        self.assertFalse(c['accepted'])
        self.assertEqual(c['drop_reason'], 'incomplete_visible_evidence')

    def test_exported_documents_are_fixed_points_of_training_retokenization(self):
        class ExpandingTokenizer(m.ByteTokenizer):
            world3_fingerprint = 'expanding-fixture'

            def decode(self, ids):
                return super().decode(ids).replace('Emacs', 'E m a c s')

        tokenizer = ExpandingTokenizer()
        docs, _ = m.serialize(self.material, tokenizer, 90)
        for doc in docs:
            body = doc[:-len(m.SFX)]
            self.assertEqual(body, tokenizer.decode(tokenizer.encode(body)[:90]))


class BuildReportTests(unittest.TestCase):
    def setUp(self):
        self.material, self.models, self.mined, _ = pipeline_parts()
        self.written = [m.questions(case_for(self.mined, op), self.models) for op in ('current', 'compare', 'count')]
        self.verified = [m.verify(c, self.mined, self.models, TOK, 1000) for c in self.written]
        self.byid = {self.material['material_id']: self.mined}

    def test_training_fields_equivalents_and_joint_components_protected(self):
        qs, pools, groups = m.build(self.verified, self.byid, TOK, 1000)
        self.assertEqual(len(qs), 3)
        self.assertEqual(len(groups), 5)
        for g in groups:
            self.assertEqual(g['provenance'], 'native')
            self.assertEqual(g['split'], self.material['split'])
            self.assertEqual(g['kinds'][g['label']], 'pos')
            q = next(q for q in qs if q['question_id'] == g['qid'])
            for i, label in zip(g['round_indices'], g['kinds']):
                if label != 'pos':
                    self.assertNotIn(i, q['support_rounds'])
            self.assertTrue(all(key in g for key in ('query', 'docs', 'label', 'positive_atoms', 'nsel', 'ncov', 'ngold', 'speakers')))
        self.assertEqual(len(pools[qs[0]['question_id']]['fused']), 7)

    def test_mutation_and_visibility_change_rejected_at_export(self):
        bad = copy.deepcopy(self.verified)
        bad[0]['answer'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'changed'):
            m.build(bad, self.byid, TOK, 1000)
        bad = copy.deepcopy(self.verified)
        bad[0]['kinds'][0] = 'random'
        with self.assertRaisesRegex(ValueError, 'labels or controls changed'):
            m.build(bad, self.byid, TOK, 1000)
        with self.assertRaisesRegex(ValueError, 'changed|truncation'):
            m.build(self.verified, self.byid, TOK, 100)

    def test_positive_first_bounded_groups_cover_all_verified_negatives(self):
        qs, _, groups = m.build(self.verified, self.byid, TOK, 1000, group_size=3)
        for q in qs:
            for atom in q['answer_atoms']:
                steps = [g for g in groups if g['qid'] == q['question_id'] and g['positive_atoms'] == [atom['id']]]
                consumed_negatives = []
                for g in steps:
                    self.assertEqual(g['label'], 0)
                    self.assertEqual(g['kinds'][0], 'pos')
                    self.assertLessEqual(len(g['docs']), 3)
                    self.assertEqual(len(g['teacher']), len(g['docs']))
                    self.assertIn(g['round_indices'][0], atom['equivalent_rounds'])
                    consumed_negatives += g['round_indices'][1:]
                    self.assertEqual(g['candidate_provenance'][0]['provenance'], 'native')
                    self.assertIn('facts', g['candidate_provenance'][0])
                self.assertEqual(sorted(consumed_negatives), [i for i in range(7) if i not in q['support_rounds']])

    def test_report_flow_coverage_all_costs_and_full_review_dump(self):
        qs, _, groups = m.build(self.verified, self.byid, TOK, 1000)
        summary = m.report([self.material], [self.mined], self.written, self.verified, qs, groups, self.models.records)
        self.assertEqual(summary['source_flow']['blogs']['frozen'], 0)
        self.assertEqual(summary['source_flow']['wildchat']['update_bearing_materials'], 1)
        self.assertEqual(summary['negative_material_coverage']['wrong_person'], 1)
        self.assertEqual(summary['controls']['positive_correct_rate'], 1)
        self.assertEqual(summary['controls']['evidence_removed_correct_rate'], 0)
        self.assertFalse(summary['go'])
        self.assertIsNone(summary['gates']['blinded_dual_review'])
        self.assertEqual(summary['costs']['calls'], len(self.models.records))
        review = m.review_dump(qs, self.verified, self.byid)
        self.assertIn(self.material['segments'][0]['text'], review)
        self.assertIn('wrong_person', review)
        self.assertIn('Judge votes', review)
        self.assertGreater(summary['groups_per_100k_source_tokens'], 0)

    def test_all_zero_report(self):
        summary = m.report([], [], [], [], [], [], [])
        self.assertIsNone(summary['tokens_per_accepted_question'])
        self.assertIsNone(summary['groups_per_100k_source_tokens'])
        self.assertIsNone(summary['controls']['positive_correct_rate'])
        self.assertFalse(summary['gates']['teacher_cost'])
        self.assertEqual(set(summary['source_flow']), set(m.SOURCES))


class PersistenceAndCliTests(unittest.TestCase):
    def test_usage_retry_rejected_costs_unknown_and_dedup(self):
        count = 0

        def transport(family, prompt, p):
            nonlocal count
            count += 1
            return ('{broken' if count == 1 else '{"ok": true}'), dict(prompt_tokens=11, completion_tokens=7)
        with tempfile.TemporaryDirectory() as tmp:
            model = m.Models(out=tmp, transport=transport)
            model.ask(0, 'test', dict(task='test'), lambda r: r.get('ok') is True)
            events = m.read_rows(Path(tmp) / 'usage.jsonl')
            self.assertEqual(len(events), 2)
            self.assertFalse(events[0]['ok'])
            costs = m.cost_summary(events + events)
            self.assertEqual(costs['total_tokens'], 36)
            self.assertEqual(costs['failed_calls'], 1)
            events.append(dict(call_id='error', task='failed-network', ok=False, usage=None))
            self.assertEqual(m.cost_summary(events)['unknown_usage_calls'], 1)

    def test_checkpoint_resume_fingerprint_parallel_and_failure(self):
        calls = []
        rows = [dict(material_id='m' + str(i), x=i) for i in range(4)]

        def fn(row):
            calls.append(row['material_id'])
            if row['x'] == 2:
                raise ValueError('expected failure')
            return dict(row, ok=True)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            c = m.Checkpoints(tmp, ['model'], dict(setting=1))
            a = c.run('mine', rows, fn, par=3)
            b = c.run('mine', rows, fn, par=2)
            self.assertEqual(a, b)
            self.assertEqual(len(calls), 4)
            self.assertEqual(a[2]['error'], 'expected failure')
            c.run('mine', [dict(rows[0], x=9)], fn)
            self.assertEqual(len(calls), 5)
            m.Checkpoints(tmp, ['different-model']).run('mine', [rows[0]], fn)
            self.assertEqual(len(calls), 6)
            m.Checkpoints(tmp, ['model'], dict(setting=1), retry_failed=True).run('mine', rows, fn)
            self.assertEqual(len(calls), 7)  # only the previously failed material

    def test_same_model_cannot_vote_as_multiple_families(self):
        with self.assertRaisesRegex(ValueError, 'multiple judge families'):
            m.Models(['qwen@1', 'gemma@2', 'qwen@3'])

    def test_corrupt_checkpoint_is_recomputed(self):
        calls = []

        def fn(row):
            calls.append(row)
            return dict(row, ok=True)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            cache = m.Checkpoints(tmp, ['fixture'])
            cache.run('mine', [dict(material_id='m')], fn)
            path = next((Path(tmp) / 'checkpoints' / 'mine').glob('*.json'))
            path.write_text('{truncated')
            cache.run('mine', [dict(material_id='m')], fn)
            self.assertEqual(len(calls), 2)

    def test_cli_all_resume_and_individual_report(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            blog = root / 'blogs.zip'
            xml = ''.join(f'<date>{i:02d},January,2004</date><post>' +
                          ('My editor is Vim.' if i == 1 else 'My editor is now Emacs.' if i == 20 else 'I read books today.') + PAD +
                          '</post>' for i in range(1, 31))
            with zipfile.ZipFile(blog, 'w') as z:
                z.writestr('author.xml', xml.encode('latin-1'))
            wild = root / 'wild.jsonl'
            m.write_rows(wild, [dict(id='conv', language='English', timestamp=1700000000000,
                                    conversation=[dict(role='user' if i % 2 == 0 else 'assistant',
                                                       timestamp=1700000000000 + i * 1000,
                                                       content=('My editor is Vim.' if i == 0 else
                                                                'My editor is now Emacs.' if i == 8 else 'I read books today.') + PAD)
                                                  for i in range(30)])])
            irc = root / 'irc.jsonl'
            m.write_rows(irc, [dict(id=i, date='2004-12-25', channel='ubuntu',
                                   raw=f'[{10 + i // 60:02d}:{i % 60:02d}] <alice> ' + ('My editor is Vim.' if i == 0 else
                                         'My editor is now Emacs.' if i == 3 else 'I read books today.') + PAD, connections=[])
                              for i in range(300)])
            out = root / 'out'
            args = ['all', '--mock', '--out', str(out), '--blogs', str(blog), '--wildchat', str(wild), '--irc', str(irc),
                    '--n-blogs', '1', '--n-wildchat', '1', '--n-irc', '1', '--par', '3', '--token-limit', '1000']
            m.main(args)
            self.assertEqual(len(m.read_rows(out / 'materials.jsonl')), 3)
            self.assertTrue((out / 'human_review.md').exists())
            self.assertTrue((out / 'report.md').exists())
            self.assertGreater(len(m.read_rows(out / 'groups.jsonl')), 0)
            calls = len(m.read_rows(out / 'usage.jsonl'))
            m.main(args)
            self.assertEqual(len(m.read_rows(out / 'usage.jsonl')), calls)
            m.main(['report', '--out', str(out)])
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['costs']['calls'], calls)
            self.assertEqual(report['source_flow']['blogs']['frozen'], 1)

    def test_insert_arm_stub_and_live_requires_real_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                m.main(['all', '--out', tmp, '--insert-arm'])
            with self.assertRaises(SystemExit):
                m.main(['verify', '--out', tmp])


if __name__ == '__main__':
    unittest.main()
