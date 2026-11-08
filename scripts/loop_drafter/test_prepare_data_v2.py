import copy
import unittest

from .prepare_data_v2 import (APPS, MATH, OverlapIndex, apps_prompt, arc_prompt, capture_plan,
                              freeze_ids, record, select_records, select_reserves)


class Tokenizer:
    chat_template = True

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['return_dict'] is False
        return [1] + self.encode(messages[0]['content']) + [2]


def sample(source, sid, **kwargs):
    return record(source, sid, f'{sid} alpha{sid} beta{sid} gamma{sid} delta{sid} epsilon{sid}',
                  'code' if source == APPS else 'math', **kwargs)


class PrepareDataTests(unittest.TestCase):
    def test_templates_preserve_original_labels_and_skeleton_without_gold(self):
        row = {'question': 'Which item?', 'choices': {'label': ['1', '2', '3', '4', '5'],
                                                     'text': ['one', 'two', 'three', 'four', 'five']},
               'answerKey': 'GOLD_MUST_NOT_APPEAR'}
        text = arc_prompt(row)
        self.assertIn('5. five', text)
        self.assertNotIn(row['answerKey'], text)
        app = {'question': 'Compute a result.', 'starter_code': 'def solve(x):\n    ',
               'solutions': 'SECRET_SOLUTION', 'input_output': 'SECRET_TEST'}
        text = apps_prompt(app)
        self.assertTrue(text.endswith('def solve(x):\n    '))
        self.assertNotIn('SECRET', text)
        self.assertTrue(apps_prompt(dict(app, starter_code='')).endswith('### Python solution:\n'))

    def test_exact_and_high_jaccard_cover_short_and_long_texts(self):
        index = OverlapIndex(.90)
        text = ' '.join(f'word{i}' for i in range(100))
        index.add(text, {'id': 'benchmark', 'source': 'test'})
        self.assertEqual(index.matches(text.upper())[0]['match'], 'exact')
        self.assertEqual(index.matches('changed ' + ' '.join(text.split()[1:]))[0]['match'], 'fivegram_jaccard')
        self.assertFalse(index.matches(' '.join(f'different{i}' for i in range(100))))
        index.add('  Short\u2028request! ', {'id': 'short', 'source': 'old'})
        self.assertEqual(index.matches('short request!')[0]['id'], 'short')

    def test_selection_retains_legacy_reports_overlap_excludes_new_duplicates(self):
        old = sample('old', 'old')
        benchmark = dict(old, id='test:old', source='test')
        duplicate_old = dict(sample('new', 'duplicate'), prompt=old['prompt'])
        duplicate_old['benchmark_text'] = old['prompt']
        grouped1 = sample('new', 'first', group='shared')
        grouped2 = sample('new', 'second', group='shared')
        safe = sample('new', 'safe')
        rows, audit = select_records([old], {'new': [duplicate_old, grouped1, grouped2, safe]},
                                     [benchmark], Tokenizer(), '/model', new_counts={'new': 2})
        self.assertEqual(len(rows), 3)
        self.assertIn('old', [r['source'] for r in rows])
        self.assertEqual(len(audit['legacy_benchmark_matches']), 1)
        self.assertEqual(len({r['group_id'] for r in rows}), 3)
        self.assertNotIn(duplicate_old['id'], [r['id'] for r in rows])
        again, _ = select_records([old], {'new': [safe, grouped2, grouped1, duplicate_old]},
                                  [benchmark], Tokenizer(), '/model', new_counts={'new': 2})
        self.assertEqual(rows, again)

    def test_round_robin_covers_strata_and_redistributes_exhausted_capacity(self):
        pool = [sample(APPS, f'a{i}', difficulty='introductory', starter_code_present=True) for i in range(5)]
        pool += [sample(APPS, 'b0', difficulty='interview', starter_code_present=False)]
        rows, audit = select_records([], {APPS: pool}, [], Tokenizer(), '/model', new_counts={APPS: 4})
        self.assertEqual(sum(r['starter_code_present'] for r in rows), 3)
        self.assertEqual(sum(audit['selected_strata'].values()), 4)
        math = [sample(MATH, f'm{i}', math_type='Algebra', math_level=str(i + 1)) for i in range(5)]
        rows, _ = select_records([], {MATH: math}, [], Tokenizer(), '/model', new_counts={MATH: 5})
        self.assertEqual({r['math_level'] for r in rows}, {'1', '2', '3', '4', '5'})

    def test_shortage_fails_without_source_substitution_and_long_prompts_are_not_truncated(self):
        long = sample('new', 'long')
        long['prompt'] = 'x' * 2049
        with self.assertRaisesRegex(ValueError, 'Insufficient eligible'):
            select_records([], {'new': [long]}, [], Tokenizer(), '/model', new_counts={'new': 1})
        self.assertIsNone(freeze_ids(long, Tokenizer(), '/model', 2048))

    def test_capture_plan_reuses_only_audited_same_input_eos_records(self):
        rows = [freeze_ids(sample('old', sid), Tokenizer(), '/model', 2048) for sid in ('eos', 'cap')]
        for row in rows:
            row['origin'] = 'retained_v1'
        rows.append(freeze_ids(sample('new', 'fresh'), Tokenizer(), '/model', 2048))
        audit = {'status': 'passed', 'records': []}
        for row, action, terminal in zip(rows, ('reuse_complete', 'recapture_to_512'), ('native_eos', 'cap256_non_eos')):
            audit['records'].append(dict(id=row['id'], path='/old.pt', generated_length=256,
                                         terminal=terminal, reuse_decision=action,
                                         actual_prompt_ids_match_frozen_prompt=True,
                                         prompt_input_ids_sha256=row['input_ids_sha256']))
        plan = capture_plan(rows, audit)
        self.assertEqual(plan['counts'], {'reuse_complete': 1, 'recapture_to_512': 1, 'capture_new_to_512': 1})
        self.assertEqual(plan['max_new_tokens'], 512)
        bad = copy.deepcopy(audit)
        bad['records'][0]['prompt_input_ids_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'actual legacy target inputs'):
            capture_plan(rows, bad)
        bad['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'must pass'):
            capture_plan(rows, bad)

    def test_reserves_keep_training_immutable_and_report_source_shortage(self):
        training = [freeze_ids(sample('old', 'trained'), Tokenizer(), '/model', 2048)]
        original = copy.deepcopy(training)
        benchmark = sample('test', 'heldout')
        bad = dict(sample('new', 'bad'), prompt=benchmark['prompt'], benchmark_text=benchmark['prompt'])
        pools = {'old': [sample('old', 'trained'), sample('old', 'available')],
                 'new': [bad, sample('new', 'good1'), sample('new', 'good2')]}
        reserves, report = select_reserves(training, pools, [benchmark], Tokenizer(), '/model', limit=2)
        self.assertEqual(training, original)
        self.assertEqual(report['actual_counts'], {'old': 1, 'new': 2})
        self.assertEqual(report['shortages'], {'old': 1})
        self.assertEqual(len(reserves), 3)
        self.assertTrue(all(r['panel_role'] == 'reserve' and r['origin'] == 'reserve_v2' for r in reserves))
        self.assertNotIn(bad['id'], [r['id'] for r in reserves])


if __name__ == '__main__':
    unittest.main()
