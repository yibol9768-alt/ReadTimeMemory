"""CPU-only tests; no transformers, Docker, or model files required."""
import hashlib
import json
import multiprocessing
import tempfile
import unittest

from plan_flops import TokenCountCache, token_cache_key


def cache_writer(directory, barrier):
    barrier.wait()
    cache = TokenCountCache(directory)
    for n in range(300):
        assert cache.get(str(n), lambda: n * 17) == n * 17
    cache.close()


class TokenCacheTests(unittest.TestCase):
    def test_key_canonical_and_sensitive_only_to_gtok_inputs(self):
        row = {"query": "查询", "docs": ["first", "second", "ignored"], "teacher": [0.1]}
        settings = dict(model="/model", max_q=4, max_d=11, group=2, nsfx=7)
        key = token_cache_key(row, **settings)
        canonical = json.dumps({"query": "查询", "docs": ["first", "second"], **settings},
                               sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(key, hashlib.sha256(canonical.encode("utf-8")).hexdigest())
        changed_metadata = {"teacher": [0.9], "docs": ["first", "second", "different"],
                            "query": "查询", "qid": "new"}
        self.assertEqual(key, token_cache_key(changed_metadata, **settings))
        for field, value in [("query", "other"), ("docs", ["second", "first"])]:
            self.assertNotEqual(key, token_cache_key({**row, field: value}, **settings))
        for field, value in [("model", "/other"), ("max_q", 5), ("max_d", 12),
                             ("group", 3), ("nsfx", 8)]:
            self.assertNotEqual(key, token_cache_key(row, **{**settings, field: value}))

    def test_pending_and_persisted_hits_do_not_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            def unexpected():
                self.fail("cached row was tokenized")
            cache = TokenCountCache(directory)
            self.assertEqual(cache.get("row", lambda: 123), 123)
            self.assertEqual(cache.get("row", unexpected), 123)
            cache.close()
            cache = TokenCountCache(directory)
            self.assertEqual(cache.get("row", unexpected), 123)
            self.assertEqual((cache.hits, cache.misses), (1, 0))
            cache.close()

    def test_two_concurrent_processes_share_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = multiprocessing.get_context("spawn")
            barrier = ctx.Barrier(2)
            processes = [ctx.Process(target=cache_writer, args=(directory, barrier)) for _ in range(2)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join()
                    self.fail("cache writer timed out")
                self.assertEqual(process.exitcode, 0)
            cache = TokenCountCache(directory)
            self.assertEqual(cache.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(cache.db.execute("SELECT COUNT(*) FROM tokens").fetchone()[0], 300)
            self.assertEqual(cache.db.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            cache.close()


if __name__ == "__main__":
    unittest.main()
