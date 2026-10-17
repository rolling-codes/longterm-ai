"""Retrieval engine: BM25 ranking, token-budget bounding, and gateway-bounded tool."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from longterm_ai import (  # noqa: E402
    Action, ArtifactStore, Criterion, DeterministicVerifier, FakeModel, Framework,
    FunctionTool, RiskClass, RuleBasedPolicy, ScriptedTurn, Task, ToolContext,
    ToolDefinition, ToolGateway, ToolRegistry, ToolStatus,
)
from longterm_ai.contracts import ToolCall  # noqa: E402
from longterm_ai.retrieval import (  # noqa: E402
    BM25Index, Document, MemoryStore, Retriever, _tokenize, assemble_context, make_retrieve_tool,
)


class Tokenize(unittest.TestCase):
    def test_lowercases_and_splits_punctuation(self):
        self.assertEqual(_tokenize("Build FAILED: module_7.py!"), ["build", "failed", "module_7", "py"])

    def test_empty(self):
        self.assertEqual(_tokenize("  ... "), [])


class BM25(unittest.TestCase):
    def test_empty_index_scores_nothing(self):
        self.assertEqual(BM25Index().score(["anything"]), {})

    def test_ranks_on_topic_above_off_topic(self):
        store = MemoryStore()
        store.index(Document("a", "the database migration failed on the users table"))
        store.index(Document("b", "the cat sat quietly on a warm windowsill all day"))
        store.index(Document("c", "migration of the database schema for users completed"))
        hits = store.search("database migration users", k=3)
        ids = [d.id for d, _ in hits]
        self.assertEqual(set(ids), {"a", "c"})   # both on-topic match; off-topic "b" scores nothing
        self.assertNotIn("b", ids)
        self.assertGreater(hits[0][1], 0.0)

    def test_reindex_replaces_not_duplicates(self):
        store = MemoryStore()
        store.index(Document("a", "alpha beta"))
        store.index(Document("a", "gamma delta"))         # same id, new text
        self.assertEqual(store._index.n_docs, 1)
        self.assertEqual(store.search("alpha", k=5), [])  # old tokens gone
        self.assertEqual([d.id for d, _ in store.search("gamma", k=5)], ["a"])


class RetrieverBudget(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore()
        # Three ~equally relevant docs; each ~10 rough tokens (40 chars).
        for i in "abc":
            self.store.index(Document(i, ("keyword " * 5).strip(), source=i))
        self.r = Retriever(self.store)

    def test_k_limits_count(self):
        self.assertEqual(len(self.r.retrieve("keyword", k=2)), 2)

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.r.retrieve("nomatch", k=5), [])

    def test_token_budget_drops_tail(self):
        # Each doc is 39 chars -> rough_tokens ~= 9. Budget 20 fits only two.
        kept = self.r.retrieve("keyword", k=3, token_budget=20)
        self.assertEqual(len(kept), 2)

    def test_no_budget_returns_all_hits(self):
        self.assertEqual(len(self.r.retrieve("keyword", k=3)), 3)


class AssembleContext(unittest.TestCase):
    def test_empty_on_no_hits(self):
        r = Retriever(MemoryStore())
        self.assertEqual(assemble_context(r, "anything", budget=100), "")

    def test_includes_provenance_header(self):
        store = MemoryStore()
        store.index(Document("ref1", "disk usage at 92 percent", source="shell"))
        out = assemble_context(Retriever(store), "disk usage", budget=100)
        self.assertIn("[shell]", out)
        self.assertIn("disk usage at 92 percent", out)


class RetrieveToolThroughGateway(unittest.TestCase):
    def test_tool_is_bounded_and_policy_gated(self):
        tmp = tempfile.mkdtemp()
        store = MemoryStore()
        store.index(Document("ref1", "the error was a null pointer in the parser", source="shell"))
        reg = ToolRegistry()
        reg.register(make_retrieve_tool(store))
        policy = RuleBasedPolicy(reg.risk_map())          # READ_ONLY -> allowed
        gw = ToolGateway(reg, policy, artifact_store=ArtifactStore(tmp))
        r = gw.execute(Action("retrieve", args={"query": "null pointer parser"}),
                       ToolContext(task_id="t", state_dir=Path(tmp)))
        self.assertEqual(r.status, ToolStatus.SUCCESS)
        self.assertIn("null pointer", r.summary + str(r.data))

    def test_cannot_run_when_denied(self):
        tmp = tempfile.mkdtemp()
        reg = ToolRegistry()
        # Force the retrieve tool to a prohibited class via a hand-built definition.
        reg.register(FunctionTool(
            ToolDefinition(name="retrieve", arg_schema={"query": "str"}, risk=RiskClass.PROHIBITED),
            lambda a, c: "secret",
        ))
        policy = RuleBasedPolicy(reg.risk_map())
        gw = ToolGateway(reg, policy, artifact_store=ArtifactStore(tmp))
        r = gw.execute(Action("retrieve", args={"query": "x"}),
                       ToolContext(task_id="t", state_dir=Path(tmp)))
        self.assertEqual(r.status, ToolStatus.DENIED)


class FrameworkIntegration(unittest.TestCase):
    """store= indexes truncated artifacts; store=None leaves behavior unchanged."""

    def _run(self, *, with_store):
        state = Path(tempfile.mkdtemp(prefix="lt_retr_"))
        (state / "build.log").write_text("BUILD OK\n")
        big = "\n".join(f"line {i}: null pointer in parser module_{i}" for i in range(200))
        reg = ToolRegistry()
        reg.register(FunctionTool(
            ToolDefinition(name="read_log", arg_schema={"path": "str"},
                           risk=RiskClass.READ_ONLY, output_limit=200),
            lambda a, c: big,
        ))
        policy = RuleBasedPolicy(reg.risk_map())
        model = FakeModel([
            ScriptedTurn(tool_calls=[ToolCall(name="read_log", args={"path": "build.log"}, id="op1")]),
            ScriptedTurn(content="done"),
        ])
        task = Task(
            objective="read log",
            acceptance_criteria=[Criterion("ok", "file_exists", {"path": "build.log"})],
            max_steps=5,
        )
        store = MemoryStore() if with_store else None
        fw = Framework(model, reg, policy, DeterministicVerifier(base_dir=state),
                       state_dir=state, store=store)
        result = fw.run(task)
        return result, store

    def test_store_indexes_truncated_artifact(self):
        result, store = self._run(with_store=True)
        self.assertTrue(result.verified_complete)
        assert store is not None
        self.assertGreater(store._index.n_docs, 0)                  # evidence was indexed
        hits = Retriever(store).retrieve("null pointer parser", k=1)
        self.assertEqual(len(hits), 1)
        self.assertIn("null pointer", hits[0].text)

    def test_none_store_still_completes(self):
        result, store = self._run(with_store=False)
        self.assertTrue(result.verified_complete)
        self.assertIsNone(store)


if __name__ == "__main__":
    unittest.main()
