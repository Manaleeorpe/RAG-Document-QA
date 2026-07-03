"""
test_resilience.py — retry and partial-failure resilience tests

Categories covered
──────────────────
• Graph API  — retry on Timeout then success
• Graph API  — all 3 attempts fail → exception re-raised
• Graph API  — non-retryable error is NOT retried
• OpenAI/DeepSeek — successful callLLM round-trip (mocked)
• OpenAI/DeepSeek — RateLimitError (429) propagates when no retry wrapper
• OpenAI/DeepSeek — tenacity retry wrapper recovers from one 429
• Partial ingestion — 5 threads, 1 fails → 4 processed, 1 error recorded

Theory notes
────────────
• mocker   — pytest-mock fixture; preferred over @patch decorators because it
             auto-reverts after each test with no context-manager boilerplate.
• monkeypatch — used for environment variables; mocker.patch for objects/methods.
• tenacity — already in requirements.txt; used to define retry policies that are
             tested here and can be promoted to production code as needed.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pytest
import requests
import tenacity

import openai
from calLLM import callLLM


# ─────────────────────────────────────────────────────────────────────────────
# Hypothetical Graph API client
# In a real project this would live in app/graph_client.py.
# Defined here so the retry tests are self-contained.
# ─────────────────────────────────────────────────────────────────────────────

class GraphEmailClient:
    """Minimal interface for fetching email threads via Microsoft Graph API."""

    def __init__(self, access_token: str):
        self.access_token = access_token

    def get_email_threads(self, folder: str = "inbox", top: int = 10):
        raise NotImplementedError("Implemented by the real HTTP layer")


@tenacity.retry(
    wait=tenacity.wait_none(),  # instant retries — no sleep in tests
    stop=tenacity.stop_after_attempt(3),
    retry=tenacity.retry_if_exception_type((requests.Timeout, ConnectionError)),
    reraise=True,
)
def fetch_threads_with_retry(client: GraphEmailClient, folder: str = "inbox"):
    """Thin retry wrapper around GraphEmailClient.get_email_threads()."""
    return client.get_email_threads(folder=folder)


# ─────────────────────────────────────────────────────────────────────────────
# Batch ingestion helper (simulates parallel email-thread processing)
# ─────────────────────────────────────────────────────────────────────────────

def ingest_threads_batch(process_fn, thread_ids: list) -> dict:
    """
    Process each thread_id concurrently.
    Returns {"processed": [...], "errors": [{"thread_id": ..., "error": ...}]}.
    One thread failing does not stop the others.
    """
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=len(thread_ids)) as pool:
        future_to_id = {pool.submit(process_fn, tid): tid for tid in thread_ids}
        for future in as_completed(future_to_id):
            tid = future_to_id[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append({"thread_id": tid, "error": str(exc)})
    return {"processed": results, "errors": errors}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a minimal openai.RateLimitError without a live HTTP connection
# ─────────────────────────────────────────────────────────────────────────────

def _make_rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError(
        "Rate limit exceeded. Please try again later.",
        response=response,
        body={"error": {"message": "rate_limit_exceeded", "type": "tokens"}},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graph API retry tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphAPIRetry:
    def test_retry_on_timeout_then_success(self, mocker):
        """First call raises Timeout; second succeeds — verify 2 total calls."""
        mock_client = mocker.MagicMock(spec=GraphEmailClient)
        mock_client.get_email_threads.side_effect = [
            requests.Timeout("Connection timed out"),
            [{"id": "1", "subject": "Q3 Report"}],
        ]

        result = fetch_threads_with_retry(mock_client)

        assert result == [{"id": "1", "subject": "Q3 Report"}]
        assert mock_client.get_email_threads.call_count == 2

    def test_first_attempt_succeeds_no_retry_needed(self, mocker):
        mock_client = mocker.MagicMock(spec=GraphEmailClient)
        mock_client.get_email_threads.return_value = [{"id": "42", "subject": "Budget"}]

        result = fetch_threads_with_retry(mock_client)

        assert len(result) == 1
        mock_client.get_email_threads.assert_called_once()

    def test_retry_exhausted_after_3_consecutive_timeouts(self, mocker):
        """Three Timeout failures must exhaust the policy and re-raise the last error."""
        mock_client = mocker.MagicMock(spec=GraphEmailClient)
        mock_client.get_email_threads.side_effect = requests.Timeout("Persistent timeout")

        with pytest.raises(requests.Timeout):
            fetch_threads_with_retry(mock_client)

        assert mock_client.get_email_threads.call_count == 3

    def test_connection_error_is_also_retried(self, mocker):
        """ConnectionError (e.g. DNS failure) is in the retry policy alongside Timeout."""
        mock_client = mocker.MagicMock(spec=GraphEmailClient)
        mock_client.get_email_threads.side_effect = [
            ConnectionError("DNS resolution failed"),
            [{"id": "7", "subject": "Network recovered"}],
        ]

        result = fetch_threads_with_retry(mock_client)

        assert result[0]["id"] == "7"
        assert mock_client.get_email_threads.call_count == 2

    def test_non_retryable_exception_is_not_retried(self, mocker):
        """ValueError is outside the retry policy — must propagate immediately."""
        mock_client = mocker.MagicMock(spec=GraphEmailClient)
        mock_client.get_email_threads.side_effect = ValueError("Malformed token")

        with pytest.raises(ValueError, match="Malformed token"):
            fetch_threads_with_retry(mock_client)

        mock_client.get_email_threads.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI / DeepSeek retry tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAIRetry:
    """
    callLLM now uses LangChain's ChatOpenAI (DeepSeek). These tests patch
    `calLLM.ChatOpenAI`; the chat model is invoked as `prompt | llm`, so the
    mock's `.invoke()` is what produces the AIMessage-like response.
    """

    @staticmethod
    def _mock_llm(mocker, content="ok", invoke_side_effect=None):
        """
        A ChatOpenAI stand-in whose .invoke returns an object with .content.

        spec=Runnable so `prompt | llm` keeps it as the sequence's last step
        (otherwise a bare MagicMock gets wrapped in a RunnableLambda and .invoke
        is never called).
        """
        from langchain_core.runnables import Runnable

        ai_message = mocker.MagicMock()
        ai_message.content = content
        llm = mocker.MagicMock(spec=Runnable)
        if invoke_side_effect is not None:
            llm.invoke.side_effect = invoke_side_effect
        else:
            llm.invoke.return_value = ai_message
        return llm

    def test_successful_llm_call_returns_answer(self, mocker):
        """Happy-path: mocked chat model returns content; callLLM extracts it."""
        llm = self._mock_llm(mocker, content="Revenue grew by 15%.")
        mocker.patch("calLLM.ChatOpenAI", return_value=llm)

        result = callLLM("Context: Revenue grew 15%.", "What happened to revenue?")

        assert result == "Revenue grew by 15%."

    def test_llm_call_uses_deepseek_model(self, mocker):
        """callLLM must construct ChatOpenAI with the 'deepseek-chat' model."""
        captured = {}

        def capture(*args, **kwargs):
            captured.update(kwargs)
            return self._mock_llm(mocker)

        mocker.patch("calLLM.ChatOpenAI", side_effect=capture)

        callLLM("ctx", "question?")

        assert captured.get("model") == "deepseek-chat"
        assert captured.get("base_url") == "https://api.deepseek.com"

    def test_llm_call_passes_context_and_question_in_messages(self, mocker):
        """The rendered prompt must contain both the context and the question."""
        llm = self._mock_llm(mocker, content="answer")
        mocker.patch("calLLM.ChatOpenAI", return_value=llm)

        callLLM("My context paragraph.", "My specific question?")

        # The piped runnable invokes the chat model with the formatted messages.
        args, kwargs = llm.invoke.call_args
        rendered = str(args[0] if args else kwargs)
        assert "My context paragraph." in rendered
        assert "My specific question?" in rendered

    def test_rate_limit_429_propagates_without_retry(self, mocker):
        """callLLM has no retry policy; a 429 must bubble up unchanged."""
        llm = self._mock_llm(mocker, invoke_side_effect=_make_rate_limit_error())
        mocker.patch("calLLM.ChatOpenAI", return_value=llm)

        with pytest.raises(openai.RateLimitError):
            callLLM("context", "question?")

    def test_tenacity_retry_wrapper_recovers_from_one_429(self, mocker):
        """Recommended pattern: wrap callLLM with tenacity. 429 then success."""

        @tenacity.retry(
            wait=tenacity.wait_none(),
            stop=tenacity.stop_after_attempt(3),
            retry=tenacity.retry_if_exception_type(openai.RateLimitError),
            reraise=True,
        )
        def resilient_llm(context, question):
            return callLLM(context, question)

        ai_message = mocker.MagicMock()
        ai_message.content = "Answer after retry."
        llm = self._mock_llm(
            mocker, invoke_side_effect=[_make_rate_limit_error(), ai_message]
        )
        mocker.patch("calLLM.ChatOpenAI", return_value=llm)

        result = resilient_llm("Context.", "Question?")

        assert result == "Answer after retry."
        assert llm.invoke.call_count == 2

    def test_tenacity_retry_exhausted_after_3_rate_limits(self, mocker):
        """Three consecutive 429s should exhaust the retry budget and re-raise."""

        @tenacity.retry(
            wait=tenacity.wait_none(),
            stop=tenacity.stop_after_attempt(3),
            retry=tenacity.retry_if_exception_type(openai.RateLimitError),
            reraise=True,
        )
        def resilient_llm(context, question):
            return callLLM(context, question)

        llm = self._mock_llm(mocker, invoke_side_effect=_make_rate_limit_error())
        mocker.patch("calLLM.ChatOpenAI", return_value=llm)

        with pytest.raises(openai.RateLimitError):
            resilient_llm("ctx", "q?")

        assert llm.invoke.call_count == 3

    def test_api_key_read_from_environment(self, mocker, monkeypatch):
        """callLLM must read DEEPSEEK_API_KEY from the environment, not hardcode it."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-12345")

        captured = {}

        def capture(*args, **kwargs):
            captured["api_key"] = kwargs.get("api_key")
            return self._mock_llm(mocker)

        mocker.patch("calLLM.ChatOpenAI", side_effect=capture)

        callLLM("ctx", "q?")

        key = captured.get("api_key")
        # ChatOpenAI may wrap the key in a SecretStr; compare the raw value.
        key_value = key.get_secret_value() if hasattr(key, "get_secret_value") else key
        assert key_value == "test-key-12345"


# ─────────────────────────────────────────────────────────────────────────────
# Partial ingestion tests — 5 threads, 1 fails → 4 processed
# ─────────────────────────────────────────────────────────────────────────────

class TestPartialIngestion:
    THREAD_IDS = ["t1", "t2", "t3", "t4", "t5"]

    def test_5_threads_1_fail_yields_4_processed_and_1_error(self):
        failing_id = "t3"

        def mock_process(thread_id: str):
            if thread_id == failing_id:
                raise ValueError(f"Corrupt email body in thread {thread_id}")
            return {"thread_id": thread_id, "status": "ok"}

        outcome = ingest_threads_batch(mock_process, self.THREAD_IDS)

        assert len(outcome["processed"]) == 4
        assert len(outcome["errors"]) == 1
        assert outcome["errors"][0]["thread_id"] == failing_id

    def test_error_dict_contains_thread_id_and_message(self):
        def mock_process(thread_id: str):
            if thread_id == "t2":
                raise RuntimeError("Attachment too large")
            return {"thread_id": thread_id, "status": "ok"}

        outcome = ingest_threads_batch(mock_process, ["t1", "t2", "t3"])

        error = outcome["errors"][0]
        assert "thread_id" in error
        assert "error" in error
        assert "Attachment too large" in error["error"]

    def test_all_threads_succeed_gives_no_errors(self):
        def mock_process(thread_id: str):
            return {"thread_id": thread_id, "status": "ok"}

        outcome = ingest_threads_batch(mock_process, self.THREAD_IDS)

        assert len(outcome["processed"]) == 5
        assert len(outcome["errors"]) == 0

    def test_all_threads_fail_gives_no_processed(self):
        def mock_process(thread_id: str):
            raise ConnectionError("Graph API unreachable")

        outcome = ingest_threads_batch(mock_process, self.THREAD_IDS)

        assert len(outcome["processed"]) == 0
        assert len(outcome["errors"]) == 5

    def test_first_failure_does_not_abort_remaining_threads(self):
        def mock_process(thread_id: str):
            if thread_id == "t1":
                raise ValueError("First thread fails immediately")
            return {"thread_id": thread_id, "status": "ok"}

        outcome = ingest_threads_batch(mock_process, ["t1", "t2", "t3", "t4"])

        assert len(outcome["processed"]) == 3
        assert len(outcome["errors"]) == 1

    def test_processed_results_contain_expected_thread_ids(self):
        failing_id = "t5"

        def mock_process(thread_id: str):
            if thread_id == failing_id:
                raise OSError("Disk full")
            return {"thread_id": thread_id, "status": "ok"}

        outcome = ingest_threads_batch(mock_process, self.THREAD_IDS)
        processed_ids = {r["thread_id"] for r in outcome["processed"]}

        assert processed_ids == {"t1", "t2", "t3", "t4"}
        assert failing_id not in processed_ids
