"""Tests for rate limit retry logic with exponential backoff."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import RateLimitError
from src.embedding.embedder import Embedder
from src.agents.llm_agent import ChatAgent
from src.config.settings import Settings
import time


def create_rate_limit_error():
    """Helper to create a properly mocked RateLimitError."""
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_response.request = mock_request
    return RateLimitError("Rate limit exceeded", response=mock_response, body=None)


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Settings)
    config.openai_api_key = "test-api-key"
    config.openai_embedding_model = "text-embedding-3-small"
    config.openai_llm_model = "gpt-3.5-turbo"
    config.max_workers = 3
    return config


class TestEmbedderRateLimitRetry:
    """Test rate limit retry logic for Embedder."""
    
    @patch('src.embedding.embedder.time.sleep')
    @patch('src.embedding.embedder.OpenAI')
    def test_embed_batch_retries_on_rate_limit(self, mock_openai, mock_sleep, mock_config):
        """Test that _embed_batch retries with exponential backoff on rate limit."""
        # Setup mock to fail twice with rate limit, then succeed
        mock_client = mock_openai.return_value
        call_count = [0]
        
        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise create_rate_limit_error()
            # Third call succeeds
            result = MagicMock()
            result.data = [MagicMock(embedding=[0.1] * 1536)]
            return result
        
        mock_client.embeddings.create.side_effect = mock_create
        
        # Test
        embedder = Embedder(mock_config)
        result = embedder._embed_batch(["Test text"])
        
        # Verify
        assert len(result) == 1
        assert result[0] == [0.1] * 1536
        
        # Should have called API 3 times (2 failures + 1 success)
        assert mock_client.embeddings.create.call_count == 3
        
        # Should have slept twice with exponential backoff (1s, 2s)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)  # First retry: 1 * 2^0
        mock_sleep.assert_any_call(2.0)  # Second retry: 1 * 2^1
    
    @patch('src.embedding.embedder.time.sleep')
    @patch('src.embedding.embedder.OpenAI')
    def test_embed_batch_raises_after_max_retries(self, mock_openai, mock_sleep, mock_config):
        """Test that _embed_batch raises RateLimitError after max retries."""
        # Setup mock to always fail with rate limit
        mock_client = mock_openai.return_value
        mock_client.embeddings.create.side_effect = create_rate_limit_error()
        
        # Test
        embedder = Embedder(mock_config)
        
        with pytest.raises(RateLimitError):
            embedder._embed_batch(["Test text"])
        
        # Should have called API 5 times (max_retries)
        assert mock_client.embeddings.create.call_count == 5
        
        # Should have slept 4 times (between 5 attempts)
        assert mock_sleep.call_count == 4
    
    @patch('src.embedding.embedder.OpenAI')
    def test_embed_batch_no_retry_on_other_errors(self, mock_openai, mock_config):
        """Test that _embed_batch doesn't retry on non-rate-limit errors."""
        # Setup mock to fail with a different error
        mock_client = mock_openai.return_value
        mock_client.embeddings.create.side_effect = ValueError("Some other error")
        
        # Test
        embedder = Embedder(mock_config)
        
        with pytest.raises(ValueError):
            embedder._embed_batch(["Test text"])
        
        # Should only have called API once (no retries)
        assert mock_client.embeddings.create.call_count == 1
    
    @patch('src.embedding.embedder.time.sleep')
    @patch('src.embedding.embedder.OpenAI')
    def test_embed_single_uses_retry_logic(self, mock_openai, mock_sleep, mock_config):
        """Test that embed_single uses retry logic through _embed_batch."""
        # Setup mock to fail once with rate limit, then succeed
        mock_client = mock_openai.return_value
        call_count = [0]
        
        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise create_rate_limit_error()
            # Second call succeeds
            result = MagicMock()
            result.data = [MagicMock(embedding=[0.5] * 1536)]
            return result
        
        mock_client.embeddings.create.side_effect = mock_create
        
        # Test
        embedder = Embedder(mock_config)
        result = embedder.embed_single("Test text")
        
        # Verify
        assert result == [0.5] * 1536
        
        # Should have called API 2 times (1 failure + 1 success)
        assert mock_client.embeddings.create.call_count == 2
        
        # Should have slept once
        assert mock_sleep.call_count == 1


class TestChatAgentRateLimitRetry:
    """Test rate limit retry logic for ChatAgent."""
    
    @patch('src.agents.llm_agent.time.sleep')
    @patch('src.agents.llm_agent.OpenAI')
    def test_chat_retries_on_rate_limit(self, mock_openai, mock_sleep, mock_config):
        """Test that chat retries with exponential backoff on rate limit."""
        # Setup mock to fail twice with rate limit, then succeed
        mock_client = mock_openai.return_value
        call_count = [0]
        
        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise create_rate_limit_error()
            # Third call succeeds
            result = MagicMock()
            result.choices = [MagicMock(message=MagicMock(content="Success"))]
            return result
        
        mock_client.chat.completions.create.side_effect = mock_create
        
        # Test
        agent = ChatAgent(mock_config)
        result = agent.chat([{"role": "user", "content": "Test"}])
        
        # Verify
        assert result == "Success"
        
        # Should have called API 3 times (2 failures + 1 success)
        assert mock_client.chat.completions.create.call_count == 3
        
        # Should have slept twice with exponential backoff (1s, 2s)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)  # First retry: 1 * 2^0
        mock_sleep.assert_any_call(2.0)  # Second retry: 1 * 2^1
    
    @patch('src.agents.llm_agent.time.sleep')
    @patch('src.agents.llm_agent.OpenAI')
    def test_chat_raises_after_max_retries(self, mock_openai, mock_sleep, mock_config):
        """Test that chat raises RateLimitError after max retries."""
        # Setup mock to always fail with rate limit
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = create_rate_limit_error()
        
        # Test
        agent = ChatAgent(mock_config)
        
        with pytest.raises(RateLimitError):
            agent.chat([{"role": "user", "content": "Test"}])
        
        # Should have called API 5 times (max_retries)
        assert mock_client.chat.completions.create.call_count == 5
        
        # Should have slept 4 times (between 5 attempts)
        assert mock_sleep.call_count == 4
    
    @patch('src.agents.llm_agent.OpenAI')
    def test_chat_no_retry_on_other_errors(self, mock_openai, mock_config):
        """Test that chat doesn't retry on non-rate-limit errors."""
        # Setup mock to fail with a different error
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = ValueError("Some other error")
        
        # Test
        agent = ChatAgent(mock_config)
        
        with pytest.raises(ValueError):
            agent.chat([{"role": "user", "content": "Test"}])
        
        # Should only have called API once (no retries)
        assert mock_client.chat.completions.create.call_count == 1
    
    @patch('src.agents.llm_agent.time.sleep')
    @patch('src.agents.llm_agent.OpenAI')
    def test_chat_single_uses_retry_logic(self, mock_openai, mock_sleep, mock_config):
        """Test that chat_single uses retry logic through chat."""
        # Setup mock to fail once with rate limit, then succeed
        mock_client = mock_openai.return_value
        call_count = [0]
        
        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise create_rate_limit_error()
            # Second call succeeds
            result = MagicMock()
            result.choices = [MagicMock(message=MagicMock(content="Response"))]
            return result
        
        mock_client.chat.completions.create.side_effect = mock_create
        
        # Test
        agent = ChatAgent(mock_config)
        result = agent.chat_single("Test prompt")
        
        # Verify
        assert result == "Response"
        
        # Should have called API 2 times (1 failure + 1 success)
        assert mock_client.chat.completions.create.call_count == 2
        
        # Should have slept once
        assert mock_sleep.call_count == 1
    
    @patch('src.agents.llm_agent.time.sleep')
    @patch('src.agents.llm_agent.OpenAI')
    def test_exponential_backoff_delays(self, mock_openai, mock_sleep, mock_config):
        """Test that exponential backoff delays follow correct pattern."""
        # Setup mock to fail 4 times, then succeed
        mock_client = mock_openai.return_value
        call_count = [0]
        
        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 4:
                raise create_rate_limit_error()
            # Fifth call succeeds
            result = MagicMock()
            result.choices = [MagicMock(message=MagicMock(content="Success"))]
            return result
        
        mock_client.chat.completions.create.side_effect = mock_create
        
        # Test
        agent = ChatAgent(mock_config)
        result = agent.chat([{"role": "user", "content": "Test"}])
        
        # Verify exponential backoff: 1s, 2s, 4s, 8s
        assert mock_sleep.call_count == 4
        mock_sleep.assert_any_call(1.0)   # 1 * 2^0
        mock_sleep.assert_any_call(2.0)   # 1 * 2^1
        mock_sleep.assert_any_call(4.0)   # 1 * 2^2
        mock_sleep.assert_any_call(8.0)   # 1 * 2^3
