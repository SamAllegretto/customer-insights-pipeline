"""Tests for multithreading optimization in Embedder."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.embedding.embedder import Embedder
from src.config.settings import Settings
import time


@pytest.fixture
def mock_config():
    """Create a mock configuration with max_workers."""
    config = Mock(spec=Settings)
    config.openai_api_key = "test-api-key"
    config.openai_embedding_model = "text-embedding-3-small"
    config.max_workers = 3
    return config


@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return [f"Sample text {i}" for i in range(10)]


class TestMultithreadedEmbedding:
    """Test multithreaded embedding functionality."""
    
    @patch('src.embedding.embedder.OpenAI')
    def test_embedder_initialization_with_max_workers(self, mock_openai, mock_config):
        """Test Embedder initializes with max_workers from config."""
        embedder = Embedder(mock_config)
        
        assert embedder.max_workers == 3
        mock_openai.assert_called_once_with(api_key="test-api-key")
    
    @patch('src.embedding.embedder.OpenAI')
    def test_embed_texts_single_batch(self, mock_openai, mock_config):
        """Test embed_texts with texts that fit in a single batch (no threading)."""
        # Setup mock
        mock_client = mock_openai.return_value
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
            MagicMock(embedding=[0.3] * 1536)
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        # Test
        embedder = Embedder(mock_config)
        texts = ["Text 1", "Text 2", "Text 3"]
        results = embedder.embed_texts(texts, batch_size=100)
        
        # Verify
        assert len(results) == 3
        assert results[0] == [0.1] * 1536
        assert results[1] == [0.2] * 1536
        assert results[2] == [0.3] * 1536
        
        # Should only call API once
        assert mock_client.embeddings.create.call_count == 1
    
    @patch('src.embedding.embedder.OpenAI')
    def test_embed_texts_multiple_batches(self, mock_openai, mock_config):
        """Test embed_texts with multiple batches using threading."""
        # Setup mock
        mock_client = mock_openai.return_value
        
        # Mock to return different embeddings for each batch
        call_count = [0]
        
        def mock_create(**kwargs):
            result = MagicMock()
            batch_size = len(kwargs['input'])
            result.data = [MagicMock(embedding=[call_count[0] + i * 0.1] * 1536) for i in range(batch_size)]
            call_count[0] += 1
            return result
        
        mock_client.embeddings.create.side_effect = mock_create
        
        # Test with 6 texts and batch_size=2 (creates 3 batches)
        embedder = Embedder(mock_config)
        texts = [f"Text {i}" for i in range(6)]
        results = embedder.embed_texts(texts, batch_size=2)
        
        # Verify
        assert len(results) == 6
        assert all(isinstance(r, list) for r in results)
        assert all(len(r) == 1536 for r in results)
        
        # Should call API 3 times (one per batch)
        assert mock_client.embeddings.create.call_count == 3
    
    @patch('src.embedding.embedder.OpenAI')
    def test_embed_texts_maintains_order(self, mock_openai, mock_config):
        """Test that results maintain order despite parallel processing."""
        # Setup mock
        mock_client = mock_openai.return_value
        
        call_count = [0]
        
        def mock_create(**kwargs):
            batch_num = call_count[0]
            call_count[0] += 1
            # Add delay to simulate async behavior
            time.sleep(0.01)
            result = MagicMock()
            batch_size = len(kwargs['input'])
            # Each batch gets embeddings with the batch number as the first value
            result.data = [MagicMock(embedding=[batch_num, i] + [0.0] * 1534) for i in range(batch_size)]
            return result
        
        mock_client.embeddings.create.side_effect = mock_create
        
        # Test with 9 texts and batch_size=3 (creates 3 batches)
        embedder = Embedder(mock_config)
        texts = [f"Text {i}" for i in range(9)]
        results = embedder.embed_texts(texts, batch_size=3)
        
        # Verify ordering: first 3 should have batch_num=0, next 3 batch_num=1, last 3 batch_num=2
        assert len(results) == 9
        assert results[0][0] == 0  # First batch
        assert results[0][1] == 0  # First item in batch
        assert results[1][0] == 0  # First batch
        assert results[1][1] == 1  # Second item in batch
        assert results[2][0] == 0  # First batch
        assert results[2][1] == 2  # Third item in batch
        assert results[3][0] == 1  # Second batch
        assert results[6][0] == 2  # Third batch
    
    @patch('src.embedding.embedder.OpenAI')
    def test_embed_texts_empty_list(self, mock_openai, mock_config):
        """Test embed_texts with empty input."""
        embedder = Embedder(mock_config)
        results = embedder.embed_texts([])
        
        # Should return empty list without calling API
        assert results == []
        assert mock_openai.return_value.embeddings.create.call_count == 0
    
    @patch('src.embedding.embedder.OpenAI')
    def test_embed_texts_error_handling(self, mock_openai, mock_config):
        """Test embed_texts handles errors gracefully."""
        # Setup mock to raise an exception
        mock_client = mock_openai.return_value
        
        call_count = [0]
        
        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("API Error")
            result = MagicMock()
            batch_size = len(kwargs['input'])
            result.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(batch_size)]
            return result
        
        mock_client.embeddings.create.side_effect = mock_create
        
        # Test with 6 texts and batch_size=2 (creates 3 batches)
        embedder = Embedder(mock_config)
        texts = [f"Text {i}" for i in range(6)]
        
        # Should raise RuntimeError with context
        with pytest.raises(RuntimeError) as exc_info:
            embedder.embed_texts(texts, batch_size=2)
        
        assert "Error processing embedding batch" in str(exc_info.value)
        assert "API Error" in str(exc_info.value)
    
    @patch('src.embedding.embedder.OpenAI')
    def test_embed_batch_method(self, mock_openai, mock_config):
        """Test the _embed_batch internal method."""
        # Setup mock
        mock_client = mock_openai.return_value
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536)
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        # Test
        embedder = Embedder(mock_config)
        batch = ["Text 1", "Text 2"]
        results = embedder._embed_batch(batch)
        
        # Verify
        assert len(results) == 2
        assert results[0] == [0.1] * 1536
        assert results[1] == [0.2] * 1536
        
        # Verify API was called with correct parameters
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=batch
        )
    
    @patch('src.embedding.embedder.OpenAI')
    def test_embed_texts_with_custom_batch_size(self, mock_openai, mock_config):
        """Test embed_texts with custom batch_size parameter."""
        # Setup mock
        mock_client = mock_openai.return_value
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(5)]
        mock_client.embeddings.create.return_value = mock_response
        
        # Test with custom batch_size
        embedder = Embedder(mock_config)
        texts = [f"Text {i}" for i in range(5)]
        results = embedder.embed_texts(texts, batch_size=5)
        
        # Verify
        assert len(results) == 5
        # Should process in single batch since all fit
        assert mock_client.embeddings.create.call_count == 1
    
    @patch('src.embedding.embedder.OpenAI')
    def test_embed_single_unchanged(self, mock_openai, mock_config):
        """Test that embed_single method still works correctly."""
        # Setup mock
        mock_client = mock_openai.return_value
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.5] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        
        # Test
        embedder = Embedder(mock_config)
        result = embedder.embed_single("Single text")
        
        # Verify
        assert result == [0.5] * 1536
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["Single text"]
        )
