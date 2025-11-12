"""Unit tests for the Embedder class."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.embedding.embedder import Embedder
from src.config.settings import Settings


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Settings)
    config.openai_api_key = "test-api-key"
    config.openai_embedding_model = "text-embedding-3-small"
    return config


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch('src.embedding.embedder.OpenAI') as mock_client:
        yield mock_client


class TestEmbedder:
    """Test Embedder class."""
    
    def test_embedder_initialization(self, mock_config, mock_openai_client):
        """Test Embedder initializes correctly."""
        embedder = Embedder(mock_config)
        assert embedder.config == mock_config
        assert embedder.model == "text-embedding-3-small"
        mock_openai_client.assert_called_once_with(api_key="test-api-key")
    
    def test_embed_single(self, mock_config, mock_openai_client):
        """Test embedding a single text."""
        # Setup mock response
        mock_client_instance = mock_openai_client.return_value
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client_instance.embeddings.create.return_value = mock_response
        
        embedder = Embedder(mock_config)
        result = embedder.embed_single("Test text")
        
        assert len(result) == 1536
        assert result[0] == 0.1
        mock_client_instance.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["Test text"]
        )
    
    def test_embed_texts(self, mock_config, mock_openai_client):
        """Test embedding multiple texts."""
        # Setup mock response
        mock_client_instance = mock_openai_client.return_value
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
            MagicMock(embedding=[0.3] * 1536)
        ]
        mock_client_instance.embeddings.create.return_value = mock_response
        
        embedder = Embedder(mock_config)
        texts = ["Text 1", "Text 2", "Text 3"]
        result = embedder.embed_texts(texts)
        
        assert len(result) == 3
        assert len(result[0]) == 1536
        assert result[0][0] == 0.1
        assert result[1][0] == 0.2
        assert result[2][0] == 0.3
        mock_client_instance.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=texts
        )
    
    def test_embed_texts_empty_list(self, mock_config, mock_openai_client):
        """Test embedding an empty list."""
        mock_client_instance = mock_openai_client.return_value
        mock_response = MagicMock()
        mock_response.data = []
        mock_client_instance.embeddings.create.return_value = mock_response
        
        embedder = Embedder(mock_config)
        result = embedder.embed_texts([])
        
        assert result == []
