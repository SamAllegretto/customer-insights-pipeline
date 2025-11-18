"""Unit tests for the Embedder class."""
import pytest
import os
from src.embedding.embedder import Embedder
from src.config.settings import Settings


@pytest.fixture
def real_config():
    """Create a real configuration with OpenAI API key from environment."""
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    # Create a minimal Settings object with required fields
    # Use default or dummy values for non-OpenAI fields
    return type('Settings', (), {
        'openai_api_key': api_key,
        'openai_embedding_model': 'text-embedding-3-small',
        'openai_llm_model': 'gpt-3.5-turbo',
        'sql_server_host': 'dummy',
        'sql_server_database': 'dummy',
        'sql_server_username': 'dummy',
        'sql_server_password': 'dummy',
        'postgres_host': 'dummy',
        'postgres_database': 'dummy',
        'postgres_username': 'dummy',
        'postgres_password': 'dummy',
    })()


class TestEmbedder:
    """Test Embedder class."""
    
    def test_embedder_initialization(self, real_config):
        """Test Embedder initializes correctly."""
        embedder = Embedder(real_config)
        assert embedder.config == real_config
        assert embedder.model == "text-embedding-3-small"
        assert embedder.client is not None
    
    def test_embed_single(self, real_config):
        """Test embedding a single text with real API."""
        embedder = Embedder(real_config)
        result = embedder.embed_single("Test text")
        
        # Check that we get a valid embedding vector
        assert isinstance(result, list)
        assert len(result) > 0  # Should have embedding dimensions
        # text-embedding-3-small produces 1536-dimensional vectors
        assert len(result) == 1536
        # Check that all values are floats
        assert all(isinstance(x, float) for x in result)
    
    def test_embed_texts(self, real_config):
        """Test embedding multiple texts with real API."""
        embedder = Embedder(real_config)
        texts = ["Text 1", "Text 2", "Text 3"]
        result = embedder.embed_texts(texts)
        
        # Check that we get embeddings for all texts
        assert len(result) == 3
        # Each embedding should be 1536-dimensional
        for embedding in result:
            assert isinstance(embedding, list)
            assert len(embedding) == 1536
            assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_texts_empty_list(self, real_config):
        """Test embedding an empty list with real API."""
        embedder = Embedder(real_config)
        result = embedder.embed_texts([])
        
        assert result == []
