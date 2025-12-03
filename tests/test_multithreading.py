"""Tests for multithreading optimization in FeedbackTagger."""
import pytest
from unittest.mock import Mock, patch, call
from src.agents.llm_agent import FeedbackTagger, ChatAgent
from src.config.settings import Settings
import time
from concurrent.futures import ThreadPoolExecutor


@pytest.fixture
def mock_config():
    """Create a mock configuration with max_workers."""
    config = Mock(spec=Settings)
    config.openai_api_key = "test-api-key"
    config.openai_llm_model = "gpt-5-nano"
    config.max_workers = 3
    return config


@pytest.fixture
def feedback_texts():
    """Create sample feedback texts for testing."""
    return [
        "Shoes are leaking after 2 weeks",
        "Great shoes, very comfortable!",
        "Size runs small, had to return",
        "Not waterproof at all",
        "Too heavy for daily use",
        "Perfect fit and style",
    ]


class TestMultithreadedTagging:
    """Test multithreaded tagging functionality."""
    
    @patch('src.agents.llm_agent.ChatAgent')
    def test_tagger_initialization_with_max_workers(self, mock_chat_agent, mock_config):
        """Test FeedbackTagger initializes with max_workers from config."""
        tagger = FeedbackTagger(mock_config)
        
        assert tagger.max_workers == 3
        mock_chat_agent.assert_called_once_with(mock_config)
    
    @patch('src.agents.llm_agent.ChatAgent')
    def test_tag_batch_uses_threadpool(self, mock_chat_agent, mock_config, feedback_texts):
        """Test that tag_batch uses ThreadPoolExecutor for parallel processing."""
        # Setup mock agent
        mock_agent_instance = mock_chat_agent.return_value
        
        # Mock tag_feedback_batch to return appropriate results for each batch
        def mock_tag_feedback_batch(batch, categories, allow_multiple):
            return [["Tag1"] for _ in batch]
        
        mock_agent_instance.tag_feedback_batch.side_effect = mock_tag_feedback_batch
        
        # Create tagger and process feedback
        tagger = FeedbackTagger(mock_config)
        results = tagger.tag_batch(feedback_texts, allow_multiple=True, batch_size=2)
        
        # Verify results
        assert len(results) == len(feedback_texts)
        assert all(isinstance(r, list) for r in results)
        
        # Verify tag_feedback_batch was called multiple times (once per batch)
        # With 6 texts and batch_size=2, we should have 3 batches
        assert mock_agent_instance.tag_feedback_batch.call_count == 3
    
    @patch('src.agents.llm_agent.ChatAgent')
    def test_tag_batch_maintains_order(self, mock_chat_agent, mock_config, feedback_texts):
        """Test that results are returned in the original order despite parallel processing."""
        # Setup mock agent
        mock_agent_instance = mock_chat_agent.return_value
        
        # Mock to return different tags for each batch to verify ordering
        call_count = [0]
        
        def mock_tag_feedback_batch(batch, categories, allow_multiple):
            result = [[f"Tag{call_count[0]}"] for _ in batch]
            call_count[0] += 1
            # Add delay to simulate async behavior
            time.sleep(0.01)
            return result
        
        mock_agent_instance.tag_feedback_batch.side_effect = mock_tag_feedback_batch
        
        # Create tagger and process feedback
        tagger = FeedbackTagger(mock_config)
        results = tagger.tag_batch(feedback_texts, allow_multiple=True, batch_size=2)
        
        # Verify results length
        assert len(results) == len(feedback_texts)
        
        # Results should be in order based on input position, not completion order
        # First 2 items should have Tag0, next 2 should have Tag1, last 2 should have Tag2
        assert results[0] == ["Tag0"]
        assert results[1] == ["Tag0"]
        assert results[2] == ["Tag1"]
        assert results[3] == ["Tag1"]
        assert results[4] == ["Tag2"]
        assert results[5] == ["Tag2"]
    
    @patch('src.agents.llm_agent.ChatAgent')
    def test_tag_batch_error_handling(self, mock_chat_agent, mock_config):
        """Test that tag_batch handles errors gracefully."""
        # Setup mock agent
        mock_agent_instance = mock_chat_agent.return_value
        
        # Mock to raise an exception for one batch
        call_count = [0]
        
        def mock_tag_feedback_batch(batch, categories, allow_multiple):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("API Error")
            return [["Tag"] for _ in batch]
        
        mock_agent_instance.tag_feedback_batch.side_effect = mock_tag_feedback_batch
        
        # Create tagger and process feedback
        tagger = FeedbackTagger(mock_config)
        feedback_texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        results = tagger.tag_batch(feedback_texts, allow_multiple=True, batch_size=2)
        
        # Verify results - failed batch should return "Uncategorized"
        assert len(results) == 4
        assert results[0] == ["Tag"]
        assert results[1] == ["Tag"]
        # Second batch failed, should return "Uncategorized"
        assert results[2] == ["Uncategorized"]
        assert results[3] == ["Uncategorized"]
    
    @patch('src.agents.llm_agent.ChatAgent')
    def test_tag_batch_with_single_batch(self, mock_chat_agent, mock_config):
        """Test that tag_batch works correctly with a single batch (no parallelism needed)."""
        # Setup mock agent
        mock_agent_instance = mock_chat_agent.return_value
        mock_agent_instance.tag_feedback_batch.return_value = [
            ["Tag1"], ["Tag2"]
        ]
        
        # Create tagger and process feedback
        tagger = FeedbackTagger(mock_config)
        feedback_texts = ["Text 1", "Text 2"]
        results = tagger.tag_batch(feedback_texts, allow_multiple=True, batch_size=10)
        
        # Verify results
        assert len(results) == 2
        assert results[0] == ["Tag1"]
        assert results[1] == ["Tag2"]
        
        # Should only be called once since all texts fit in one batch
        assert mock_agent_instance.tag_feedback_batch.call_count == 1
    
    @patch('src.agents.llm_agent.ChatAgent')
    def test_tag_batch_with_allow_multiple_false(self, mock_chat_agent, mock_config, feedback_texts):
        """Test that tag_batch correctly passes allow_multiple parameter."""
        # Setup mock agent
        mock_agent_instance = mock_chat_agent.return_value
        mock_agent_instance.tag_feedback_batch.return_value = [
            ["Tag"] for _ in range(2)
        ]
        
        # Create tagger and process feedback
        tagger = FeedbackTagger(mock_config)
        results = tagger.tag_batch(feedback_texts, allow_multiple=False, batch_size=2)
        
        # Verify allow_multiple=False was passed to all calls
        for call_args in mock_agent_instance.tag_feedback_batch.call_args_list:
            args, kwargs = call_args
            # Third argument is allow_multiple
            assert args[2] == False
    
    @patch('src.agents.llm_agent.ChatAgent')
    def test_tag_batch_empty_list(self, mock_chat_agent, mock_config):
        """Test that tag_batch handles empty input gracefully."""
        # Setup mock agent
        mock_agent_instance = mock_chat_agent.return_value
        
        # Create tagger and process empty feedback list
        tagger = FeedbackTagger(mock_config)
        results = tagger.tag_batch([], allow_multiple=True, batch_size=10)
        
        # Verify results
        assert len(results) == 0
        
        # Should not call tag_feedback_batch at all for empty input
        assert mock_agent_instance.tag_feedback_batch.call_count == 0
