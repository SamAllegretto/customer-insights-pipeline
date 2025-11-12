"""Unit tests for the ChatAgent class."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.agents.llm_agent import ChatAgent
from src.config.settings import Settings


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Settings)
    config.openai_api_key = "test-api-key"
    config.openai_llm_model = "gpt-5-nano"
    return config


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch('src.agents.llm_agent.OpenAI') as mock_client:
        yield mock_client


class TestChatAgent:
    """Test ChatAgent class."""
    
    def test_agent_initialization(self, mock_config, mock_openai_client):
        """Test ChatAgent initializes correctly."""
        agent = ChatAgent(mock_config)
        assert agent.config == mock_config
        assert agent.model == "gpt-5-nano"
        mock_openai_client.assert_called_once_with(api_key="test-api-key")
    
    def test_chat_single(self, mock_config, mock_openai_client):
        """Test single prompt chat."""
        mock_client_instance = mock_openai_client.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        agent = ChatAgent(mock_config)
        result = agent.chat_single("Test prompt")
        
        assert result == "Test response"
        mock_client_instance.chat.completions.create.assert_called_once()
    
    def test_chat_with_messages(self, mock_config, mock_openai_client):
        """Test chat with message list."""
        mock_client_instance = mock_openai_client.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        agent = ChatAgent(mock_config)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        result = agent.chat(messages)
        
        assert result == "Response"
    
    def test_label_cluster(self, mock_config, mock_openai_client):
        """Test cluster labeling."""
        mock_client_instance = mock_openai_client.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Waterproof Issues"))]
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        agent = ChatAgent(mock_config)
        feedback_texts = [
            "Shoes leaked after one day",
            "Water got through the material",
            "Not waterproof at all"
        ]
        label = agent.label_cluster(feedback_texts)
        
        assert label == "Waterproof Issues"
    
    def test_tag_feedback_single_tag(self, mock_config, mock_openai_client):
        """Test tagging feedback with single tag."""
        mock_client_instance = mock_openai_client.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Waterproof Leak"))]
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        agent = ChatAgent(mock_config)
        categories = ["Waterproof Leak", "Sizes not standard", "Too Heavy"]
        result = agent.tag_feedback("Water got in", categories, allow_multiple=False)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "Waterproof Leak"
    
    def test_tag_feedback_multiple_tags(self, mock_config, mock_openai_client):
        """Test tagging feedback with multiple tags."""
        mock_client_instance = mock_openai_client.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(
            content='["Waterproof Leak", "Sizes not standard"]'
        ))]
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        agent = ChatAgent(mock_config)
        categories = ["Waterproof Leak", "Sizes not standard", "Too Heavy"]
        result = agent.tag_feedback("Leaking and too small", categories, allow_multiple=True)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert "Waterproof Leak" in result
        assert "Sizes not standard" in result
    
    def test_parse_json_array_valid_json(self, mock_config, mock_openai_client):
        """Test parsing valid JSON array."""
        agent = ChatAgent(mock_config)
        categories = ["Cat1", "Cat2", "Cat3"]
        result = agent._parse_json_array('["Cat1", "Cat2"]', categories)
        
        assert result == ["Cat1", "Cat2"]
    
    def test_parse_json_array_with_markdown(self, mock_config, mock_openai_client):
        """Test parsing JSON array with markdown code blocks."""
        agent = ChatAgent(mock_config)
        categories = ["Cat1", "Cat2", "Cat3"]
        result = agent._parse_json_array('```json\n["Cat1", "Cat3"]\n```', categories)
        
        assert result == ["Cat1", "Cat3"]
    
    def test_parse_json_array_empty(self, mock_config, mock_openai_client):
        """Test parsing empty JSON array."""
        agent = ChatAgent(mock_config)
        categories = ["Cat1", "Cat2"]
        result = agent._parse_json_array('[]', categories)
        
        assert result == []
    
    def test_parse_single_tag_exact_match(self, mock_config, mock_openai_client):
        """Test parsing single tag with exact match."""
        agent = ChatAgent(mock_config)
        categories = ["Waterproof Leak", "Too Heavy"]
        result = agent._parse_single_tag("Waterproof Leak", categories)
        
        assert result == "Waterproof Leak"
    
    def test_parse_single_tag_case_insensitive(self, mock_config, mock_openai_client):
        """Test parsing single tag with case-insensitive match."""
        agent = ChatAgent(mock_config)
        categories = ["Waterproof Leak", "Too Heavy"]
        result = agent._parse_single_tag("waterproof leak", categories)
        
        assert result == "Waterproof Leak"
    
    def test_parse_single_tag_no_match(self, mock_config, mock_openai_client):
        """Test parsing single tag with no match returns Uncategorized."""
        agent = ChatAgent(mock_config)
        categories = ["Waterproof Leak", "Too Heavy"]
        result = agent._parse_single_tag("Something Random", categories)
        
        assert result == "Uncategorized"
    
    def test_tag_feedback_batch(self, mock_config, mock_openai_client):
        """Test batch tagging of feedback."""
        mock_client_instance = mock_openai_client.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(
            content='{"1": ["Waterproof Leak"], "2": ["Too Heavy"]}'
        ))]
        mock_client_instance.chat.completions.create.return_value = mock_response
        
        agent = ChatAgent(mock_config)
        feedback_texts = ["Leaking shoes", "Very heavy"]
        categories = ["Waterproof Leak", "Too Heavy", "Sizes not standard"]
        result = agent.tag_feedback_batch(feedback_texts, categories, allow_multiple=True)
        
        assert len(result) == 2
        assert result[0] == ["Waterproof Leak"]
        assert result[1] == ["Too Heavy"]
