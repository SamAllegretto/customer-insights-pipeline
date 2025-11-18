# src/tagging/tagger.py
from typing import List, Dict, Optional
from src.agents.llm_agent import ChatAgent
from src.config.settings import Settings

class FeedbackTagger:
    """Tag customer feedback with predefined issue categories."""
    
    # Predefined Vessi product issue categories
    DEFAULT_CATEGORIES = [
        "Waterproof Leak",
        "Upper Knit Separation",
        "Insole Issue",
        "Inner Lining Rip",
        "Glue Gap",
        "Discolouration",
        "Sizes not standard",
        "Toe Area too narrow",
        "Toe area too big",
        "Instep too small",
        "instep too high",
        "shoe too narrow",
        "shoe too wide",
        "half size requests",
        "no heel lock/heel slip",
        "Lack of grip/traction",
        "Squeaky sound",
        "Not breathable enough",
        "hard to take off",
        "hard to put on",
        "Lack of support",
        "Heel Cup - too big",
        "Smelly",
        "Back Heel Rubbing",
        "Warping",
        "Stains",
        "Looks different than picture/ ugly/ not what was expected",
        "blisters",
        "Too Bulky",
        "Too Heavy"
    ]
    
    def __init__(self, config: Settings, custom_categories: Optional[List[str]] = None):
        """
        Initialize the feedback tagger.
        
        Args:
            config: Settings object with OpenAI configuration
            custom_categories: Optional custom category list (uses DEFAULT_CATEGORIES if None)
        """
        self.agent = ChatAgent(config)
        self.categories = custom_categories if custom_categories else self.DEFAULT_CATEGORIES
    
    def tag_single(self, feedback_text: str, allow_multiple: bool = True) -> List[str]:
        """
        Tag a single piece of customer feedback.
        
        Args:
            feedback_text: Customer feedback text
            allow_multiple: If True, returns list of all applicable tags; if False, returns single-item list with best tag
            
        Returns:
            List of category strings
        """
        return self.agent.tag_feedback(feedback_text, self.categories, allow_multiple)
    
    def tag_batch(self, feedback_texts: List[str], allow_multiple: bool = True, 
                  batch_size: int = 10) -> List[List[str]]:
        """
        Tag multiple pieces of feedback efficiently.
        
        Args:
            feedback_texts: List of customer feedback texts
            allow_multiple: If True, returns lists of multiple tags; if False, returns single-tag lists
            batch_size: Number of feedback items to process per API call (max 20 recommended)
            
        Returns:
            List of tag lists (each feedback item gets a list of tags)
        """
        all_results = []
        
        # Process in batches for efficiency
        for i in range(0, len(feedback_texts), batch_size):
            batch = feedback_texts[i:i + batch_size]
            batch_results = self.agent.tag_feedback_batch(batch, self.categories, allow_multiple)
            all_results.extend(batch_results)
        
        return all_results
    
    def tag_with_confidence(self, feedback_text: str, num_iterations: int = 3) -> Dict[str, float]:
        """
        Tag feedback multiple times and return confidence scores for each category.
        
        Args:
            feedback_text: Customer feedback text
            num_iterations: Number of times to tag (more iterations = more reliable confidence)
            
        Returns:
            Dict mapping category names to confidence scores (0.0 to 1.0)
        """
        all_tags = []
        
        for _ in range(num_iterations):
            tags = self.tag_single(feedback_text, allow_multiple=True)
            all_tags.extend(tags)
        
        # Calculate confidence as frequency of appearance
        confidence_scores = {}
        for category in set(all_tags):
            if category != "Uncategorized":  # Exclude uncategorized from confidence scores
                confidence_scores[category] = all_tags.count(category) / num_iterations
        
        # Sort by confidence descending
        return dict(sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True))
    
    def get_category_distribution(self, feedback_texts: List[str], 
                                 allow_multiple: bool = True) -> Dict[str, int]:
        """
        Get distribution of categories across multiple feedback items.
        
        Args:
            feedback_texts: List of customer feedback texts
            allow_multiple: If True, counts all tags per feedback; if False, counts only primary tag
            
        Returns:
            Dict mapping category names to counts
        """
        all_tags = self.tag_batch(feedback_texts, allow_multiple)
        
        distribution = {cat: 0 for cat in self.categories}
        distribution["Uncategorized"] = 0
        
        for tag_list in all_tags:
            for tag in tag_list:
                if tag in distribution:
                    distribution[tag] += 1
                else:
                    distribution["Uncategorized"] += 1
        
        # Remove categories with zero counts and sort by count descending
        return dict(sorted(
            {k: v for k, v in distribution.items() if v > 0}.items(),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def get_multi_tag_stats(self, feedback_texts: List[str]) -> Dict[str, any]:
        """
        Get statistics about multi-tag occurrences.
        
        Args:
            feedback_texts: List of customer feedback texts
            
        Returns:
            Dict with statistics about tagging patterns
        """
        all_tags = self.tag_batch(feedback_texts, allow_multiple=True)
        
        tag_counts = [len(tags) for tags in all_tags]
        
        return {
            'total_feedback': len(feedback_texts),
            'avg_tags_per_feedback': sum(tag_counts) / len(tag_counts) if tag_counts else 0,
            'max_tags_single_feedback': max(tag_counts) if tag_counts else 0,
            'single_tag_count': sum(1 for c in tag_counts if c == 1),
            'multi_tag_count': sum(1 for c in tag_counts if c > 1),
            'no_tag_count': sum(1 for tags in all_tags if tags == ["Uncategorized"]),
            'distribution': self.get_category_distribution(feedback_texts, allow_multiple=True)
        }
    
    def add_category(self, category: str):
        """Add a new category to the tagger."""
        if category not in self.categories:
            self.categories.append(category)
    
    def remove_category(self, category: str):
        """Remove a category from the tagger."""
        if category in self.categories:
            self.categories.remove(category)
    
    def get_categories(self) -> List[str]:
        """Get current list of categories."""
        return self.categories.copy()


if __name__ == "__main__":
    # Initialize
    from src.config.settings import Settings
    from src.tagging.tagger import FeedbackTagger

    config = Settings()
    tagger = FeedbackTagger(config)

    print("=== Testing Multiple Tags Per Review ===\n")

    # Test 1: Single feedback with multiple issues
    feedback = "The shoe leaked after just one week and the sizing was way off. Also very uncomfortable."
    print(f"Feedback: {feedback}")
    tags = tagger.tag_single(feedback, allow_multiple=True)
    print(f"Tags: {tags}\n")

    # Test 2: Batch processing with multiple tags
    feedbacks = [
        "Water got in after a few wears and they smell terrible",
        "The toe box is way too narrow for my feet and causes blisters",
        "Smells bad after wearing them and the insole came loose",
        "Perfect fit, no issues"  # Should get "Uncategorized" or empty
    ]
    
    print("=== Batch Tagging (Multiple Tags) ===")
    batch_tags = tagger.tag_batch(feedbacks, allow_multiple=True)
    for i, (fb, tags) in enumerate(zip(feedbacks, batch_tags), 1):
        print(f"{i}. {fb}")
        print(f"   Tags: {tags}\n")

    # Test 3: Get confidence scores for multi-issue feedback
    print("=== Confidence Scores (5 iterations) ===")
    confidence = tagger.tag_with_confidence(feedback, num_iterations=5)
    for category, score in confidence.items():
        print(f"{category}: {score:.1%}")
    print()

    # Test 4: Distribution analysis
    print("=== Category Distribution ===")
    distribution = tagger.get_category_distribution(feedbacks, allow_multiple=True)
    for category, count in distribution.items():
        print(f"{category}: {count}")
    print()

    # Test 5: Multi-tag statistics
    print("=== Multi-Tag Statistics ===")
    stats = tagger.get_multi_tag_stats(feedbacks)
    print(f"Total feedback: {stats['total_feedback']}")
    print(f"Avg tags per feedback: {stats['avg_tags_per_feedback']:.2f}")
    print(f"Max tags in single feedback: {stats['max_tags_single_feedback']}")
    print(f"Single-tag feedback: {stats['single_tag_count']}")
    print(f"Multi-tag feedback: {stats['multi_tag_count']}")
    print(f"No valid tags: {stats['no_tag_count']}")