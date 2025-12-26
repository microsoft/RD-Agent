"""
Web Search Tool for Agentic System
Intelligent SearxNG for external knowledge retrieval
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.agentic_sys.tools.searxng_client import SearxNGClient

class WebSearchTool:
    """
    High-level web search tool for hypothesis generation

    Features:
    - Query generation from context
    - Multi-source search support
    - Result ranking and filtering
    - Source validation
    - Knowledge extraction 
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize WebSearchTool with SearxNG client

        Args:
            config_path (Optional[Path]): Path to SearxNG configuration file
        """
        self.client = SearxNGClient(config_path)
        #search strategy configuration
        self.max_results_per_query = 5
        self.min_relevance_score = 0.3
        self.preferred_engines = ['duckduckgo', 'google','bing']
        logger.info("WebSearchTool initialized with SearxNGClient")
        

    def search_for_hypothesis(self, task_description, current_gaps, context):
        """
        search for information to support hypothesis generation
        Args:
            task_description: Description of the research task
            current_gaps: List of identified knowledge gaps
            context: Additional context
        Returns:
            List of relevant external sources with metadata
        """
        #generate search queries
        queries = self.generate_queries(task_description, current_gaps, context)

        #execute searches
        all_results = []
        for query in queries:
            try:
                results = self.client.search_with_filter(
                query = query,
                min_score = self.min_relevance_score,
                max_results = self.max_results_per_query,
                engines = self.preferred_engines
            )
                all_results.extend(results)
            except Exception as e:
                    logger.error(f"Error during search for query '{query}': {e}")
                    continue

        #rank and filter results
        ranked_results = self.deduplicate_results(all_results)

        #validate sources
        validated_results = self.validate_sources(ranked_results)

        #extract key information
        enriched = self.extract_knowledge(validated_results)

        logger.info(f"Search completed with {len(enriched)} relevant sources found")
        return enriched
    
    def generate_queries(self, task_description, gaps, context):
        """
        Generate search queries based on task and gaps.
        Strategy:
        1. Primary queries: Direct task-related questions
        2. Gap-specific queries: Target identified knowledge gaps
        3. Exploratory queries, adjacent topics and methodologies
        """
        queries = []
        #Primary query
        if task_description:
            queries.append(task_description[:200])
        
        #Gap-specific queries
        for gap in gaps:
            queries.append(f"how to improve {gap}")
        
        #context-based queries
        if context:
            #If previous experiments failed in specific dimension
            if 'weak_dimension' in context:
                dim = context['weak_dimension']
                queries.append(f"improve {dim} in research system")
                queries.append(f"{dim} optimization techniques")

            # If specific methodology is being used
            if 'methodology' in context:
                method = context['methodology']
                queries.append(f"{method} case studies")

        #Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        logger.info(f"Generated {len(unique_queries)} search queries")
        return unique_queries
    
    def deduplicate_results(self, results):
        """
        Remove duplicate results based on URL
        """
        seen_urls = set()
        deduplicated = []

        #sort by score first
        sorted_results = sorted(
            results,
            key = lambda x: x.get('score', 0), reverse=True
        )
        for result in sorted_results:
            url = result.get('url')
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduplicated.append(result)

    def extract_knowledge(self, results):
        """
        Extract and structure key knowledge from search results
        Args:
            results: Validated search results
        Returns:
            Enriched results with structured knowledge
        """

        enriched = []
        for idx, result in enumerate(results):
            enriched_result = {
                'citation': f"{result.get('title', 'Untitled')} ({result.get('url', 'No URL')})",
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'summary': result.get('content', '')[:300],  # First 300 chars
                'relevance': result.get('score', 0),
                'credibility': result.get('credibility', 0.5),
                'credibility_level': result.get('credibility_level', 'Medium'),
                'source_engine': result.get('engine', 'unknown'),
                'rank': idx
            }
            enriched.append(enriched_result)
        return enriched

    def validate_sources(self, results):
        """
        Validate source credibility.
        """
        validated = []
        for result in results:
            url = result.get('url', '')

            #calculate credibility score
            credibility = self.calculate_credibility(url, result)

            #Add credibility to result
            result['credibility'] = credibility
            result['credibility_level'] = self.credibility_level(credibility)
            validated.append(result)

        validated.sort(
            key = lambda r: (r.get('credibility', 0), r.get('score', 0)),
            reverse=True
        )
        return validated
    
    def calculate_credibility(self, url, result):
        """
        Calculate source credibility score based on heuristics
        """
        score = 0.5  # Baseline
        
        # Domain-based scoring
        if any(domain in url.lower() for domain in ['.edu', '.gov', '.org']):
            score += 0.3
        elif any(domain in url.lower() for domain in ['arxiv.org', 'scholar.google', 'pubmed']):
            score += 0.4  # Academic sources
        elif any(domain in url.lower() for domain in ['medium.com', 'towardsdatascience']):
            score += 0.1  # Tech blogs
        
        # Title-based signals
        title = result.get('title', '').lower()
        if any(keyword in title for keyword in ['research', 'study', 'analysis', 'survey']):
            score += 0.1
        
        # Engine-based trust
        engine = result.get('engine', '')
        if engine in ['google_scholar', 'semantic_scholar']:
            score += 0.2
        
        # Normalize to [0, 1]
        return min(1.0, score)

    def credibility_level(self, score):
        """
        convert credibility score to qualitative label
        """
        if score >= 0.8:
            return 'High'
        elif score >= 0.5:
            return 'Medium'
        else:
            return 'Low'
        
def create_web_search_tool(config_path):
    """
    Factory function to create web search tool
    """
    return WebSearchTool(config_path=config_path)