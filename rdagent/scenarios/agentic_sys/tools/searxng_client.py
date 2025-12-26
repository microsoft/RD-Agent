"""
SearXNG client for web search. Based on the deployment script's search API
"""
import requests
import json
import csv
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any, Optional
from rdagent.log import rdagent_logger as logger
import yaml

class SearxNGClient:
    """
    Client for SearxNG search engine with multi-format support.

    Features:
    - Multiple output formats (json,csv, html)
    - Result filtering by relevant 
    - error handling and retry logic
    - Configuration requirement
    """

    def __init__(self,config_path):
        """
        Initialize SearxNG client
        Args:
            config_path (str): Path to the SearxNG configuration file
        """
        #load configuration
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.load(f)
        else:
            #base configuration
            config = {
                'base_url': "http://localhost:8888",
                'timeout': 30,
                'max_retries': 3,
                'default_format': 'json',
                'relevant_threshold': 0.3
            }

        self.base_url = config.get('base_url', 'http://localhost:8888')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.default_format = config.get('default_format', 'json')
        self.relevance_threshold = config.get('relevant_threshold', 0.3)

        logger.info(f"SearxNGClient initialized with base_url: {self.base_url}")

    def search(self, query, format, categories, engines, languages, time_range,safesearch):
        """Perform web search using SearxNG API."""
        if not query or not query.strip():
            logger.warning("Empty query provided to SearxNGClient.search")
            return self.empty_result(query)
        format = format or self.default_format

        #build search parameters
        params = {
            'q': query,
            'format': format
        }
        if categories:
            params['categories'] = ','.join(categories)
        
        if engines:
            params['engines'] = ','.join(engines)

        if languages != 'auto':
            params['language'] = languages

        if time_range:
            params['time_range'] = time_range
        
        if safesearch > 0:
            params['safesearch'] = safesearch

        #perform search with retry logic
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Searching SearxNG (attempt {attempt + 1}/{self.max_retries}): {query}")
                response = requests.get(
                    f"{self.base_url}/search",
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()

                #Parse response based on format
                if format == 'json':
                    result = response.json()
                elif format == 'csv':
                    result = self.parse_csv_response(response.text, query)
                elif format == 'html':
                    result = self.parse_html_response(response.text, query)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                logger.info(f"Search completed: {len(result.get('results', []))} results")
                return result

            except requests.Timeout:
                logger.warning(f"Search timeout")
                if attempt == self.max_retries - 1:
                    return self.empty_result(query, error="Timeout")
            
            except requests.RequestException as e:
                logger.error(f"Search request failed: {e}")
                if attempt == self.max_retries - 1:
                    return self.empty_result(query, error=str(e))
                
            except Exception as e:
                logger.error(f"Error processing search response: {e}")
                return self.empty_result(query, error=str(e))
        return self.empty_result(query)
    
    def search_json(self, query, **kwargs):
        """Search with JSON output"""
        return self.search(query, format = 'json', **kwargs)
    
    def search_with_filter(
            self,
            query,
            min_score,
            max_results,
            **kwargs
    ):
        """Search and filter results by relevance score.
        Args:
            query: Search query
            min_score: Minimum relevance score (0 to 1)
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters
        Returns:
            Filtered list of search results
        """

        min_score = min_score or self.relevance_threshold
        result = self.search(query, format = 'json', **kwargs)

        #filter and sort results
        filtered = [
            r for r in result.get('results', [])
            if r.get('relevance_score', 0) >= min_score
        ]

        #sort by score (descending)
        filtered.sort(key=lambda r: r.get('score', 0), reverse=True)

        #limit results
        if max_results:
            filtered = filtered[:max_results]

        return filtered
    
    def empty_result(self, query, error=None):
        """Return empty search result"""
        result = {
            'query': query, 
            'number_of_results': 0,
            'results': [],
            'answers': [],
            'suggestions': [],
            'corrections': [],
            'infoboxes': [],
            'unresponsive_engines': []
        }
        if error:
            result['error'] = error
        return result
    
    def parse_csv_response(self, csv_text, query):
        """Parse CSV response to JSON format"""
        import csv
        from io import StringIO
        results = []
        reader = csv.DictReader(StringIO(csv_text))
        for row in reader:
            results.append({
                'title': row.get('title',''),
                'url': row.get('url',''),
                'content': row.get('content',''),
                'score': 1 / (len(results) + 1)  #simple score based on order
            })
        return {
            'query': query,
            'number_of_results': len(results),
            'results': results,
            'answers': [],
            'suggestions': [],
        }
    
    def parse_html_response(self, html_text, query):
        """Parse HTML response"""
        return {
            'query': query,
            'number_of_results': 0,
            'results': [],
            'answers': [],
            'suggestions': [],
        }
    

def create_searxng_client(config_path=None):
    """Factory method to create SearxNG client"""
    return SearxNGClient(config_path)



        










