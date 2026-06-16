import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import asyncio
from typing import Dict, List, Optional, Union
import uuid
import http.client
import json
import os
import sys

SERPER_KEY = os.environ.get('SERPER_KEY_ID')


def _abort_on_serper_fatal_error(results: dict, query: str, endpoint: str):
    if isinstance(results, dict) and "statusCode" in results and isinstance(results.get("statusCode"), int) and results["statusCode"] >= 400:
        msg = results.get("message", "unknown")
        sys.stderr.write(
            f"\n[FATAL][Serper:{endpoint}] API returned error statusCode={results['statusCode']}: {msg}\n"
            f"  query: {query!r}\n"
            f"  Aborting entire process to prevent generating garbage rollouts.\n"
            f"  Check SERPER_KEY_ID balance at https://serper.dev/dashboard\n"
        )
        sys.stderr.flush()
        os._exit(1)

def _is_blocked_url(url: str) -> bool:
    """
    Check if the URL should be blocked to prevent data leakage.
    (Adapted from MiroThinker.)

    Args:
        url: The URL to check

    Returns:
        bool: True if the URL should be blocked, False otherwise
    """
    if not url:
        return False
    blocked_patterns = [
        "huggingface.co/datasets",
        "huggingface.co/spaces",
        "grok.com/share",
        "modelscope.cn/datasets",
    ]
    return any(pattern in url for pattern in blocked_patterns)

@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
    
    def google_search_with_serp(self, query: str):
        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)
        
        conn = http.client.HTTPSConnection("google.serper.dev")
        
        if contains_chinese_basic(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn"
            })
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
            })
        
        headers = {
            'X-API-KEY': SERPER_KEY,
            'Content-Type': 'application/json'
        }
        
        for i in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Google search Timeout, return None, Please try again later."
                continue
        
        data = res.read()
        results = json.loads(data.decode("utf-8"))

        _abort_on_serper_fatal_error(results, query, "search")

        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    # Filter out HuggingFace dataset/space URLs (as in MiroThinker).
                    if _is_blocked_url(page.get("link", "")):
                        continue  # skip blocked URL
                    
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)
            
            # Add search-result statistics.
            total_found = len(results.get("organic", []))
            filtered_count = total_found - len(web_snippets)
            
            if len(web_snippets) > 0:
                content = f"A Google search for '{query}' found {total_found} results "
                if filtered_count > 0:
                    content += f"(filtered {filtered_count} HuggingFace links) "
                content += f":\n\n## Web Results\n" + "\n\n".join(web_snippets)
                return content
            else:
                return f"No results found for '{query}' after filtering. Try with a more general query."
                
        except Exception as e:
            return f"No results found for '{query}'. Error: {str(e)}. Try with a more general query."
    
    def search_with_serp(self, query: str):
        result = self.google_search_with_serp(query)
        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            # single query
            response = self.search_with_serp(query)
        else:
            # multiple queries
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.search_with_serp(q))
            response = "\n=======\n".join(responses)
            
        return response