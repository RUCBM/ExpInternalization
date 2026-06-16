import json
import os
import sys
import time
import random
from typing import Optional, Union, List, Any
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from prompt import EXTRACTOR_PROMPT
from openai import OpenAI
from transformers import AutoTokenizer
import tiktoken


_FATAL_HTTP_STATUSES = {401, 402, 403}


def _abort_on_fatal_api(service: str, status: int, body: str, ctx: str = ""):
    sys.stderr.write(
        f"\n[FATAL][{service}] HTTP {status} indicates auth/quota failure — aborting.\n"
        f"  ctx: {ctx}\n"
        f"  body: {body[:500]}\n"
        f"  Fix the credential/balance, then restart. The script supports resume.\n"
    )
    sys.stderr.flush()
    os._exit(1)


def _abort_on_openai_fatal(service: str, exc: Exception, ctx: str = ""):
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
    if status in _FATAL_HTTP_STATUSES:
        _abort_on_fatal_api(service, status, str(exc), ctx)
    msg = str(exc).lower()
    if any(k in msg for k in ("insufficient_quota", "invalid_api_key", "incorrect api key",
                              "authentication", "unauthorized", "permission denied",
                              "insufficient balance", "insufficient credit")):
        _abort_on_fatal_api(service, status or -1, str(exc), ctx)

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))
JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")
print(f"[visit] JINA_API_KEYS loaded: {'SET (len=' + str(len(JINA_API_KEYS)) + ')' if JINA_API_KEYS else 'EMPTY'}")

# Reuse the HTTPS connection to avoid exhausting file descriptors under load.
_jina_session = requests.Session()
_jina_session.headers.update({"Authorization": f"Bearer {JINA_API_KEYS}"})

# Reuse the OpenAI client connection to reduce file-descriptor overhead.
_summary_client = OpenAI(
    api_key=os.environ.get("API_KEY", ""),
    base_url=os.environ.get("API_BASE", ""),
    timeout=120.0,
)

@staticmethod
def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""

def _is_blocked_url(url: str) -> bool:
    """
    Check if the URL should be blocked to prevent data leakage.
    Based on MiroThinker's implementation.

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

@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {
                    "type": "string"
                    },
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
            }
        },
        "required": ["url", "goal"]
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        # URL block check (adapted from MiroThinker).
        block_result = self._check_url_blocking(url, goal)
        if block_result:
            return block_result

        start_time = time.time()
        
        # Create log folder if it doesn't exist
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        if isinstance(url, str):
            response = self.readpage_jina(url, goal)
        else:
            response = []
            assert isinstance(url, List)
            start_time = time.time()
            for u in url: 
                if time.time() - start_time > 900:
                    cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    try:
                        cur_response = self.readpage_jina(u, goal)
                    except Exception as e:
                        cur_response = f"Error fetching {u}: {str(e)}"
                response.append(cur_response)
            response = "\n=======\n".join(response)
        
        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()

    def _is_blocked_url(self, url: str) -> bool:
        """
        Check whether a URL is in the block list (extends MiroThinker's logic).

        Args:
            url: the URL to check

        Returns:
            bool: True if the URL is blocked, otherwise False
        """
        blocked_patterns = [
            # HuggingFace datasets and spaces (as in MiroThinker)
            "huggingface.co/datasets",
            "huggingface.co/spaces",
            # Other sites that may leak answers
            "grok.com/share",
            "modelscope.cn/datasets",
        ]

        # Return True if the URL matches any blocked pattern.
        for pattern in blocked_patterns:
            if pattern in url:
                return True
        return False

    def _check_url_blocking(self, url: Union[str, List[str]], goal: str) -> Optional[str]:
        """
        Run the URL block check (based on MiroThinker's logic).

        Args:
            url: a URL or list of URLs to check
            goal: the visit goal (used in the error message)

        Returns:
            Optional[str]: None if access is allowed, otherwise an error message
        """
        if isinstance(url, str):
            # single-URL check
            if self._is_blocked_url(url):
                return json.dumps({
                    "success": False,
                    "url": url,
                    "extracted_info": "",
                    "error": "You are trying to scrape a Hugging Face dataset for answers, please do not use the scrape tool for this purpose.",
                    "scrape_stats": {},
                    "tokens_used": 0,
                }, ensure_ascii=False)
        else:
            # multi-URL check
            blocked_urls = []

            for u in url:
                if self._is_blocked_url(u):
                    blocked_urls.append(u)

            if blocked_urls:
                # Return an error if any URL is blocked.
                return json.dumps({
                    "success": False,
                    "url": blocked_urls,
                    "extracted_info": "",
                    "error": f"You are trying to scrape Hugging Face datasets for answers, please do not use the scrape tool for this purpose. Blocked URLs: {', '.join(blocked_urls)}",
                    "scrape_stats": {},
                    "tokens_used": 0,
                }, ensure_ascii=False)
        
        return None  # access allowed

    def call_server(self, msgs, max_retries=5):
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
        for attempt in range(max_retries):
            try:
                chat_response = _summary_client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=0.7
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string
                        left = content.find('{')
                        right = content.rfind('}')
                        if left != -1 and right != -1 and left <= right:
                            content = content[left:right+1]
                    return content
                else:
                    print(f"[visit] call_server attempt {attempt+1}/{max_retries}: empty content")
            except Exception as e:
                _abort_on_openai_fatal("DeepSeek/Summary", e, ctx=f"model={model_name}")
                print(f"[visit] call_server attempt {attempt+1}/{max_retries}: {type(e).__name__}: {str(e)[:200]}")
                if attempt == (max_retries - 1):
                    return ""
                sleep_time = min(2 ** attempt + random.uniform(0, 1), 30)
                time.sleep(sleep_time)
                continue

    def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.
        """
        max_retries = 3
        timeout = 15

        for attempt in range(max_retries):
            try:
                response = _jina_session.get(
                    f"https://r.jina.ai/{url}",
                    timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                elif response.status_code in _FATAL_HTTP_STATUSES:
                    body = response.text or ""
                    body_l = body.lower()
                    # Account-level auth/quota failures MUST abort.
                    account_fatal = (
                        response.status_code in (401, 402)
                        or "authenticationfailederror" in body_l
                        or "invalid api key" in body_l
                        or "insufficient" in body_l
                        or "quota" in body_l
                        or "billing" in body_l
                        or "payment" in body_l
                    )
                    # Per-URL transient blocks (Jina cools down specific URLs) are NOT fatal.
                    url_level_block = (
                        "operationnotallowederror" in body_l
                        or "consecutive error detected on url" in body_l
                        or "no content available for url" in body_l
                    )
                    if account_fatal and not url_level_block:
                        _abort_on_fatal_api("Jina", response.status_code, body, ctx=f"url={url[:120]}")
                    # treat as a normal read failure; let outer retry/fallback handle it
                    print(f"[visit] jina {response.status_code} url-level block for {url[:80]}, skipping")
                    return "[visit] Failed to read page."
                elif response.status_code == 429:
                    # Jina rate limited, exponential backoff
                    wait_time = min(2 ** attempt * 3 + random.uniform(0, 2), 30)
                    print(f"[visit] jina rate limited (429), waiting {wait_time:.0f}s")
                    time.sleep(wait_time)
                else:
                    print(f"[visit] jina status {response.status_code}: {response.text[:200]}")
                    raise ValueError("jina readpage error")
            except Exception as e:
                print(f"[visit] jina_readpage attempt {attempt+1}/{max_retries}: {type(e).__name__}: {str(e)[:200]}")
                sleep_time = min(2 ** attempt + random.uniform(0, 1), 10)
                time.sleep(sleep_time)
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."

        return "[visit] Failed to read page."

    def html_readpage_jina(self, url: str) -> str:
        max_attempts = 3
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content
            else:
                if attempt < max_attempts - 1:
                    wait_time = min(2 ** attempt * 2 + random.uniform(0, 1), 15)
                    print(f"[visit] html_readpage_jina fail attempt {attempt+1}/{max_attempts}, retrying in {wait_time:.0f}s")
                    time.sleep(wait_time)
        print(f"[visit] html_readpage_jina ALL {max_attempts} attempts FAILED for {url[:80]}")
        return "[visit] Failed to read page."

    def readpage_jina(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        """
        summary_page_func = self.call_server
        max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 3))

        content = self.html_readpage_jina(url)

        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            parse_retry_times = 0
            raw = summary_page_func(messages, max_retries=max_retries)
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                status_msg = (
                    f"[visit] Summary url[{url}] " 
                    f"attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, "
                    f"truncating to {truncate_length} chars"
                ) if summary_retries > 0 else (
                    f"[visit] Summary url[{url}] failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )
                print(status_msg)
                content = content[:truncate_length]
                extraction_prompt = EXTRACTOR_PROMPT.format(
                    webpage_content=content,
                    goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = summary_page_func(messages, max_retries=max_retries)
                summary_retries -= 1

            parse_retry_times = 0
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = summary_page_func(messages, max_retries=max_retries)
                    parse_retry_times += 1
            
            if parse_retry_times >= 3:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                print("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"
            
            return useful_information
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            return useful_information