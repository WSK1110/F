import os
import re
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Generator, Tuple
from collections import OrderedDict
from pathlib import Path

# LLM and Embedding imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Performance monitoring
from dotenv import load_dotenv

# Enhanced modules
from config import CHROMA_DIR, CHROMA_COLLECTION, QUERY_CACHE_TTL, MAX_CACHE_SIZE

# Load API keys from ~/.hermes/.env
_hermes_env = os.path.expanduser("~/.hermes/.env")
if os.path.exists(_hermes_env):
    load_dotenv(_hermes_env)

# ── Chroma singleton ──────────────────────────────────────────────────────────
_chroma_client = None
_chroma_persist_dir = None


def _get_chroma_client(persist_directory: str):
    global _chroma_client, _chroma_persist_dir
    if _chroma_client is None or _chroma_persist_dir != persist_directory:
        import chromadb
        from chromadb.config import Settings
        _chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        _chroma_persist_dir = persist_directory
    return _chroma_client


# ── LangChain compatibility ───────────────────────────────────────────────────
try:
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_classic.prompts import PromptTemplate
    from langchain_classic.schema import Document
    from langchain_classic.retrievers import EnsembleRetriever
    _LC = 'classic'
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    from langchain.retrievers import EnsembleRetriever
    _LC = 'standard'

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Provider mappings ─────────────────────────────────────────────────────────

_PROVIDER_DEFAULT_MODELS = {
    'gemini': 'models/gemini-1.5-pro',
    'openai': 'gpt-4o',
    'minimax': 'MiniMax-M2.7',
    'anthropic': 'claude-sonnet-4-20250514',
    'deepseek': 'deepseek-chat',
    'zhipu': 'glm-4-plus',
    'aliyun': 'qwen-plus',
    'siliconflow': 'Pro/deepseek-ai/DeepSeek-V3',
    'groq': 'llama3-70b-8192',
    'together': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
    'baidu': 'ernie-4.0-8k-latest',
    'bytedance': 'doubao-pro-32k',
    'openai_compat': 'gpt-4o',
    'ollama': 'llama3',
}

_PROVIDER_BASE_URLS = {
    'gemini': '',
    'openai': 'https://api.openai.com/v1',
    'minimax': 'https://api.minimaxi.com/v1',
    'anthropic': '',
    'deepseek': 'https://api.deepseek.com/v1',
    'zhipu': 'https://open.bigmodel.cn/api/paas/v4',
    'aliyun': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'siliconflow': 'https://api.siliconflow.cn/v1',
    'groq': 'https://api.groq.com/openai/v1',
    'together': 'https://api.together.xyz/v1',
    'baidu': 'https://qianfan.baidubce.com/v2',
    'bytedance': 'https://ark.cn-beijing.volces.com/api/v3',
    'openai_compat': '',
    'ollama': 'http://localhost:11434/v1',
}

_PROVIDER_API_KEYS = {
    'gemini': ['GOOGLE_API_KEY', 'KIMI_API_KEY'],
    'openai': ['OPENAI_API_KEY'],
    'minimax': ['MINIMAX_CN_API_KEY'],
    'anthropic': ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY'],
    'deepseek': ['DEEPSEEK_API_KEY'],
    'zhipu': ['ZHIPU_API_KEY', 'GLM_API_KEY'],
    'baidu': ['BAIDU_API_KEY', 'ERNIE_API_KEY'],
    'aliyun': ['ALIYUN_API_KEY', 'DASHSCOPE_API_KEY'],
    'bytedance': ['BYTEDANCE_API_KEY', 'DOUBAO_API_KEY'],
    'siliconflow': ['SILICONFLOW_API_KEY'],
    'groq': ['GROQ_API_KEY'],
    'together': ['TOGETHER_API_KEY'],
    'ollama': [],
    'openai_compat': ['OPENAI_API_KEY', 'OPENAI_COMPAT_KEY'],
}


# ── Default Config ───────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'RAG': {
        'chunk_size': '850',
        'chunk_overlap': '300',
        'similarity_top_k': '8'
    },
    'LLM': {
        'provider': 'deepseek',
        'model': 'deepseek-chat',
    },
    'EMBEDDINGS': {
        'provider': 'deepseek',
        'model': 'deepseek-embedding',
    },
    'LIGHT_RAG': {
        'use_hybrid_retrieval': 'true',
        'sparse_weight': '0.50',
        'dense_weight': '0.80',
        'rerank_top_k': '12',
    }
}

# ── Finance Ratio Engine ─────────────────────────────────────────────────────
class FinanceRatioEngine:
    """
    Parses financial text for monetary values and ratios.
    Pure regex-based, no LLM.
    """
    def __init__(self):
        self._number_pattern = re.compile(
            r'[\$€£]?\s*([+-]?[\d,]+(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?',
            re.IGNORECASE
        )
        self._year_pattern = re.compile(r'\b(19|20)\d{2}\b')

    def extract_number(self, text: str) -> Optional[float]:
        """Extract the first monetary number from text, return as float (in billions)."""
        match = self._number_pattern.search(text)
        if not match:
            return None
        raw = match.group(1).replace(',', '')
        magnitude = match.group(2).lower() if match.lastindex >= 2 else ''
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        if magnitude in ('billion', 'b'):
            value *= 1_000_000_000
        elif magnitude in ('million', 'm'):
            value *= 1_000_000
        elif magnitude in ('thousand', 'k'):
            value *= 1_000
        return value

    def extract_year_value_pairs(self, text: str, keywords: List[str]) -> Dict[str, Dict[int, float]]:
        """Find year-value pairs near keywords. Returns {keyword: {year: value}}."""
        results = {}
        for kw in keywords:
            pattern = re.compile(
                rf'{kw}[^\d]{{0,100}}({self._year_pattern.pattern})[^\d]{{0,20}}([\d,\.]+(?:\.\d+)?)\s*(?:billion|million|m|b|k)?',
                re.IGNORECASE
            )
            matches = pattern.findall(text)
            if matches:
                results[kw] = {int(yr): self._parse_value(val) for yr, val in matches}
        return results

    def _parse_value(self, val_str: str) -> float:
        val_str = val_str.replace(',', '')
        if val_str.replace('.', '').isdigit():
            return float(val_str)
        return 0.0

    def _safe_ratio(self, num: float, denom: float, decimals: int = 2) -> Optional[float]:
        if not denom or denom == 0:
            return None
        return round(num / denom, decimals)

    # ── Keyword-based extraction ───────────────────────────────────────────
    REVENUE_KEYWORDS = [
        r'(?:total\s+)?(?:net\s+)?revenue(?:s)?',
        r'(?:total\s+)?(?:net\s+)?sales',
        r'(?:total\s+)?gross\s+profit',
        r'(?:total\s+)?operating\s+income',
        r'(?:total\s+)?net\s+income',
        r'(?:total\s+)?cost\s+of\s+(?:sales|revenue|goods)',
        r'(?:total\s+)?research\s+(?:and|&)\s+development',
        r'(?:total\s+)?selling,?\s*(?:general|and|&)?\s*(?:administrative)?',
        r'(?:total\s+)?sg&?a',
        r'(?:total\s+)?current\s+assets',
        r'(?:total\s+)?current\s+liabilities',
        r'(?:total\s+)?assets',
        r'(?:total\s+)?liabilities',
        r'(?:total\s+)?stockholders?\s*(?:\'s|s\')?\s*equity',
        r'(?:total\s+)?(?:stockholders?\s*)?equity',
        r'(?:total\s+)?shareholders?\s*equity',
        r'(?:total\s+)?property,?\s*(?:and|&)?\s*equipment',
        r'(?:total\s+)?ppe',
        r'(?:total\s+)?depreciation',
        r'(?:total\s+)?amortization',
        r'(?:total\s+)?cash\s+(?:and\s+)?cash\s+equivalents',
        r'(?:total\s+)?accounts?\s+receivable',
        r'(?:total\s+)?inventor',
        r'(?:total\s+)?goodwill',
        r'(?:total\s+)?long.?term\s+debt',
        r'(?:total\s+)?short.?term\s+debt',
        r'(?:total\s+)?debt',
        r'(?:total\s+)?interest\s+expense',
        r'(?:total\s+)?income\s+tax',
        r'(?:total\s+)?diluted\s+eps',
        r'(?:total\s+)?basic\s+eps',
        r'(?:total\s+)?operating\s+cash\s+flow',
        r'(?:total\s+)?free\s+cash\s+flow',
        r'(?:total\s+)?investing\s+cash\s+flow',
        r'(?:total\s+)?financing\s+cash\s+flow',
        r'(?:total\s+)?dividends?\s+per\s+share',
        r'(?:total\s+)?weighted.?average\s+shares',
        r'(?:total\s+)?shares\s+outstanding',
        r'(?:total\s+)?employees?',
        r'(?:total\s+)?stock.?based\s+compensation',
        r'(?:total\s+)?restructuring',
        r'(?:total\s+)?impairment',
        r'(?:total\s+)?goodwill\s+impairment',
        r'(?:total\s+)?inventory',
        r'(?:total\s+)?marketable\s+securities',
        r'(?:total\s+)?prepaid',
        r'(?:total\s+)?accrued',
        r'(?:total\s+)?deferred\s+revenue',
        r'(?:total\s+)?lease',
        r'(?:total\s+)?commitments?\s+and\s+contingencies',
        r'(?:total\s+)?legal\s+proceedings',
        r'(?:total\s+)?risk\s+factors',
        r'(?:total\s+)?competition',
        r'(?:total\s+)?segment',
        r'(?:total\s+)?geographic',
        r'(?:total\s+)?international',
        r'(?:total\s+)?domestic',
    ]

    # ── Consolidated compute ───────────────────────────────────────────────
    def compute(self, texts: List[str], question: str) -> Dict[str, Any]:
        """Main entry: parse documents and extract financial line items."""
        combined = "\n".join(texts)[:100_000]

        extracted = {}
        # Extract line items: handle various 10-K formats
        # Format 1: "Label: Description $value" or "Label $value"
        # Format 2: "Label value1 value2 value3" (multi-year table rows)
        line_pattern = re.compile(
            r'(total\s+)?(net\s+)?(sales|revenue|income|profit|cost|margin|assets|liabilities|equity|cash|debt|eps|shares|employees?|depreciation|amortization|research|development|selling|general|administrative|interest|tax|dividend|stockholder|shareholder|investment|property|equipment|goodwill|inventory|receivable|payable|expense)'
            r'(?:\s*(?::|was|were|is|are|=)\s*|\s+)(?:\w+\s+)?'
            r'[\$€£]?\s*([+-]?[\d,]+(?:\.\d+)?)'
            r'(?:\s*(?:billion|million|thousand|B|M|K|m|b|k))?',
            re.IGNORECASE
        )

        for match in line_pattern.finditer(combined):
            prefix = (match.group(1) or '').strip()
            net_prefix = (match.group(2) or '').strip()
            label = match.group(3).lower()
            raw_val = match.group(4).replace(',', '')

            # Build a clean label
            parts = []
            if prefix:
                parts.append(prefix)
            if net_prefix:
                parts.append(net_prefix)
            parts.append(label)
            clean_label = '_'.join(parts)

            try:
                val = float(raw_val)
            except ValueError:
                continue

            # 10-K tables typically report in millions. Check if context says otherwise.
            before_ctx = combined[max(0, match.start()-30):match.start()]
            is_billion = bool(re.search(r'(?i)billion', before_ctx))
            is_million = bool(re.search(r'(?i)million|\(in\s*millions?\)', before_ctx))

            # Store value and unit tag for display purposes
            if is_billion:
                unit = 'B'
                val_display = val  # already billions
            elif is_million:
                unit = 'M'
            else:
                # Unknown unit — assume millions (typical for 10-K tables)
                unit = 'M'

            # Keep the first match (highest priority)
            if clean_label not in extracted:
                extracted[clean_label] = {"val": val, "unit": unit}
                if len(extracted) >= 20:
                    break

        # Compute ratios from available values
        ratios = self._compute_ratios(extracted)

        # Build summary with display formatting
        summary_parts = []
        for k, v in extracted.items():
            if isinstance(v, dict) and 'val' in v:
                val = v['val']
                unit = v.get('unit', '')
                summary_parts.append(f"{k}: {val}{unit}")
            else:
                summary_parts.append(f"{k}: {v}")
        if ratios:
            summary_parts.append("---")
            summary_parts.append("Computed Ratios:")
            summary_parts.extend(f"{k}: {v}" for k, v in ratios.items())
        summary = "; ".join(summary_parts)

        # Convert extracted to display format for the API response
        display_extracted = {}
        for k, v in extracted.items():
            if isinstance(v, dict) and 'val' in v:
                val = v['val']
                unit = v.get('unit', '')
                display_extracted[k] = f"{round(val, 1)}{unit}"
            else:
                display_extracted[k] = str(v)

        return {"extracted": display_extracted, "ratios": ratios, "summary": summary}

    def _compute_ratios(self, extracted: Dict[str, Any]) -> Dict[str, str]:
        def _get_val(key_patterns: list) -> Optional[float]:
            for pat in key_patterns:
                for k, v in extracted.items():
                    if re.search(pat, k, re.IGNORECASE):
                        if isinstance(v, dict) and 'val' in v:
                            return float(v['val'])
            return None

        revenue = _get_val([r'total.*net.*sales', r'total.*revenue', r'net_sales', r'revenue', r'sales'])
        net_income = _get_val([r'net_income', r'net.*income'])
        total_assets = _get_val([r'total_assets', r'total.*assets', r'assets'])
        current_assets = _get_val([r'current_assets', r'current.*assets'])
        current_liabilities = _get_val([r'current_liabilities', r'current.*liabilities'])
        gross_profit = _get_val([r'gross_profit', r'gross.*profit'])
        operating_income = _get_val([r'operating_income', r'operating.*income'])
        equity = _get_val([r'equity', r'stockholder.*equity', r'shareholder.*equity'])
        total_debt = _get_val('total_debt') or _get_val('debt') or _get_val('long_term_debt')
        cash = _get_val('cash_and_cash_equivalents') or _get_val('cash')

        ratios = {}
        if revenue and gross_profit:
            ratios['gross_margin_%'] = f"{round(gross_profit / revenue * 100, 1)}%"
        if revenue and operating_income:
            ratios['operating_margin_%'] = f"{round(operating_income / revenue * 100, 1)}%"
        if revenue and net_income:
            ratios['net_profit_margin_%'] = f"{round(net_income / revenue * 100, 1)}%"
        if total_assets and net_income:
            ratios['roa_%'] = f"{round(net_income / total_assets * 100, 1)}%"
        if equity and net_income:
            ratios['roe_%'] = f"{round(net_income / equity * 100, 1)}%"
        if current_assets and current_liabilities:
            ratios['current_ratio'] = round(current_assets / current_liabilities, 2)
        if total_debt and equity:
            ratios['debt_to_equity'] = round(total_debt / equity, 2)
        if cash and current_liabilities:
            ratios['cash_ratio'] = round(cash / current_liabilities, 2)

        return ratios


# ── Risk Detection Engine ─────────────────────────────────────────────────────
class RiskDetectionEngine:
    def __init__(self):
        # Key risk indicators (lowercase)
        self.risk_keywords = {
            'revenue_decline': ['revenue declined', 'revenue decrease', 'sales declined', 'lower revenue'],
            'profit_decline': ['net income declined', 'profit declined', 'net loss', 'operating loss'],
            'debt_risk': ['debt', 'leverage', 'covenant', 'default', 'credit rating'],
            'liquidity_risk': ['liquidity', 'cash flow', 'working capital', 'solvency'],
            'competition_risk': ['competition', 'competitive', 'market share', 'pricing pressure'],
            'regulatory_risk': ['regulatory', 'regulation', 'compliance', 'legal', 'litigation', 'investigation'],
            'macro_risk': ['inflation', 'interest rate', 'currency', 'supply chain', 'geopolitical'],
            'operational_risk': ['restructuring', 'layoff', 'closure', 'disruption', 'cyber'],
        }

        # Thresholds for flagging (values in millions)
        self.thresholds = {
            'debt_to_equity': 2.0,
            'current_ratio': 1.0,
            'net_profit_margin': 0.0,  # below 0% flags
            'revenue_decline_pct': -5.0,  # worse than -5% flags
        }

    def detect(self, texts: List[str], ratio_result: Dict[str, Any], question: str) -> Dict[str, Any]:
        combined = "\n".join(texts)[:100_000].lower()
        signals = {}

        # Keyword-based risk detection
        for risk_type, keywords in self.risk_keywords.items():
            for kw in keywords:
                if kw.lower() in combined:
                    if risk_type not in signals:
                        signals[risk_type] = {'found': [], 'severity': 'low'}
                    signals[risk_type]['found'].append(kw)
                    # Count occurrences as severity indicator
                    count = combined.count(kw.lower())
                    if count >= 5:
                        signals[risk_type]['severity'] = 'high'
                    elif count >= 2:
                        signals[risk_type]['severity'] = 'medium'

        # Ratio-based risk flags
        ratios = ratio_result.get('ratios', {})
        for ratio_key, threshold in self.thresholds.items():
            if ratio_key == 'debt_to_equity' and 'debt_to_equity' in ratios:
                val = ratios['debt_to_equity']
                if isinstance(val, (int, float)) and val > threshold:
                    signals['high_leverage'] = {'found': [f'debt_to_equity={val}'], 'severity': 'high' if val > 3 else 'medium'}
            elif ratio_key == 'current_ratio' and 'current_ratio' in ratios:
                val = ratios['current_ratio']
                if isinstance(val, (int, float)) and val < threshold:
                    signals['liquidity_risk'] = {'found': [f'current_ratio={val}'], 'severity': 'high' if val < 0.5 else 'medium'}
            elif ratio_key == 'net_profit_margin' and 'net_profit_margin_%' in ratios:
                val_str = ratios['net_profit_margin_%'].replace('%', '')
                try:
                    val = float(val_str)
                    if val < threshold:
                        signals['profitability_risk'] = {'found': [f'net_profit_margin={val}%'], 'severity': 'high'}
                except ValueError:
                    pass

        summary = "; ".join(f"{k}: {v['severity']}" for k, v in signals.items()) if signals else "No significant risk signals detected."
        return {"signals": signals, "summary": summary}


# ── Helper functions ─────────────────────────────────────────────────────────
def extract_company_name(file_path: str) -> str:
    """Extract company name from filename or first page content."""
    basename = os.path.splitext(os.path.basename(file_path))[0]
    known = {
        'apple': 'Apple Inc.',
        'msft': 'Microsoft Corporation',
        'amazon': 'Amazon.com Inc.',
        'googl': 'Alphabet Inc.',
        'meta': 'Meta Platforms Inc.',
        'nflx': 'Netflix Inc.',
        'tsla': 'Tesla Inc.',
        'nvda': 'NVIDIA Corporation',
        'aapl': 'Apple Inc.',
        'goog': 'Alphabet Inc.',
    }
    base_lower = basename.lower()
    for key, name in known.items():
        if key in base_lower:
            return name
    # Fallback: clean up filename
    cleaned = re.sub(r'[_-]', ' ', basename)
    cleaned = re.sub(r'\s+\d{2,4}.*$', '', cleaned)
    return cleaned.strip().title()


def is_comparison_question(question: str) -> bool:
    q = question.lower()
    comparative_words = ['compare', 'versus', 'vs', 'difference', 'better', 'worse', 'stronger', 'weaker', 'both', 'which']
    return any(w in q for w in comparative_words)


def retrieve_per_company(vector_store, question: str, document_splits: List[Document]) -> List[Document]:
    """Fallback: retrieve top-k per company for comparison questions."""
    seen = set()
    results = []
    for doc in document_splits:
        company = doc.metadata.get('company', 'Unknown')
        if company not in seen:
            seen.add(company)
            hits = vector_store.similarity_search(question, k=3, filter={"company": company})
            results.extend(hits)
    return results


# ── Query Cache ──────────────────────────────────────────────────────────────
_QUERY_CACHE_TTL = QUERY_CACHE_TTL
_MAX_CACHE_SIZE = MAX_CACHE_SIZE
_query_cache: OrderedDict = OrderedDict()


def _get_question_cache_key(question: str, config_hash: str) -> str:
    return hashlib.md5(f"{question}|{config_hash}".encode()).hexdigest()


# ── Performance Monitor ─────────────────────────────────────────────────────
_PERFORMANCE_METRICS = []
_MAX_METRICS = 1000


class performance_monitor:
    """Context manager to track operation timing."""
    def __init__(self, operation: str):
        self.operation = operation
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        _PERFORMANCE_METRICS.append({
            'operation': self.operation,
            'duration': round(elapsed, 3),
            'timestamp': time.time(),
        })
        if len(_PERFORMANCE_METRICS) > _MAX_METRICS:
            _PERFORMANCE_METRICS.pop(0)


# ── RAG Chatbot Core ─────────────────────────────────────────────────────────
class RAGChatbotCore:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.vector_store = None
        self.qa_chain = None
        self.document_splits: List[Document] = []
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer",
        )
        self.ratio_engine = FinanceRatioEngine()
        self.risk_engine = RiskDetectionEngine()

    # ── API Key resolution ─────────────────────────────────────────────────
    @staticmethod
    def _get_api_key(provider: str) -> str:
        """Resolve API key for a provider from env vars."""
        keys = _PROVIDER_API_KEYS.get(provider, [])
        for k in keys:
            val = os.getenv(k)
            if val:
                return val
        return ""

    # ── LLM ────────────────────────────────────────────────────────────────
    def get_llm(self, provider: str = None, model: str = None, base_url: str = None):
        provider = provider or self.config['LLM']['provider']
        if model is None:
            model = _PROVIDER_DEFAULT_MODELS.get(provider, self.config['LLM'].get('model', ''))
        api_key = self._get_api_key(provider)

        if not api_key and provider != 'ollama':
            logger.error(f"API key not found for LLM provider: {provider}")
            return None

        return self._build_llm(provider, model, api_key, base_url)

    def _build_llm(self, provider: str, model: str, api_key: str, base_url: str = None):
        """Build a ChatLLM instance for the given provider."""
        try:
            # ── Google Gemini ──────────────────────────────────────────────
            if provider == 'gemini':
                return ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=api_key,
                    temperature=0.3,
                    max_tokens=2048,
                )

            # ── OpenAI-compatible providers ────────────────────────────────
            llm_base_url = base_url or _PROVIDER_BASE_URLS.get(provider, '')

            # ── Anthropic (via langchain_anthropic SDK) ────────────────────
            if provider == 'anthropic':
                try:
                    from langchain_anthropic import ChatAnthropic
                    return ChatAnthropic(
                        model=model,
                        anthropic_api_key=api_key,
                        temperature=0.3,
                        max_tokens=2048,
                    )
                except ImportError:
                    logger.warning("langchain_anthropic not installed, falling back to ChatOpenAI for Anthropic")
                    return ChatOpenAI(
                        model=model,
                        openai_api_key=api_key,
                        openai_api_base="https://api.anthropic.com/v1",
                        temperature=0.3,
                        max_tokens=2048,
                    )

            # ── Ollama ─────────────────────────────────────────────────────
            if provider == 'ollama':
                return ChatOpenAI(
                    model=model,
                    openai_api_key="ollama",
                    openai_api_base=llm_base_url,
                    temperature=0.3,
                    max_tokens=2048,
                )

            # ── Default: ChatOpenAI for all OpenAI-compatible endpoints ────
            return ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                openai_api_base=llm_base_url,
                temperature=0.3,
                max_tokens=2048,
            )

        except Exception as e:
            logger.error(f"Failed to build LLM for {provider}: {e}")
            raise

    # ── Embeddings ───────────────────────────────────────────────────────────
    def get_embeddings(self, provider: str = None, model: str = None):
        provider = provider or self.config['EMBEDDINGS']['provider']
        model = model or self.config['EMBEDDINGS']['model']

        if provider == 'minimax_local':
            api_key = None
        elif provider == 'ollama':
            api_key = None
        elif provider == 'gemini':
            api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('KIMI_API_KEY')
        elif provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
        elif provider == 'minimax':
            api_key = os.getenv('MINIMAX_CN_API_KEY')
        elif provider == 'deepseek':
            api_key = os.getenv('DEEPSEEK_API_KEY')
        else:
            api_key = ""

        if provider not in ('minimax_local', 'ollama') and not api_key:
            logger.error(f"API key not found for embeddings provider: {provider}")
            return None

        try:
            return self._build_embeddings(provider, model, api_key)
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            return None

    def _build_embeddings(self, provider: str, model: str, api_key: str):
        if provider == 'gemini':
            return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        elif provider == 'openai':
            return OpenAIEmbeddings(model=model, openai_api_key=api_key)
        elif provider == 'minimax':
            return OpenAIEmbeddings(
                model="embeddings",
                openai_api_key=api_key,
                openai_api_base="https://api.minimaxi.com/v1",
            )
        elif provider == 'deepseek':
            return OpenAIEmbeddings(
                model=model,
                openai_api_key=api_key,
                openai_api_base="https://api.deepseek.com/v1",
            )
        elif provider == 'minimax_local':
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model)
        elif provider == 'ollama':
            import requests as _req
            # Bypass any proxy for localhost connections
            _no_proxy = {"http": "", "https": ""}
            class _OllamaEmbeddings:
                def __init__(self, model_name):
                    self.model_name = model_name
                    self.base_url = "http://localhost:11434"
                def embed_documents(self, texts):
                    return [self._embed(t) for t in texts]
                def embed_query(self, text):
                    return self._embed(text)
                def _embed(self, text):
                    r = _req.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model_name, "prompt": text},
                        timeout=60,
                        proxies=_no_proxy,
                    )
                    return r.json()["embedding"]
            return _OllamaEmbeddings(model)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    # ── PDF Upload ───────────────────────────────────────────────────────────
    def load_pdf(self, file_path: str, use_enhanced_parser: bool = True) -> List[Document]:
        """Load a single PDF file and tag with company name.

        When use_enhanced_parser=True and pdfplumber is available, uses
        table-aware extraction for better financial document parsing.
        Also handles .htm/.html files from SEC EDGAR.
        """
        ext = os.path.splitext(file_path)[1].lower()

        # Handle HTML files (SEC EDGAR filings)
        if ext in ('.htm', '.html'):
            return self._load_html(file_path)

        with performance_monitor("Document Loading"):
            try:
                if use_enhanced_parser:
                    from enhanced_parser import pdfplumber_load, HAS_PDFPLUMBER
                    if HAS_PDFPLUMBER:
                        docs = pdfplumber_load(file_path)
                        if docs:
                            return docs
                # Fallback to basic PyPDFLoader
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                company = extract_company_name(file_path)
                for d in docs:
                    d.metadata['company'] = company
                return docs
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                return []

    def _load_html(self, file_path: str) -> List[Document]:
        """Load an HTML/HTM SEC filing and extract text content."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Fallback: basic HTML tag stripping
            import re as _re
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            text = _re.sub(r'<[^>]+>', ' ', content)
            text = _re.sub(r'\s+', ' ', text).strip()
        else:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            soup = BeautifulSoup(content, 'html.parser')
            # Remove scripts and styles
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            text = soup.get_text(separator='\n')
            text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())

        company = extract_company_name(file_path)
        return [Document(
            page_content=text,
            metadata={
                "source": file_path,
                "company": company,
                "type": "html",
            }
        )]

    def load_pdfs(self, file_paths: List[str]) -> List[Document]:
        """Load multiple PDFs and flatten into a single document list."""
        all_docs = []
        for fp in file_paths:
            docs = self.load_pdf(fp)
            all_docs.extend(docs)
        return all_docs

    # ── Text Splitter ──────────────────────────────────────────────────────
    def _get_text_splitter(self):
        chunk_size = int(self.config['RAG'].get('chunk_size', '850'))
        chunk_overlap = int(self.config['RAG'].get('chunk_overlap', '300'))
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Vector Store ───────────────────────────────────────────────────────
    def create_vector_store(self, documents: List[Document]):
        with performance_monitor("Vector Store Creation"):
            try:
                splitter = self._get_text_splitter()
                new_splits = splitter.split_documents(documents)

                if self.vector_store:
                    # Add to existing store (preserves the original embedding function)
                    self.vector_store.add_documents(new_splits)
                    self.document_splits.extend(new_splits)
                    logger.info(f"Added {len(new_splits)} splits to existing vector store (total: {len(self.document_splits)})")
                    return self.vector_store

                # First-time creation
                self.document_splits = new_splits
                logger.info(f"Created {len(self.document_splits)} document splits")

                embeddings = self.get_embeddings()
                if not embeddings:
                    raise ValueError("Failed to initialize embeddings")

                self.vector_store = Chroma.from_documents(
                    documents=self.document_splits,
                    embedding=embeddings,
                    collection_name=CHROMA_COLLECTION,
                    persist_directory=str(CHROMA_DIR),
                )
                return self.vector_store
            except Exception as e:
                logger.error(f"Vector store creation failed: {e}")
                return None

    def load_persisted_vector_store(self, persist_directory: str = None):
        """Load existing Chroma vector store from disk (no re-chunking).

        Note: document_splits remains empty after this call, so hybrid retrieval
        (BM25 + dense) is unavailable — falls back to dense-only.
        """
        if persist_directory is None:
            persist_directory = str(CHROMA_DIR)
        embeddings = self.get_embeddings()
        if not embeddings:
            raise ValueError("Failed to initialize embeddings")

        chroma_client = _get_chroma_client(persist_directory)
        self.vector_store = Chroma(
            client=chroma_client,
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings
        )
        count = self.vector_store._collection.count()
        logger.info(f"Loaded persisted vector store with {count} chunks")
        return self.vector_store

    # ── Hybrid Retriever ─────────────────────────────────────────────────────
    def _create_hybrid_retriever(self):
        """Create hybrid dense+sparse retriever."""
        with performance_monitor("Hybrid Retriever Creation"):
            dense_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": int(self.config['LIGHT_RAG'].get('rerank_top_k', '12'))}
            )
            sparse_retriever = BM25Retriever.from_documents(self.document_splits)
            sparse_retriever.k = int(self.config['LIGHT_RAG'].get('rerank_top_k', '12'))

            ensemble = EnsembleRetriever(
                retrievers=[sparse_retriever, dense_retriever],
                weights=[
                    float(self.config['LIGHT_RAG'].get('dense_weight', '0.80')),
                    float(self.config['LIGHT_RAG'].get('sparse_weight', '0.50'))
                ]
            )
            return ensemble

    # ── QA Chain ───────────────────────────────────────────────────────────
    PROMPT_TEMPLATE = """You are an AI Financial Analyst focused on interpreting and comparing 10-K annual reports.

Your Responsibilities
    • Extract and report only the facts contained in the provided filings.
    • Compare financial metrics and business strategies across companies.
    • Identify material risks, opportunities, and key business drivers.
    • Cite precise numbers and dates from the documents.
    • If a topic is missing, state explicitly: "Not mentioned in the provided documents."

Guiding Principles
    • Context-Only: Do not infer any information beyond what's in the supplied documents.
    • No Guesswork: If unsure, say so rather than speculate.
    • Data-Driven Comparisons: Anchor all comparisons to exact figures.
    • Investor Focus: Emphasize material information for investment decisions.
    • Objective & Analytical: Maintain a neutral, evidence-based tone.

Context: {context}
Question: {question}
Chat History: {chat_history}

Answer:
"""

    def create_qa_chain(self, llm_provider: str = None, llm_model: str = None, system_prompt: str = None):
        with performance_monitor("QA Chain Creation"):
            llm = self.get_llm(llm_provider, llm_model)
            if not llm:
                logger.error("Failed to initialize LLM for QA chain")
                return None

            prompt = PromptTemplate(
                template=system_prompt or self.PROMPT_TEMPLATE,
                input_variables=["context", "question", "chat_history"],
            )

            retriever = None
            if self.document_splits and self.vector_store:
                use_hybrid = self.config['LIGHT_RAG'].get('use_hybrid_retrieval', 'true').lower() == 'true'
                if use_hybrid:
                    retriever = self._create_hybrid_retriever()
                else:
                    retriever = self.vector_store.as_retriever(
                        search_kwargs={"k": int(self.config['RAG'].get('similarity_top_k', '8'))}
                    )
            elif self.vector_store:
                logger.warning("Hybrid retrieval requested but document_splits unavailable (loaded from persistence). Falling back to dense-only.")
                retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": int(self.config['RAG'].get('similarity_top_k', '8'))}
                )
            else:
                logger.error("No vector store available for QA chain")
                return None

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True,
                verbose=False,
                output_key="answer"
            )
            return self.qa_chain

    # ── LLM with retry ────────────────────────────────────────────────────────
    def _llm_with_retry(self, question: str, chat_history: list, max_retries: int = 3) -> Dict[str, Any]:
        """Call LLM with exponential backoff retry and graceful degradation."""
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.qa_chain({"question": question, "chat_history": chat_history})
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if any(keyword in err_str for keyword in
                       ['rate', 'limit', 'timeout', 'connection', '429', '500', '502', '503', '504', 'network']):
                    wait = 2 ** attempt
                    logger.warning(f"Transient error on attempt {attempt + 1}, retrying in {wait}s: {e}")
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(f"Non-transient error, skipping retry: {e}")
                    break

        # Graceful degradation: return top retrieved documents as answer
        logger.warning(f"All LLM retry attempts failed, using fallback retrieval: {last_error}")
        try:
            docs = self.vector_store.similarity_search(question, k=3)
            fallback_answer = (
                "I'm sorry — the AI model encountered an error and could not generate a response. "
                "Here are the most relevant excerpts from the documents instead:\n\n"
                + "\n\n---\n\n".join(f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content[:500]}" for d in docs)
            )
            return {
                "answer": fallback_answer,
                "source_documents": docs,
                "_fallback": True,
                "_error": str(last_error),
            }
        except Exception as fallback_err:
            return {
                "answer": "",
                "source_documents": [],
                "_fallback": True,
                "_error": f"LLM error: {last_error}; Fallback also failed: {fallback_err}",
            }

    # ── Ask Question ─────────────────────────────────────────────────────────
    def _compute_config_hash(self) -> str:
        config_str = json.dumps({
            'chunk_size': self.config['RAG'].get('chunk_size', '850'),
            'chunk_overlap': self.config['RAG'].get('chunk_overlap', '300'),
            'similarity_top_k': self.config['RAG'].get('similarity_top_k', '8'),
            'provider': self.config['LLM'].get('provider', 'gemini'),
            'model': self.config['LLM'].get('model', 'models/gemini-1.5-pro'),
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def ask_question(
        self, question: str, use_hyde: bool = False, use_rerank: bool = False,
        return_all_details: bool = False,
    ) -> Dict[str, Any]:
        """Ask a question with caching, retry, performance monitoring, and optional enhancements.

        Args:
            question: The user's question.
            use_hyde: Use Hypothetical Document Embedding for better retrieval.
            use_rerank: Use cross-encoder reranking of retrieved documents.
            return_all_details: Include faithfulness verification in response.

        Returns:
            dict with answer, sources, performance, financial_ratios, risk_signals,
            and optionally faithfulness results.
        """
        start_time = time.time()

        if not self.qa_chain:
            return {"error": "QA chain not initialized"}

        # Check cache
        config_hash = self._compute_config_hash()
        cache_key = _get_question_cache_key(question, config_hash)
        cached_entry = _query_cache.get(cache_key)
        if cached_entry is not None:
            cached_result, cached_at = cached_entry
            if time.time() - cached_at < _QUERY_CACHE_TTL:
                cached_result["from_cache"] = True
                return cached_result
            else:
                del _query_cache[cache_key]

        try:
            with performance_monitor("Question Processing"):
                # ── Stage 1: Retrieve documents ─────────────────────────────────
                search_query = question

                # Optional HyDE: generate hypothetical document for retrieval
                if use_hyde:
                    try:
                        from query_enhancer import generate_hypothetical_answer
                        llm = self.get_llm()
                        hyde_answer = generate_hypothetical_answer(question, llm)
                        if hyde_answer:
                            search_query = hyde_answer
                            logger.debug("Using HyDE query for retrieval")
                    except ImportError:
                        logger.debug("query_enhancer not available, using raw query")

                docs = []
                if self.vector_store:
                    docs = self.vector_store.similarity_search(search_query, k=12)

                # Optional reranking
                if use_rerank and docs:
                    try:
                        from reranker import rerank_documents
                        reranked = rerank_documents(question, docs, top_k=8)
                        docs = [d for d, _ in reranked]
                        logger.debug(f"Reranked {len(reranked)} documents")
                    except ImportError:
                        logger.debug("reranker not available, skipping")

                texts = [doc.page_content for doc in docs]

                # ── Stage 2: Compute financial ratios ────────────────────────────
                ratio_result = {}
                if texts:
                    try:
                        ratio_result = self.ratio_engine.compute(texts, question)
                    except Exception as e:
                        logger.warning(f"Ratio computation failed: {e}")
                ratio_summary = ratio_result.get('summary', '')

                # ── Stage 3: Detect risks ─────────────────────────────────────
                risk_result = {}
                if texts:
                    try:
                        risk_result = self.risk_engine.detect(texts, ratio_result, question)
                    except Exception as e:
                        logger.warning(f"Risk detection failed: {e}")
                risk_summary = risk_result.get('summary', '')

                # ── Stage 4: Build enriched prompt ──────────────────────────────
                if ratio_summary or risk_summary:
                    enriched_question = (
                        f"{question}\n\n"
                        f"--- FINANCIAL RATIOS ---\n{ratio_summary}\n\n"
                        f"--- RISK SIGNALS ---\n{risk_summary}\n\n"
                        f"Use the above financial ratios and risk signals to answer the question. "
                        f"Cite specific numbers from the ratios where applicable."
                    )
                else:
                    enriched_question = question

                if is_comparison_question(question) and self.document_splits and self.vector_store:
                    company_docs = retrieve_per_company(self.vector_store, question, self.document_splits)
                    if company_docs:
                        docs = company_docs

                # Load actual chat history from ConversationBufferMemory
                chat_history = self.memory.chat_memory.messages if hasattr(self, 'memory') and self.memory else []

                result = self._llm_with_retry(enriched_question, chat_history)

                sources = []
                if 'source_documents' in result:
                    sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]

                response_time = time.time() - start_time

                result_dict = {
                    "answer": result.get('answer', ''),
                    "sources": sources,
                    "performance": {
                        "response_time": response_time,
                        "source_count": len(sources),
                        "answer_length": len(result.get('answer', '')),
                        "fallback_used": result.get('_fallback', False),
                    },
                    **({"error": result.get('_error', '')} if result.get('_fallback') else {}),
                    "financial_ratios": ratio_result,
                    "risk_signals": risk_result,
                }

                # ── Stage 5: Faithfulness verification ────────────────────────
                if return_all_details and not result.get('_fallback'):
                    try:
                        from faithfulness import check_faithfulness
                        source_docs = result.get('source_documents', [])
                        if source_docs:
                            faith_result = check_faithfulness(
                                answer=result.get('answer', ''),
                                source_documents=source_docs,
                                use_nli=False,
                            )
                            result_dict["faithfulness"] = {
                                "score": faith_result.faithfulness_score,
                                "total_claims": faith_result.total_claims,
                                "supported": faith_result.supported,
                                "contradicted": faith_result.contradicted,
                                "not_verifiable": faith_result.not_verifiable,
                                "summary": faith_result.summary,
                            }
                    except ImportError:
                        logger.debug("faithfulness module not available, skipping")

                if not result.get('_fallback'):
                    # Enforce max cache size - evict oldest entry if at capacity
                    if len(_query_cache) >= _MAX_CACHE_SIZE:
                        _query_cache.pop(next(iter(_query_cache)))
                    _query_cache[cache_key] = (result_dict, time.time())

                return result_dict

        except Exception as e:
            logger.error(f"Unexpected error in ask_question: {e}")
            return {"error": str(e)}

    # ── Text-to-SQL ───────────────────────────────────────────────────────────
    def ask_table(self, question: str) -> Dict[str, Any]:
        """Answer a financial question using Text-to-SQL against stored tables.

        Uses the configured LLM to convert natural language → SQL, executes
        against the SQLite table store, and returns structured results.
        """
        llm = self.get_llm()
        if not llm:
            return {"error": "LLM not available", "answer": ""}

        try:
            from text_to_sql import ask_table as _ask_table, is_table_question
            if not is_table_question(question):
                return {"answer": "", "auto_routed": False}
            result = _ask_table(question, llm)
            result["auto_routed"] = True
            return result
        except ImportError as e:
            return {"error": f"Text-to-SQL module not available: {e}", "answer": ""}
        except Exception as e:
            logger.error(f"ask_table error: {e}")
            return {"error": str(e), "answer": ""}

    # ── Streaming ──────────────────────────────────────────────────────────
    def _stream_openai_compat(self, messages: list, llm_provider: str = None, llm_model: str = None) -> Generator[str, None, None]:
        """Stream tokens from any OpenAI-compatible LLM. Yields SSE-formatted strings."""
        import requests as _req

        provider = llm_provider or self.config['LLM']['provider']
        model = llm_model or self.config['LLM'].get('model', 'deepseek-chat')
        temperature = float(self.config['LLM'].get('temperature', 0.3))
        max_tokens = int(self.config['LLM'].get('max_tokens', 2048))
        api_key = self._get_api_key(provider)
        base_url = _PROVIDER_BASE_URLS.get(provider, 'https://api.deepseek.com/v1')

        chat_messages = []
        for msg in messages:
            role = msg.type if hasattr(msg, 'type') else 'user'
            if role == 'human':
                role = 'user'
            elif role == 'ai':
                role = 'assistant'
            elif role == 'system':
                role = 'system'
            chat_messages.append({
                "role": role,
                "content": msg.content if hasattr(msg, 'content') else str(msg)
            })

        payload = {
            "model": model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        try:
            with _req.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=180,
                stream=True
            ) as resp:
                if resp.status_code != 200:
                    error_body = resp.text[:200]
                    yield f"data: [ERROR] API error {resp.status_code}: {error_body}\n\n"
                    return

                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode('utf-8').strip()
                    if not line.startswith('data: '):
                        continue
                    data_str = line[6:]  # strip "data: "
                    if data_str == '[DONE]':
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        data = json.loads(data_str)
                        choices = data.get('choices', [])
                        if not choices:
                            continue
                        delta = choices[0].get('delta', {})
                        token = delta.get('content', '')
                        if token:
                            escaped = token.replace('\n', '\\n').replace('\r', '\\r')
                            yield f"data: {escaped}\n\n"
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    def _stream_ollama(self, messages: list, llm_provider: str = None, llm_model: str = None) -> Generator[str, None, None]:
        """Stream tokens from Ollama LLM. Yields SSE-formatted strings."""
        import requests as _req

        provider = llm_provider or self.config['LLM']['provider']
        model = llm_model or self.config['LLM'].get('model', 'llama3')
        temperature = float(self.config['LLM'].get('temperature', 0.3))
        max_tokens = int(self.config['LLM'].get('max_tokens', 2048))

        ollama_messages = []
        for msg in messages:
            role = msg.type if hasattr(msg, 'type') else 'user'
            if role == 'human':
                role = 'user'
            elif role == 'ai':
                role = 'assistant'
            ollama_messages.append({
                "role": role,
                "content": msg.content if hasattr(msg, 'content') else str(msg)
            })

        payload = {
            "model": model,
            "messages": ollama_messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": True
        }

        try:
            with _req.post(
                f"http://localhost:11434/api/chat",
                json=payload,
                timeout=180,
                stream=True
            ) as resp:
                if resp.status_code != 200:
                    yield f"data: [ERROR] Ollama API error: {resp.status_code}\n\n"
                    return

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode('utf-8'))
                        token = data.get('message', {}).get('content', '')
                        if token:
                            escaped = token.replace('\n', '\\n').replace('\r', '\\r')
                            yield f"data: {escaped}\n\n"
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    # ── Ask Question (Streaming) ──────────────────────────────────────────────

    def ask_question_stream(self, question: str, llm_provider: str = None, llm_model: str = None, system_prompt: str = None) -> Generator[str, None, None]:
        """
        Streaming version of ask_question.
        Yields SSE-formatted strings: sources, ratios, risks first, then tokens.
        """
        import requests as _req

        # Stage 1: Retrieve documents
        docs = []
        if self.vector_store:
            docs = self.vector_store.similarity_search(question, k=8)
        texts = [doc.page_content for doc in docs]

        # Send sources upfront
        sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
        sources_json = json.dumps(sources)
        yield f"event: sources\ndata: {sources_json}\n\n"

        # Stage 2: Financial ratios
        ratio_result = {}
        if texts:
            try:
                ratio_result = self.ratio_engine.compute(texts, question)
            except Exception as e:
                logger.warning(f"Streaming ratio computation failed: {e}")
        yield f"event: ratios\ndata: {json.dumps(ratio_result)}\n\n"

        # Stage 3: Risk signals
        risk_result = {}
        if texts:
            try:
                risk_result = self.risk_engine.detect(texts, ratio_result, question)
            except Exception as e:
                logger.warning(f"Streaming risk detection failed: {e}")
        yield f"event: risks\ndata: {json.dumps(risk_result)}\n\n"

        # Stage 4: Build enriched question
        ratio_summary = ratio_result.get('summary', '')
        risk_summary = risk_result.get('summary', '')
        if ratio_summary or risk_summary:
            enriched_question = (
                f"{question}\n\n"
                f"--- FINANCIAL RATIOS ---\n{ratio_summary}\n\n"
                f"--- RISK SIGNALS ---\n{risk_summary}\n\n"
                f"Use the above financial ratios and risk signals to answer. "
                f"Cite specific numbers from the ratios where applicable."
            )
        else:
            enriched_question = question

        # Build messages for streaming LLM call
        default_system_prompt = """You are an AI Financial Analyst focused on interpreting and comparing 10-K annual reports.

Your Responsibilities
    • Extract and report only the facts contained in the provided filings.
    • Compare financial metrics and business strategies across companies.
    • Identify material risks, opportunities, and key business drivers.
    • Cite precise numbers and dates from the documents.
    • If a topic is missing, state explicitly: "Not mentioned in the provided documents."

Guiding Principles
    • Context-Only: Do not infer any information beyond what's in the supplied documents.
    • No Guesswork: If unsure, say so rather than speculate.
    • Data-Driven Comparisons: Anchor all comparisons to exact figures.
    • Investor Focus: Emphasize material information for investment decisions.
    • Objective & Analytical: Maintain a neutral, evidence-based tone.

Context: {context}
Question: {question}
Chat History: {chat_history}

Answer:
"""
        if _LC == 'classic':
            from langchain_classic.schema import HumanMessage, SystemMessage, AIMessage
        else:
            from langchain.schema import HumanMessage, SystemMessage, AIMessage
        prompt_text = system_prompt or default_system_prompt
        context = "\n\n".join(texts[:5])  # limit context to top 5 docs
        prompt_filled = prompt_text.replace("{context}", context).replace("{question}", enriched_question).replace("{chat_history}", "")

        messages = [SystemMessage(content=prompt_filled), HumanMessage(content=enriched_question)]

        # Route to the appropriate streaming method
        provider = llm_provider or self.config['LLM']['provider']
        if provider == 'ollama':
            for chunk in self._stream_ollama(messages, llm_provider, llm_model):
                yield chunk
        else:
            # All other providers (openai, deepseek, minimax, etc.) use OpenAI-compatible streaming
            for chunk in self._stream_openai_compat(messages, llm_provider, llm_model):
                yield chunk

    # ── Performance Metrics ─────────────────────────────────────────────────
    def get_performance_summary(self) -> Dict[str, Any]:
        if not _PERFORMANCE_METRICS:
            return {"message": "No metrics available yet"}
        total = len(_PERFORMANCE_METRICS)
        avg_time = sum(m['duration'] for m in _PERFORMANCE_METRICS) / total
        return {
            "total_operations": total,
            "average_duration": round(avg_time, 3),
            "max_duration": max(m['duration'] for m in _PERFORMANCE_METRICS),
            "min_duration": min(m['duration'] for m in _PERFORMANCE_METRICS),
        }

    def get_all_metrics(self) -> List[Dict]:
        return list(_PERFORMANCE_METRICS)
