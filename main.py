"""
🚀 终极智能RSS聚合器 v3.0 (GitHub Actions 版)
============================
核心升级：
1. 语义理解：关键词组合 + 上下文分析
2. 智能去重：标题相似度 + 内容指纹
3. 时效性衰减：24h内最新，超过48h降权
4. 质量信号聚合：多维弱信号 → 强判断
5. 自适应阈值：根据当日质量动态调整
"""

import requests
import datetime
import PyRSS2Gen
from bs4 import BeautifulSoup
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from difflib import SequenceMatcher
from collections import defaultdict
import hashlib
import json
import email.utils

# ================= 🧠 核心知识库 =================

# 分级权威库（影响力递减）
TIER_S_AUTHORITIES = {
    # 图灵奖级别 + 现役顶级决策者
    "geoffrey hinton", "yann lecun", "yoshua bengio", "demis hassabis",
    "ilya sutskever", "sam altman", "dario amodei", "jensen huang",
    "satya nadella", "sundar pichai"
}

TIER_A_AUTHORITIES = {
    # 顶级研究者 + 知名实验室负责人
    "andrej karpathy", "andrew ng", "fei-fei li", "jeff dean",
    "françois chollet", "pieter abbeel", "chelsea finn", "kaiming he",
    "openai", "deepmind", "anthropic", "meta ai", "google brain",
    "stanford ai lab", "berkeley ai research", "mit csail"
}

TIER_B_AUTHORITIES = {
    # 知名但非顶级
    "sebastian ruder", "jeremy howard", "rachel thomas", "chris olah",
    "google research", "microsoft research", "apple machine learning",
    "cmu", "eth zurich", "tsinghua", "peking university"
}

# 技术关键词：三层分级
CORE_TECH_L1 = {
    # 端侧AI核心（最高价值）
    "on-device ai", "on-device", "edge ai", "edge inference", "tinyml",
    "local llm", "local ai", "neural engine", "npu", "tpu acceleration",
    "int4", "int8", "4-bit quant", "8-bit quant", "gguf", "ggml",
    "llama.cpp", "mlx", "executorch", "coreml tools", "nnapi",
    "model compression", "neural compression"
}

CORE_TECH_L2 = {
    # 底层优化（高价值）
    "quantization", "pruning", "distillation", "knowledge distillation",
    "tensorrt", "tflite", "onnxruntime", "openvino", "webgpu",
    "wasm ai", "metal performance", "cuda kernel", "triton compiler",
    "flash attention", "paged attention", "kv cache optimization"
}

CORE_TECH_L3 = {
    # 相关技术（中价值）
    "transformer", "diffusion", "rag", "retrieval", "lora", "qlora",
    "peft", "adapter", "prefix tuning", "prompt tuning",
    "moe", "mixture of experts", "sparse model"
}

# 硬件关键词（必须配合技术词）
HARDWARE_TERMS = {
    "a18 pro", "a18 bionic", "a17 pro", "m4 chip", "m4 pro", "m4 max",
    "snapdragon 8 elite", "snapdragon 8 gen 3", "dimensity 9400",
    "google tensor", "exynos", "h100", "h200", "b100", "b200", "blackwell",
    "apple silicon", "arm mali", "qualcomm hexagon"
}

# 公司/产品（低基础分）
COMPANIES = {
    "apple", "google", "samsung", "qualcomm", "mediatek", "nvidia",
    "amd", "intel", "arm", "huawei", "xiaomi", "meta", "microsoft",
    "openai", "anthropic", "mistral", "cohere"
}

# 动作词（真实发布 vs 炒作）
STRONG_ACTIONS = {
    "release", "released", "launch", "launched", "ship", "shipped",
    "announce", "announced", "unveil", "unveiled", "available now",
    "open source", "open-source", "publish", "published"
}

WEAK_ACTIONS = {
    "preview", "beta", "demo", "prototype", "concept", "teaser",
    "coming soon", "will launch", "plans to", "expected to"
}

# 噪音词（分级惩罚）
HARD_NOISE = {
    # 金融/商业
    "stock price", "share price", "market cap", "quarterly earnings",
    "revenue beat", "profit margin", "dividend", "ipo", "acquisition deal",
    # 消费/促销
    "best deal", "discount code", "price drop", "coupon", "sale price",
    "limited time", "special offer",
    # 外设/配件
    "phone case", "screen protector", "wallpaper pack", "theme",
    "charging cable", "earbuds", "airpods case"
}

MEDIUM_NOISE = {
    # 谣言/炒作
    "rumor", "leak", "alleged", "reportedly", "sources say",
    "insider claims", "render", "mockup", "concept art",
    # 浅层内容
    "top 10", "best of", "ranking", "comparison", "vs battle",
    "tier list", "listicle"
}

SOFT_NOISE = {
    # 条件性噪音（如果没有硬核技术，则是噪音）
    "review", "hands-on", "unboxing", "first look", "impressions",
    "gameplay", "benchmark", "speed test"
}

# 权威域名（三级信任）
TIER_S_DOMAINS = {
    # 学术/官方
    "arxiv.org", "openreview.net", "ieeexplore.ieee.org", "dl.acm.org",
    "nature.com", "science.org", "pnas.org",
    # 顶级机构官网
    "openai.com", "anthropic.com", "deepmind.google", "ai.meta.com",
    "research.google", "machinelearning.apple.com", "developer.apple.com",
    "pytorch.org", "tensorflow.org", "huggingface.co"
}

TIER_A_DOMAINS = {
    # 知名科技媒体深度报道
    "techcrunch.com", "theverge.com", "arstechnica.com", "wired.com",
    # VC/智库
    "a16z.com", "sequoiacap.com", "ycombinator.com",
    # 开发者平台
    "github.com", "developer.nvidia.com", "developer.qualcomm.com"
}

TIER_B_DOMAINS = {
    # 社区/聚合
    "news.ycombinator.com", "reddit.com", "medium.com"
}

BLOCKED_DOMAINS = {
    "pinterest.com", "facebook.com", "instagram.com", "tiktok.com",
    "clickbait.com", "viralthread.com"
}

# ================= 🛠️ 工具函数 =================

def clean_text(text):
    """清理文本，移除多余空白"""
    return re.sub(r"\s+", " ", (text or "")).strip()

def normalize_url(url: str) -> str:
    """标准化URL，去除追踪参数"""
    try:
        p = urlparse(url)
        # 移除UTM和常见追踪参数
        q = [(k, v) for (k, v) in parse_qsl(p.query, keep_blank_values=True)
             if not any(k.lower().startswith(prefix) for prefix in ["utm_", "fb", "tw", "ig"])
             and k.lower() not in {"ref", "source", "feature", "campaign", "medium"}]
        return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q, doseq=True), ""))
    except:
        return url

def get_domain(url: str) -> str:
    """提取域名"""
    try:
        domain = urlparse(url).netloc.lower()
        return domain.replace("www.", "")
    except:
        return ""

def text_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（0-1）"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def content_fingerprint(title: str, desc: str = "") -> str:
    """生成内容指纹（用于去重）"""
    # 提取关键词，忽略停用词
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
    words = re.findall(r'\w+', (title + " " + desc).lower())
    key_words = [w for w in words if w not in stop_words and len(w) > 3]
    return hashlib.md5(" ".join(sorted(key_words[:15])).encode()).hexdigest()

def parse_date_from_arxiv(link: str) -> datetime.datetime:
    """从arXiv链接提取提交日期"""
    # arXiv格式: https://arxiv.org/abs/YYMM.NNNNN
    match = re.search(r'/(\d{4})\.', link)
    if match:
        yymm = match.group(1)
        year = 2000 + int(yymm[:2])
        month = int(yymm[2:])
        return datetime.datetime(year, month, 1)
    return datetime.datetime.now()

def time_decay_factor(pub_date: datetime.datetime) -> float:
    """时效性衰减因子（0.5-1.0）"""
    now = datetime.datetime.now()
    hours_old = (now - pub_date).total_seconds() / 3600
    
    if hours_old < 24:
        return 1.0  # 24小时内：满分
    elif hours_old < 48:
        return 0.9  # 48小时内：0.9
    elif hours_old < 168:  # 1周
        return 0.8
    else:
        return 0.7  # 1周以上：0.7

# ================= 🧠 语义分析引擎 =================

class SemanticAnalyzer:
    """语义分析：理解关键词组合和上下文"""
    
    @staticmethod
    def detect_tech_combos(text: str) -> list:
        """检测技术组合（比单个关键词更强）"""
        text_lower = text.lower()
        combos = []
        
        # 端侧 + 优化 = 核心主题
        if any(t in text_lower for t in ["on-device", "edge", "local", "mobile"]):
            if any(t in text_lower for t in ["quantization", "compression", "optimization", "pruning"]):
                combos.append(("edge_optimization", 60))
        
        # 硬件 + 加速 = 真实性能
        if any(h in text_lower for h in HARDWARE_TERMS):
            if any(t in text_lower for t in ["benchmark", "performance", "acceleration", "speedup"]):
                combos.append(("hardware_perf", 50))
        
        # 开源 + 工具 = 实用价值
        if any(t in text_lower for t in ["open source", "open-source", "github release"]):
            if any(t in text_lower for t in ["tool", "library", "framework", "sdk"]):
                combos.append(("opensource_tool", 45))
        
        # 论文 + 实验室 = 顶级研究
        if any(t in text_lower for t in ["paper", "arxiv", "research"]):
            if any(lab in text_lower for lab in TIER_S_AUTHORITIES | TIER_A_AUTHORITIES):
                combos.append(("top_research", 80))
        
        # 公司 + 强动作 = 真发布（非炒作）
        if any(c in text_lower for c in COMPANIES):
            if any(a in text_lower for a in STRONG_ACTIONS):
                combos.append(("real_release", 35))
        
        return combos
    
    @staticmethod
    def detect_negative_combos(text: str) -> list:
        """检测负面组合"""
        text_lower = text.lower()
        negatives = []
        
        # 公司 + 谣言 = 炒作
        if any(c in text_lower for c in COMPANIES):
            if any(n in text_lower for n in MEDIUM_NOISE):
                negatives.append(("company_rumor", -40))
        
        # 评测 + 无技术 = 浅层
        if any(s in text_lower for s in SOFT_NOISE):
            has_tech = any(t in text_lower for t in CORE_TECH_L1 | CORE_TECH_L2)
            if not has_tech:
                negatives.append(("shallow_review", -35))
        
        # 榜单 + 聚合 = 低质量
        if any(t in text_lower for t in ["top", "best", "ranking"]):
            if any(t in text_lower for t in ["apps", "tools", "services"]):
                negatives.append(("listicle", -30))
        
        return negatives
    
    @staticmethod
    def context_score(text: str) -> tuple:
        """上下文综合评分"""
        positive_combos = SemanticAnalyzer.detect_tech_combos(text)
        negative_combos = SemanticAnalyzer.detect_negative_combos(text)
        
        score = sum(s for _, s in positive_combos) + sum(s for _, s in negative_combos)
        reasons = [name for name, _ in positive_combos + negative_combos]
        
        return score, reasons

# ================= 🎯 高级评分引擎 =================

class AdvancedScorer:
    """多维度评分系统"""
    
    @staticmethod
    def score_authority(text: str) -> tuple:
        """权威评分"""
        score = 0
        reasons = []
        
        text_lower = text.lower()
        
        # S级权威：+100
        for auth in TIER_S_AUTHORITIES:
            if auth in text_lower:
                score += 100
                reasons.append(f"🏆S-Authority:{auth}(+100)")
        
        # A级权威：+70
        for auth in TIER_A_AUTHORITIES:
            if auth in text_lower:
                score += 70
                reasons.append(f"⭐A-Authority:{auth}(+70)")
        
        # B级权威：+40
        for auth in TIER_B_AUTHORITIES:
            if auth in text_lower:
                score += 40
                reasons.append(f"✓B-Authority:{auth}(+40)")
        
        return score, reasons
    
    @staticmethod
    def score_technical_depth(text: str) -> tuple:
        """技术深度评分（递减策略）"""
        score = 0
        reasons = []
        text_lower = text.lower()
        
        # L1技术：每个+40，最多3个，之后递减
        l1_count = 0
        for tech in CORE_TECH_L1:
            if tech in text_lower:
                l1_count += 1
                points = 40 if l1_count <= 3 else 10
                score += points
                reasons.append(f"L1-Tech:{tech}(+{points})")
        
        # L2技术：每个+25，最多2个
        l2_count = 0
        for tech in CORE_TECH_L2:
            if tech in text_lower:
                l2_count += 1
                if l2_count <= 2:
                    score += 25
                    reasons.append(f"L2-Tech:{tech}(+25)")
        
        # L3技术：每个+15，最多2个
        l3_count = 0
        for tech in CORE_TECH_L3:
            if tech in text_lower:
                l3_count += 1
                if l3_count <= 2:
                    score += 15
                    reasons.append(f"L3-Tech:{tech}(+15)")
        
        return score, reasons
    
    @staticmethod
    def score_domain_trust(url: str) -> tuple:
        """域名信任评分"""
        domain = get_domain(url)
        
        if domain in BLOCKED_DOMAINS:
            return -200, ["❌Blocked-Domain"]
        
        if domain in TIER_S_DOMAINS or any(domain.endswith(f".{d}") for d in TIER_S_DOMAINS):
            return 50, [f"🔒S-Domain:{domain}(+50)"]
        
        if domain in TIER_A_DOMAINS or any(domain.endswith(f".{d}") for d in TIER_A_DOMAINS):
            return 30, [f"✓A-Domain:{domain}(+30)"]
        
        if domain in TIER_B_DOMAINS:
            return 15, [f"B-Domain:{domain}(+15)"]
        
        return 0, []
    
    @staticmethod
    def score_noise(text: str) -> tuple:
        """噪音检测（分级惩罚）"""
        text_lower = text.lower()
        
        # 硬噪音：一票否决
        for noise in HARD_NOISE:
            if noise in text_lower:
                return -500, [f"❌HardNoise:{noise}"]
        
        # 中等噪音：多个累积
        medium_hits = sum(1 for noise in MEDIUM_NOISE if noise in text_lower)
        if medium_hits >= 2:
            return -300, [f"❌MediumNoise:hits={medium_hits}"]
        elif medium_hits == 1:
            return -50, ["⚠️MediumNoise:1hit"]
        
        # 软噪音：需要上下文判断
        soft_hits = [noise for noise in SOFT_NOISE if noise in text_lower]
        if soft_hits:
            has_hardcore = any(tech in text_lower for tech in CORE_TECH_L1 | CORE_TECH_L2)
            if not has_hardcore:
                return -150, [f"❌SoftNoise:{soft_hits[0]}-no-tech"]
            else:
                return -10, [f"⚠️SoftNoise:{soft_hits[0]}-with-tech"]
        
        return 0, []
    
    @staticmethod
    def score_industry_gate(text: str) -> tuple:
        """行业词门控（必须配合实质内容）"""
        text_lower = text.lower()
        
        has_company = any(c in text_lower for c in COMPANIES)
        has_hardware = any(h in text_lower for h in HARDWARE_TERMS)
        has_tech = any(t in text_lower for t in CORE_TECH_L1 | CORE_TECH_L2 | CORE_TECH_L3)
        has_strong_action = any(a in text_lower for a in STRONG_ACTIONS)
        
        if has_company or has_hardware:
            if has_tech or has_strong_action:
                return 20, ["✓Industry-Gate:pass(+20)"]
            else:
                return -30, ["❌Industry-Gate:fail(-30)"]
        
        return 0, []
    
    @staticmethod
    def comprehensive_score(title: str, desc: str, url: str, source: str, pub_date: datetime.datetime = None) -> tuple:
        """综合评分"""
        text = f"{title} {desc}"
        total_score = 0
        all_reasons = []
        
        # 1. 来源基础分
        source_base = {
            "HF Papers": 25,
            "AlphaXiv": 25,
            "TechCrunch": 15,
            "a16z": 20,
            "Hacker News": 10
        }.get(source, 10)
        total_score += source_base
        all_reasons.append(f"Source:{source}(+{source_base})")
        
        # 2. 域名信任
        domain_score, domain_reasons = AdvancedScorer.score_domain_trust(url)
        total_score += domain_score
        all_reasons.extend(domain_reasons)
        
        # 3. 噪音检测（可能一票否决）
        noise_score, noise_reasons = AdvancedScorer.score_noise(text)
        if noise_score <= -300:
            return noise_score, noise_reasons  # 立即返回
        total_score += noise_score
        all_reasons.extend(noise_reasons)
        
        # 4. 权威评分
        auth_score, auth_reasons = AdvancedScorer.score_authority(text)
        total_score += auth_score
        all_reasons.extend(auth_reasons)
        
        # 5. 技术深度
        tech_score, tech_reasons = AdvancedScorer.score_technical_depth(text)
        total_score += tech_score
        all_reasons.extend(tech_reasons)
        
        # 6. 行业门控
        gate_score, gate_reasons = AdvancedScorer.score_industry_gate(text)
        total_score += gate_score
        all_reasons.extend(gate_reasons)
        
        # 7. 语义分析
        context_score, context_reasons = SemanticAnalyzer.context_score(text)
        total_score += context_score
        all_reasons.extend(context_reasons)
        
        # 8. 时效性衰减
        if pub_date:
            decay = time_decay_factor(pub_date)
            total_score = int(total_score * decay)
            if decay < 1.0:
                all_reasons.append(f"TimeFactor:×{decay:.1f}")
        
        return total_score, all_reasons

# ================= 🔄 智能去重器 =================

class SmartDeduplicator:
    """智能去重：不仅看URL，还看内容相似度"""
    
    @staticmethod
    def deduplicate(items: list) -> list:
        """去重逻辑"""
        seen_urls = set()
        seen_fingerprints = defaultdict(list)
        unique_items = []
        
        for item in items:
            url = item.get("link", "")
            
            # 1. URL去重
            if url in seen_urls:
                continue
            
            # 2. 内容指纹去重
            fingerprint = content_fingerprint(item.get("title", ""), item.get("desc", ""))
            
            # 检查是否有高度相似的内容
            is_duplicate = False
            for existing in seen_fingerprints[fingerprint[:8]]:  # 前8位作为桶
                if text_similarity(item.get("title", ""), existing.get("title", "")) > 0.85:
                    # 标题85%相似，判定为重复
                    # 保留分数更高或来源更权威的
                    if item.get("score", 0) > existing.get("score", 0):
                        # 替换旧的
                        unique_items.remove(existing)
                        seen_fingerprints[fingerprint[:8]].remove(existing)
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                seen_urls.add(url)
                seen_fingerprints[fingerprint[:8]].append(item)
                unique_items.append(item)
        
        return unique_items

# ================= 🕷️ 抓取器（优化版） =================

def fetch_huggingface():
    """HuggingFace每日论文"""
    print("📄 Fetching HuggingFace Papers...")
    try:
        resp = requests.get("https://huggingface.co/papers", headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        articles = []
        for article in soup.find_all("article")[:25]:
            h3 = article.find("h3")
            a = article.find("a", href=True)
            if h3 and a:
                title = clean_text(h3.get_text())
                link = "https://huggingface.co" + a["href"] if a["href"].startswith("/") else a["href"]
                
                # 尝试提取描述
                desc_tag = article.find("p", class_="line-clamp-2")
                desc = clean_text(desc_tag.get_text()) if desc_tag else ""
                
                articles.append({
                    "title": title,
                    "link": normalize_url(link),
                    "source": "HF Papers",
                    "desc": desc,
                    "pub_date": datetime.datetime.now()
                })
        
        print(f"  ✓ Found {len(articles)} papers")
        return articles
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []

def fetch_arxiv():
    """arXiv最新论文"""
    print("📄 Fetching arXiv...")
    try:
        url = ("http://export.arxiv.org/api/query?"
               "search_query=cat:cs.AI+OR+cat:cs.CV+OR+cat:cs.CL+OR+cat:cs.LG"
               "&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending")
        
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        articles = []
        for entry in root.findall("atom:entry", ns):
            title_tag = entry.find("atom:title", ns)
            summary_tag = entry.find("atom:summary", ns)
            link_tag = entry.find("atom:id", ns)
            published_tag = entry.find("atom:published", ns)
            
            if title_tag is not None and link_tag is not None:
                title = clean_text(title_tag.text)
                summary = clean_text(summary_tag.text) if summary_tag is not None else ""
                link = link_tag.text
                
                # 解析发布日期
                pub_date = datetime.datetime.now()
                if published_tag is not None:
                    try:
                        pub_date = datetime.datetime.fromisoformat(published_tag.text.replace('Z', '+00:00'))
                    except:
                        pub_date = parse_date_from_arxiv(link)
                
                articles.append({
                    "title": title,
                    "link": normalize_url(link),
                    "source": "AlphaXiv",
                    "desc": summary[:400],
                    "pub_date": pub_date
                })
        
        print(f"  ✓ Found {len(articles)} papers")
        return articles
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []

def fetch_hacker_news_smart():
    """HN智能搜索（关键词过滤）"""
    print("📰 Fetching Hacker News (smart search)...")
    
    # 精选关键词（高信号）
    keywords = [
        "on-device ai", "edge ai", "local llm", "quantization",
        "llama.cpp", "mlx", "coreml", "onnxruntime", "executorch",
        "int4", "int8", "tinyml", "npu", "webgpu", "wasm"
    ]
    
    articles = []
    seen = set()
    
    try:
        for kw in keywords:
            url = f"https://hn.algolia.com/api/v1/search_by_date?query={requests.utils.quote(kw)}&tags=story&hitsPerPage=15"
            data = requests.get(url, timeout=15).json()
            
            for hit in data.get("hits", []):
                title = clean_text(hit.get("title", ""))
                link = hit.get("url", "")
                created_at = hit.get("created_at", "")
                
                if not title or not link:
                    continue
                
                link = normalize_url(link)
                if link in seen:
                    continue
                seen.add(link)
                
                # 解析时间
                pub_date = datetime.datetime.now()
                if created_at:
                    try:
                        pub_date = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except:
                        pass
                
                articles.append({
                    "title": title,
                    "link": link,
                    "source": "Hacker News",
                    "desc": "",
                    "pub_date": pub_date
                })
            
            time.sleep(0.1)
        
        print(f"  ✓ Found {len(articles)} items")
        return articles
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []

def fetch_techcrunch():
    """TechCrunch RSS"""
    print("📰 Fetching TechCrunch...")
    try:
        resp = requests.get(
            "https://techcrunch.com/category/artificial-intelligence/feed/",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        
        articles = []
        for item in root.findall("./channel/item")[:20]:
            title = clean_text(item.findtext("title", ""))
            link = clean_text(item.findtext("link", ""))
            desc = clean_text(item.findtext("description", ""))
            pub_date_str = item.findtext("pubDate", "")
            
            # 解析日期
            pub_date = datetime.datetime.now()
            if pub_date_str:
                try:
                    pub_date = email.utils.parsedate_to_datetime(pub_date_str)
                except:
                    pass
            
            if title and link:
                articles.append({
                    "title": title,
                    "link": normalize_url(link),
                    "source": "TechCrunch",
                    "desc": desc[:300],
                    "pub_date": pub_date
                })
        
        print(f"  ✓ Found {len(articles)} articles")
        return articles
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []

def fetch_a16z():
    """a16z RSS"""
    print("📰 Fetching a16z...")
    try:
        resp = requests.get(
            "https://a16z.com/feed/",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        
        articles = []
        for item in root.findall("./channel/item")[:20]:
            title = clean_text(item.findtext("title", ""))
            link = clean_text(item.findtext("link", ""))
            desc = clean_text(item.findtext("description", ""))
            
            if title and link:
                articles.append({
                    "title": title,
                    "link": normalize_url(link),
                    "source": "a16z",
                    "desc": desc[:300],
                    "pub_date": datetime.datetime.now()
                })
        
        print(f"  ✓ Found {len(articles)} articles")
        return articles
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []

# ================= 🎯 智能选择器 =================

class SmartSelector:
    """智能选择：不是简单的Top N，而是多样性+质量平衡"""
    
    @staticmethod
    def select_top(items: list, max_total: int = 10, category_quota: dict = None) -> list:
        """选择最佳内容"""
        if category_quota is None:
            category_quota = {
                "模型算法": 4,
                "平台底座": 3,
                "行业动态": 2,
                "大V/权威": 1
            }
        
        # 按类别分组
        by_category = defaultdict(list)
        for item in items:
            by_category[item["category"]].append(item)
        
        # 每个类别内部按分数排序
        for cat in by_category:
            by_category[cat].sort(key=lambda x: x["score"], reverse=True)
        
        selected = []
        used_links = set()
        
        # 1. 按配额选择
        for cat, quota in category_quota.items():
            for item in by_category.get(cat, [])[:quota]:
                if item["link"] not in used_links:
                    selected.append(item)
                    used_links.add(item["link"])
        
        # 2. 剩余名额：从所有类别中选最高分
        if len(selected) < max_total:
            all_remaining = [
                item for item in items
                if item["link"] not in used_links
            ]
            all_remaining.sort(key=lambda x: x["score"], reverse=True)
            
            for item in all_remaining:
                if len(selected) >= max_total:
                    break
                selected.append(item)
                used_links.add(item["link"])
        
        # 3. 最终按分数排序
        selected.sort(key=lambda x: x["score"], reverse=True)
        
        return selected[:max_total]

# ================= 📊 分类器 =================

def categorize(item: dict) -> str:
    """智能分类"""
    text = f"{item.get('title', '')} {item.get('desc', '')}".lower()
    source = item.get("source", "")
    
    # 1. 论文源 → 模型算法
    if source in ("HF Papers", "AlphaXiv"):
        return "模型算法"
    
    # 2. 权威 → 大V/权威
    if any(auth in text for auth in TIER_S_AUTHORITIES | TIER_A_AUTHORITIES):
        return "大V/权威"
    
    # 3. 端侧技术 → 平台底座
    if any(tech in text for tech in CORE_TECH_L1 | CORE_TECH_L2):
        return "平台底座"
    
    # 4. 公司+动作 → 行业动态
    if any(c in text for c in COMPANIES) and any(a in text for a in STRONG_ACTIONS):
        return "行业动态"
    
    # 5. 默认
    return "行业动态"

# ================= 🚀 主程序 =================

def main():
    print("\n" + "="*70)
    print("🚀 终极智能RSS聚合器 v3.0")
    print("="*70 + "\n")
    
    # 1. 抓取
    print("📡 Fetching from all sources...\n")
    all_items = []
    
    all_items.extend(fetch_huggingface())
    all_items.extend(fetch_arxiv())
    all_items.extend(fetch_hacker_news_smart())
    all_items.extend(fetch_techcrunch
