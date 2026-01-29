"""
🧠 智能RSS聚合器 v2.0
====================
核心理念：
1. 多维度评分：来源可信度 + 内容质量 + 关键词匹配
2. 动态阈值：不同来源使用不同标准
3. 上下文理解：不只看关键词，还要看组合和上下文
4. 负面信号强化：垃圾内容一票否决
"""

import requests
import datetime
import PyRSS2Gen
from bs4 import BeautifulSoup
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import os

# ================= 📊 数据结构定义 =================

class SourceTier(Enum):
    """来源等级：决定基础信任度"""
    TIER_S = 100  # 顶级：顶会论文、顶级实验室
    TIER_A = 80   # 一流：知名科技媒体的深度报道
    TIER_B = 60   # 良好：行业新闻，但需严格筛选
    TIER_C = 40   # 一般：聚合平台，需配合强关键词


@dataclass
class Article:
    """文章数据结构"""
    title: str
    link: str
    source: str
    description: str = ""
    author: str = ""
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    category: str = ""


# ================= 🎯 权威知识库 =================

class AuthorityDatabase:
    """权威人物和机构数据库"""
    
    # 图灵奖得主、领域奠基人（无论说什么都值得关注）
    PIONEERS = {
        "geoffrey hinton", "yann lecun", "yoshua bengio", "demis hassabis",
        "ilya sutskever", "andrej karpathy", "fei-fei li", "andrew ng",
        "jeff dean", "françois chollet", "jürgen schmidhuber", 
        "pieter abbeel", "chelsea finn", "kaiming he"
    }
    
    # 顶级研究机构（论文必看）
    TOP_LABS = {
        # 工业界
        "openai", "deepmind", "google brain", "google research", "meta ai", 
        "meta fair", "anthropic", "microsoft research", "apple ml research",
        "nvidia research", "stability ai",
        # 学术界
        "mit csail", "stanford", "berkeley", "cmu", "princeton", 
        "eth zurich", "oxford", "cambridge", "tsinghua", "peking university"
    }
    
    # 当前CEO/关键决策者（重大战略动态值得关注）
    CURRENT_LEADERS = {
        "sam altman": "OpenAI CEO",
        "satya nadella": "Microsoft CEO", 
        "sundar pichai": "Google CEO",
        "jensen huang": "NVIDIA CEO",
        "dario amodei": "Anthropic CEO",
        "mark zuckerberg": "Meta CEO",
        "elon musk": "xAI CEO"
    }
    
    # 知名研究者（需配合技术内容）
    RESEARCHERS = {
        "sebastian ruder", "jeremy howard", "rachel thomas", 
        "chris olah", "distill pub", "eleuther ai", "laion",
        "hugging face team", "simran kaur", "alex krizhevsky"
    }
    
    @classmethod
    def check_authority(cls, text: str) -> Tuple[int, List[str]]:
        """
        检查权威性
        返回：(分数, 匹配的权威)
        """
        text_lower = text.lower()
        score = 0
        matched = []
        
        # 先驱者：+150分（几乎确保入选）
        for pioneer in cls.PIONEERS:
            if pioneer in text_lower:
                score += 150
                matched.append(f"Pioneer: {pioneer.title()}")
        
        # 顶级实验室：+120分
        for lab in cls.TOP_LABS:
            if lab in text_lower:
                score += 120
                matched.append(f"Top Lab: {lab.title()}")
        
        # 现任领导者：+80分（但需注意是否为八卦新闻）
        for leader, title in cls.CURRENT_LEADERS.items():
            if leader in text_lower:
                score += 80
                matched.append(f"Leader: {leader.title()} ({title})")
        
        # 知名研究者：+60分
        for researcher in cls.RESEARCHERS:
            if researcher in text_lower:
                score += 60
                matched.append(f"Researcher: {researcher.title()}")
        
        return score, matched


# ================= 🔬 技术关键词库 =================

class TechnicalKeywords:
    """技术关键词分级系统"""
    
    # 核心技术（端侧AI/底层技术）- 高价值
    HARDCORE_EDGE_AI = {
        # 端侧推理
        "on-device ai", "edge ai", "tinyml", "mobile ai", "embedded ai",
        # 硬件加速
        "npu", "tpu", "neural engine", "tensor cores", "dsp acceleration",
        # 模型优化
        "quantization", "pruning", "knowledge distillation", "model compression",
        "int4", "int8", "fp16", "bnb", "awq", "gptq", "gguf",
        # 框架/工具
        "llama.cpp", "mlx", "executorch", "coreml", "tensorrt", "tflite", 
        "onnx runtime", "openvino",
        # 小模型
        "slm", "small language model", "phi-", "gemma", "tinyllama", "mobilevlm"
    }
    
    # 底层技术（架构/系统）- 高价值
    INFRASTRUCTURE = {
        "cuda kernels", "triton", "gpu optimization", "distributed training",
        "moe", "mixture of experts", "flash attention", "paged attention",
        "kv cache", "speculative decoding", "continuous batching",
        "tensor parallelism", "pipeline parallelism"
    }
    
    # 前沿算法（模型/训练）- 中高价值
    ALGORITHMS = {
        "transformer", "attention mechanism", "diffusion model", "vae",
        "rlhf", "dpo", "constitutional ai", "chain-of-thought", "reasoning",
        "retrieval augmented", "rag", "fine-tuning", "lora", "qlora",
        "sparse autoencoders", "mechanistic interpretability"
    }
    
    # 芯片/硬件 - 需配合技术内容
    CHIPS = {
        "a18 pro", "a18 bionic", "m4 chip", "m4 pro", "m4 max",
        "snapdragon 8 elite", "snapdragon 8 gen", "dimensity 9400",
        "google tensor", "exynos", "h100", "h200", "b200", "blackwell"
    }
    
    # 公司/产品 - 低价值，需强技术词配合
    COMPANIES = {
        "apple", "google", "samsung", "qualcomm", "mediatek",
        "nvidia", "amd", "intel", "arm", "huawei", "xiaomi",
        "openai", "anthropic", "meta", "microsoft"
    }
    
    @classmethod
    def analyze_technical_depth(cls, text: str) -> Tuple[int, List[str]]:
        """
        分析技术深度
        返回：(分数, 匹配的技术点)
        """
        text_lower = text.lower()
        score = 0
        matched = []
        
        # 核心技术：每个+50分
        for tech in cls.HARDCORE_EDGE_AI:
            if tech in text_lower:
                score += 50
                matched.append(f"Edge AI: {tech}")
        
        # 底层技术：每个+45分
        for infra in cls.INFRASTRUCTURE:
            if infra in text_lower:
                score += 45
                matched.append(f"Infrastructure: {infra}")
        
        # 算法：每个+35分
        for algo in cls.ALGORITHMS:
            if algo in text_lower:
                score += 35
                matched.append(f"Algorithm: {algo}")
        
        # 芯片：每个+20分（必须配合其他技术词）
        chip_count = sum(1 for chip in cls.CHIPS if chip in text_lower)
        if chip_count > 0:
            score += chip_count * 20
            matched.append(f"Hardware: {chip_count} chips mentioned")
        
        # 公司名：每个+5分（基础分，不够入选）
        company_count = sum(1 for company in cls.COMPANIES if company in text_lower)
        if company_count > 0:
            score += company_count * 5
        
        return score, matched


# ================= 🗑️ 噪音过滤器 =================

class NoiseFilter:
    """垃圾内容检测器"""
    
    # 金融/商业新闻（除非是重大战略）
    FINANCIAL_NOISE = {
        "stock price", "market cap", "quarterly earnings", "revenue beat",
        "shares surge", "dividend", "analyst rating", "price target",
        "stock split", "ipo"
    }
    
    # 消费者/评测（除非是技术深度评测）
    CONSUMER_NOISE = {
        "best deal", "discount", "sale", "price drop", "coupon",
        "unboxing", "hands-on first look", "top 10 apps", "wallpaper",
        "case", "screen protector", "accessory", "color options"
    }
    
    # 谣言/炒作
    RUMOR_NOISE = {
        "rumor", "leak suggests", "allegedly", "insider claims",
        "render shows", "concept design", "mockup", "speculation"
    }
    
    # 娱乐/社交
    ENTERTAINMENT_NOISE = {
        "meme", "viral", "tiktok trend", "instagram story",
        "celebrity", "influencer collab"
    }
    
    # 低质量聚合
    AGGREGATION_NOISE = {
        "this week in", "daily roundup", "news digest",
        "what you missed", "5 things to know"
    }
    
    @classmethod
    def check_noise(cls, text: str) -> Tuple[bool, List[str]]:
        """
        检测是否为噪音
        返回：(是否为噪音, 命中的噪音类型)
        """
        text_lower = text.lower()
        noise_found = []
        
        # 检查各类噪音
        for noise in cls.FINANCIAL_NOISE:
            if noise in text_lower:
                noise_found.append(f"Financial: {noise}")
        
        for noise in cls.CONSUMER_NOISE:
            if noise in text_lower:
                noise_found.append(f"Consumer: {noise}")
        
        for noise in cls.RUMOR_NOISE:
            if noise in text_lower:
                noise_found.append(f"Rumor: {noise}")
        
        for noise in cls.ENTERTAINMENT_NOISE:
            if noise in text_lower:
                noise_found.append(f"Entertainment: {noise}")
        
        for noise in cls.AGGREGATION_NOISE:
            if noise in text_lower:
                noise_found.append(f"Low-quality: {noise}")
        
        # 如果命中2个以上噪音关键词，判定为垃圾
        is_noise = len(noise_found) >= 2
        
        return is_noise, noise_found


# ================= 🎯 智能评分引擎 =================

class IntelligentScorer:
    """智能评分系统"""
    
    # 不同来源的基础分和阈值
    SOURCE_CONFIG = {
        "HF Papers": {"base": 100, "tier": SourceTier.TIER_S, "threshold": 120},
        "AlphaXiv": {"base": 100, "tier": SourceTier.TIER_S, "threshold": 120},
        "Hacker News": {"base": 40, "tier": SourceTier.TIER_C, "threshold": 80},
        "TechCrunch": {"base": 60, "tier": SourceTier.TIER_B, "threshold": 100},
        "a16z": {"base": 80, "tier": SourceTier.TIER_A, "threshold": 100},
    }
    
    @classmethod
    def score_article(cls, article: Article) -> Tuple[float, List[str], bool]:
        """
        综合评分
        返回：(总分, 评分原因, 是否通过)
        """
        text = f"{article.title} {article.description}"
        total_score = 0
        reasons = []
        
        # 1. 来源基础分
        config = cls.SOURCE_CONFIG.get(article.source, {"base": 50, "threshold": 80})
        source_base = config["base"]
        threshold = config["threshold"]
        
        total_score += source_base
        reasons.append(f"Source base: +{source_base} ({article.source})")
        
        # 2. 噪音检测（一票否决）
        is_noise, noise_reasons = NoiseFilter.check_noise(text)
        if is_noise:
            reasons.append(f"❌ NOISE DETECTED: {', '.join(noise_reasons)}")
            return -100, reasons, False
        
        # 3. 权威性检查
        auth_score, auth_matches = AuthorityDatabase.check_authority(text)
        if auth_score > 0:
            total_score += auth_score
            reasons.extend(auth_matches)
        
        # 4. 技术深度分析
        tech_score, tech_matches = TechnicalKeywords.analyze_technical_depth(text)
        if tech_score > 0:
            total_score += tech_score
            reasons.extend(tech_matches)
        
        # 5. 特殊加成：标题包含技术词+权威
        title_lower = article.title.lower()
        if any(tech in title_lower for tech in TechnicalKeywords.HARDCORE_EDGE_AI):
            if any(auth in title_lower for auth in AuthorityDatabase.TOP_LABS):
                total_score += 30
                reasons.append("Bonus: Tech+Authority in title")
        
        # 6. 判断是否通过
        passed = total_score >= threshold
        
        return total_score, reasons, passed
    
    @classmethod
    def categorize(cls, article: Article, reasons: List[str]) -> str:
        """智能分类"""
        # 从评分原因推断类别
        reason_text = " ".join(reasons).lower()
        
        if "pioneer" in reason_text or "leader" in reason_text:
            return "🎓 权威发声"
        elif "top lab" in reason_text:
            return "🔬 顶级研究"
        elif "edge ai" in reason_text or "infrastructure" in reason_text:
            return "⚡ 端侧/底层技术"
        elif "algorithm" in reason_text:
            return "🧠 模型算法"
        elif "hardware" in reason_text:
            return "💻 芯片硬件"
        else:
            return "📰 行业动态"


# ================= 🕷️ 数据抓取器 =================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def clean_text(text: str) -> str:
    """清理文本"""
    if not text: return ""
    return re.sub(r'\s+', ' ', text).strip()


def fetch_huggingface() -> List[Article]:
    """抓取Hugging Face每日论文（顶级来源）"""
    print("📄 Fetching Hugging Face Papers...")
    try:
        resp = requests.get("https://huggingface.co/papers", headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        articles = []
        
        for article in soup.find_all('article')[:20]:  # 只看前20篇
            h3 = article.find('h3')
            if h3:
                title = clean_text(h3.get_text())
                link_tag = article.find('a')
                if link_tag and 'href' in link_tag.attrs:
                    link = "https://huggingface.co" + link_tag['href']
                    
                    # 尝试提取作者信息
                    author_tag = article.find('div', class_='text-sm')
                    author = clean_text(author_tag.get_text()) if author_tag else ""
                    
                    articles.append(Article(
                        title=title,
                        link=link,
                        source="HF Papers",
                        author=author
                    ))
        
        print(f"  ✓ Found {len(articles)} papers")
        return articles
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []


def fetch_arxiv() -> List[Article]:
    """抓取arXiv最新论文（顶级来源）"""
    print("📄 Fetching arXiv Papers...")
    try:
        # 只抓取AI/CV/CL/LG相关的最新论文
        url = "http://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.CV+OR+cat:cs.CL+OR+cat:cs.LG&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending"
        resp = requests.get(url, timeout=15)
        root = ET.fromstring(resp.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        articles = []
        for entry in root.findall('atom:entry', ns):
            title_tag = entry.find('atom:title', ns)
            summary_tag = entry.find('atom:summary', ns)
            link_tag = entry.find('atom:id', ns)
            authors = entry.findall('atom:author', ns)
            
            if title_tag is not None and link_tag is not None:
                title = clean_text(title_tag.text)
                summary = clean_text(summary_tag.text) if summary_tag is not None else ""
                link = link_tag.text
                
                # 提取第一作者
                author = ""
                if authors:
                    author_name = authors[0].find('atom:name', ns)
                    if author_name is not None:
                        author = clean_text(author_name.text)
                
                articles.append(Article(
                    title=title,
                    link=link,
                    source="AlphaXiv",
                    description=summary[:300],
                    author=author
                ))
        
        print(f"  ✓ Found {len(articles)} papers")
        return articles
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []


def fetch_hacker_news() -> List[Article]:
    """抓取Hacker News（需严格筛选）"""
    print("📰 Fetching Hacker News...")
    try:
        # 使用 Algolia API 搜索关键词，比抓取首页更精准
        queries = ["edge ai", "on-device", "llm", "npu", "quantization", "apple intelligence"]
        articles = []
        seen_ids = set()
        
        for q in queries:
            try:
                url = f"https://hn.algolia.com/api/v1/search_by_date?query={q}&tags=story&hitsPerPage=10"
                resp = requests.get(url, timeout=5).json()
                
                for hit in resp.get('hits', []):
                    obj_id = hit.get('objectID')
                    if obj_id in seen_ids: continue
                    seen_ids.add(obj_id)
                    
                    title = clean_text(hit.get('title'))
                    link = hit.get('url') or f"https://news.ycombinator.com/item?id={obj_id}"
                    
                    if title:
                        articles.append(Article(
                            title=title,
                            link=link,
                            source="Hacker News"
                        ))
                time.sleep(0.1)
            except:
                continue
        
        print(f"  ✓ Found {len(articles)} items")
        return articles
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []


def fetch_techcrunch() -> List[Article]:
    """抓取TechCrunch AI板块（需筛选）"""
    print("📰 Fetching TechCrunch...")
    try:
        resp = requests.get(
            "https://techcrunch.com/category/artificial-intelligence/",
            headers=HEADERS,
            timeout=10
        )
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        articles = []
        # 尝试多个可能的选择器
        for selector in ['.post-block__title a', '.loop-card__title a', 'h2 a', 'h3 a']:
            links = soup.select(selector)
            for link in links[:15]:  # 每个选择器最多15个
                title = clean_text(link.get_text())
                if len(title) > 10:  # 过滤太短的标题
                    articles.append(Article(
                        title=title,
                        link=link.get('href', ''),
                        source="TechCrunch"
                    ))
        
        # 去重
        seen = set()
        unique_articles = []
        for article in articles:
            if article.link and article.link not in seen:
                seen.add(article.link)
                unique_articles.append(article)
        
        print(f"  ✓ Found {len(unique_articles)} articles")
        return unique_articles
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []


def fetch_a16z() -> List[Article]:
    """抓取a16z（顶级VC视角）"""
    print("📰 Fetching a16z...")
    try:
        resp = requests.get("https://a16z.com/news-content/", headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        articles = []
        for link in soup.find_all('a', href=True):
            title = clean_text(link.get_text())
            href = link['href']
            
            # 简单的启发式过滤
            if len(title) > 15 and "ai" in href.lower():
                if href.startswith('/'):
                    href = 'https://a16z.com' + href
                
                articles.append(Article(
                    title=title,
                    link=href,
                    source="a16z"
                ))
        
        # 去重
        seen = set()
        unique = []
        for a in articles:
            if a.link not in seen:
                seen.add(a.link)
                unique.append(a)
                
        print(f"  ✓ Found {len(unique)} articles")
        return unique
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []


# ================= 🚀 主程序 =================

def main():
    """主函数"""
    print("\n" + "="*60)
    print("🧠 Intelligent RSS Aggregator v2.0")
    print("="*60 + "\n")
    
    # 1. 抓取所有来源
    print("📡 Fetching from all sources...\n")
    all_articles = []
    
    all_articles.extend(fetch_huggingface())
    all_articles.extend(fetch_arxiv())
    all_articles.extend(fetch_hacker_news())
    all_articles.extend(fetch_techcrunch())
    all_articles.extend(fetch_a16z())
    
    print(f"\n📊 Total raw articles: {len(all_articles)}\n")
    print("="*60)
    print("🎯 Scoring and filtering...\n")
    
    # 2. 去重
    seen_links = set()
    unique_articles = []
    for article in all_articles:
        if article.link and article.link not in seen_links:
            seen_links.add(article.link)
            unique_articles.append(article)
    
    # 3. 评分和筛选
    selected_articles = []
    rejected_count = 0
    
    for article in unique_articles:
        score, reasons, passed = IntelligentScorer.score_article(article)
        
        if passed:
            article.score = score
            article.reasons = reasons
            article.category = IntelligentScorer.categorize(article, reasons)
            selected_articles.append(article)
            
            print(f"✅ [{score:.0f}] {article.category}")
            print(f"   {article.title[:80]}")
            print(f"   Source: {article.source}")
            if article.author:
                print(f"   Author: {article.author}")
            print(f"   Reasons: {reasons[0] if reasons else 'N/A'}")
            print()
        else:
            rejected_count += 1
            if score > 50:  # 只显示高分但未通过的（帮助调试）
                print(f"❌ [{score:.0f}] {article.title[:60]}...")
                print(f"   Reason: {reasons[0] if reasons else 'Below threshold'}")
                print()
    
    # 4. 按分数排序
    selected_articles.sort(key=lambda x: x.score, reverse=True)
    
    # 5. 生成RSS
    print("="*60)
    print(f"📝 Generating RSS feed...\n")
    
    rss_items = []
    for article in selected_articles:
        # 构建描述
        desc_parts = [
            f"<div style='font-family: Arial, sans-serif;'>",
            f"<p><strong>📊 Quality Score: {article.score:.0f}</strong></p>",
            f"<p><strong>📂 Category:</strong> {article.category}</p>",
            f"<p><strong>🔍 Source:</strong> {article.source}</p>"
        ]
        
        if article.author:
            desc_parts.append(f"<p><strong>✍️ Author:</strong> {article.author}</p>")
        
        desc_parts.append(f"<p><strong>✨ Why selected:</strong></p><ul>")
        for reason in article.reasons[:5]:  # 最多显示5个原因
            desc_parts.append(f"<li>{reason}</li>")
        desc_parts.append("</ul>")
        
        if article.description:
            desc_parts.append(f"<p><strong>📄 Summary:</strong> {article.description[:400]}...</p>")
        
        desc_parts.append("</div>")
        
        description = "\n".join(desc_parts)
        
        rss_items.append(PyRSS2Gen.RSSItem(
            title=f"[{article.score:.0f}] {article.category} | {article.title}",
            link=article.link,
            description=description,
            pubDate=datetime.datetime.now()
        ))
    
    # 生成RSS文件 (修改为当前目录，适配 GitHub Actions)
    rss = PyRSS2Gen.RSS2(
        title="🧠 Intelligent AI & Tech Feed",
        link="https://github.com/paramita619/hf-daily-paper-rss",
        description="High-quality, authority-focused feed for AI research, edge computing, and technical breakthroughs. Powered by multi-dimensional scoring.",
        lastBuildDate=datetime.datetime.now(),
        items=rss_items
    )
    
    output_file = "edge_ai_daily.xml"  # 修复后的文件名
    with open(output_file, "w", encoding='utf-8') as f:
        rss.write_xml(f)
    
    # 6. 统计报告
    print("="*60)
    print("📊 FINAL REPORT")
    print("="*60)
    print(f"Total articles fetched: {len(all_articles)}")
    print(f"Unique articles: {len(unique_articles)}")
    print(f"Articles passed filter: {len(selected_articles)}")
    print(f"Articles rejected: {rejected_count}")
    pass_rate = len(selected_articles)/len(unique_articles)*100 if unique_articles else 0
    print(f"Pass rate: {pass_rate:.1f}%")
    print()
    
    print(f"✅ RSS feed generated: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()
