import requests
import datetime
import PyRSS2Gen
from bs4 import BeautifulSoup
import time
import re

# ================= 配置区域 =================

# 1. 核心筛选关键词 (必须包含这些才能进入候选池)
# 确保只保留 端侧/边缘 AI 相关内容
CORE_KEYWORDS = [
    "edge ai", "on-device", "mobile", "tinyml", "embedded",
    "quantization", "pruning", "efficiency", "inference",
    "smartphone", "npu", "android", "ios", "latency",
    "apple intelligence", "pixel", "samsung", "snapdragon", 
    "mediatek", "local llm", "small language model", "slm"
]

# 2. 分类规则字典 (用于自动打标签)
# 优先级：大V访谈 > 平台底座 > 行业动态 > 模型算法 (默认)
CATEGORIES = {
    "大V访谈": [
        "interview", "talks with", "podcast", "conversation", "fireside chat", 
        "sam altman", "yann lecun", "demis hassabis", "andrew ng", "geoffrey hinton",
        "kai-fu lee", "fei-fei li", "ilya sutskever"
    ],
    "平台底座": [
        "framework", "cuda", "nvidia", "gpu", "tpu", "npu", "hardware", "chip", 
        "processor", "infrastructure", "library", "pytorch", "tensorflow", "mlx",
        "coreml", "tensorrt", "qualcomm", "arm", "risc-v", "compute"
    ],
    "行业动态": [
        "launches", "released", "announces", "update", "consumer", "app", 
        "feature", "market", "apple", "google", "samsung", "microsoft", 
        "meta", "rollout", "available now", "production"
    ],
    "模型算法": [
        "paper", "model", "architecture", "transformer", "diffusion", "rag",
        "state-of-the-art", "sota", "benchmark", "dataset", "training",
        "algorithm", "fine-tuning", "lora"
    ]
}

# 伪装头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# ================= 工具函数 =================

def classify_article(title, source_name):
    """
    根据标题和来源自动分类
    """
    title_lower = title.lower()
    
    # 规则 1: 强来源绑定
    # 比如 Hugging Face 和 AlphaXiv 99% 是模型算法，除非标题里明显是访谈
    if source_name in ["HF Papers", "AlphaXiv"]:
        if any(k in title_lower for k in CATEGORIES["大V访谈"]):
            return "大V访谈"
        return "模型算法" # 默认归宿

    # 规则 2: 关键词匹配
    # 按照优先级顺序检查
    for category, keywords in CATEGORIES.items():
        if any(k in title_lower for k in keywords):
            return category
            
    # 规则 3: 默认兜底
    # 如果是 TechCrunch/HN 但没命中任何词，通常算行业动态
    if source_name in ["TechCrunch", "Hacker News", "a16z"]:
        return "行业动态"
        
    return "其他"

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# ================= 抓取函数 =================

def fetch_huggingface():
    print("Fetching Hugging Face Papers...")
    try:
        resp = requests.get("https://huggingface.co/papers", headers=HEADERS)
        soup = BeautifulSoup(resp.text, 'html.parser')
        papers = []
        for article in soup.find_all('article'):
            h3 = article.find('h3')
            if h3:
                title = clean_text(h3.get_text())
                # HF 的链接通常是相对路径
                link = "https://huggingface.co" + article.find('a')['href']
                papers.append({'title': title, 'link': link, 'source': 'HF Papers'})
        return papers
    except Exception as e:
        print(f"HF Error: {e}")
        return []

def fetch_hacker_news():
    print("Fetching Hacker News...")
    try:
        # 只抓 Top 50，避免超时
        ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json").json()[:50]
        articles = []
        for id in ids:
            item = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{id}.json").json()
            if item and 'title' in item and 'url' in item:
                articles.append({'title': clean_text(item['title']), 'link': item['url'], 'source': 'Hacker News'})
        return articles
    except Exception as e:
        print(f"HN Error: {e}")
        return []

def fetch_techcrunch():
    print("Fetching TechCrunch AI...")
    try:
        resp = requests.get("https://techcrunch.com/category/artificial-intelligence/", headers=HEADERS)
        soup = BeautifulSoup(resp.text, 'html.parser')
        articles = []
        # TechCrunch 结构经常变，尝试抓取 loop-card__title
        for h2 in soup.find_all('h2', class_='loop-card__title'):
            a = h2.find('a')
            if a:
                articles.append({'title': clean_text(a.get_text()), 'link': a['href'], 'source': 'TechCrunch'})
        
        # 备用选择器 (旧版结构)
        if not articles:
             for h2 in soup.find_all('h2', class_='post-block__title'):
                a = h2.find('a')
                if a:
                    articles.append({'title': clean_text(a.get_text()), 'link': a['href'], 'source': 'TechCrunch'})
        return articles
    except Exception as e:
        print(f"TechCrunch Error: {e}")
        return []

def fetch_alphaxiv():
    print("Fetching AlphaXiv (via arXiv fallback)...")
    # AlphaXiv 主要是 arXiv 的壳，直接抓 arXiv API 搜索更稳定，算作 AlphaXiv 来源
    try:
        # 搜索最近提交的 CS/AI 论文
        url = "http://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.CV+OR+cat:cs.CL&start=0&max_results=20&sortBy=submittedDate&sortOrder=desc"
        resp = requests.get(url)
        soup = BeautifulSoup(resp.content, 'xml') # 使用 xml parser
        articles = []
        for entry in soup.find_all('entry'):
            title = clean_text(entry.title.text)
            link = entry.id.text
            # 简单把 arXiv 链接转成 AlphaXiv 风格（如果需要）
            # alphaxiv_link = link.replace("arxiv.org/abs", "www.alphaxiv.org/abs")
            articles.append({'title': title, 'link': link, 'source': 'AlphaXiv'})
        return articles
    except Exception as e:
        print(f"AlphaXiv Error: {e}")
        return []

def fetch_a16z():
    print("Fetching a16z AI...")
    try:
        resp = requests.get("https://a16z.com/news-content/", headers=HEADERS)
        soup = BeautifulSoup(resp.text, 'html.parser')
        articles = []
        # 查找所有文章标题链接
        for a in soup.find_all('a', href=True):
            # 简单的启发式过滤：标题长度 > 10 且 URL 包含 ai
            text = clean_text(a.get_text())
            if len(text) > 15 and "ai" in a['href']:
                # 排除导航栏等无效链接
                if "page" not in a['href']:
                     articles.append({'title': text, 'link': a['href'], 'source': 'a16z'})
        # 去重
        seen = set()
        unique_articles = []
        for art in articles:
            if art['link'] not in seen:
                unique_articles.append(art)
                seen.add(art['link'])
        return unique_articles
    except Exception as e:
        print(f"a16z Error: {e}")
        return []

# ================= 主程序 =================

def process_feeds():
    all_raw_items = []
    
    # 1. 并行抓取 (单线程依次执行)
    all_raw_items.extend(fetch_huggingface())
    all_raw_items.extend(fetch_hacker_news())
    all_raw_items.extend(fetch_techcrunch())
    all_raw_items.extend(fetch_alphaxiv())
    all_raw_items.extend(fetch_a16z())
    
    print(f"Total raw items fetched: {len(all_raw_items)}")
    
    rss_items = []
    seen_links = set()
    
    # 2. 筛选与分类
    for item in all_raw_items:
        title = item['title']
        link = item['link']
        
        # A. 去重
        if link in seen_links: continue
        seen_links.add(link)
        
        # B. 核心筛选 (Edge AI 相关性)
        # 只要标题里包含 CORE_KEYWORDS 里的任意一个词
        if not any(k in title.lower() for k in CORE_KEYWORDS):
            continue
            
        # C. 自动分类
        category = classify_article(title, item['source'])
        
        # D. 格式化输出标题
        # 格式: [行业动态] Apple Intelligence Launch (TechCrunch)
        final_title = f"[{category}] {title} ({item['source']})"
        
        rss_items.append(PyRSS2Gen.RSSItem(
            title=final_title,
            link=link,
            description=f"Source: {item['source']} | Category: {category}",
            pubDate=datetime.datetime.now()
        ))
        
    # 3. 生成 RSS
    rss = PyRSS2Gen.RSS2(
        title="Edge AI & Device Intelligence Daily",
        link="https://github.com/YourRepo",
        description="Daily curated AI news classified by Model, Platform, Industry, and Interviews.",
        lastBuildDate=datetime.datetime.now(),
        items=rss_items
    )
    
    rss.write_xml(open("edge_ai_daily.xml", "w", encoding='utf-8'))
    print(f"Successfully generated RSS with {len(rss_items)} items.")

if __name__ == "__main__":
    process_feeds()
