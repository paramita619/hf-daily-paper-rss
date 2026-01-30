"""
🚀 混合架构 RSS 智能分析系统 v4.0
====================================
核心能力：
1. 深度网页抓取 - 提取完整正文
2. 语义分析引擎 - 理解内容主题
3. 智能模板生成 - 模拟深度分析
4. 精美EML报告 - 完整HTML格式

无需API，完全本地运行
"""

import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import base64
from urllib.parse import urlparse
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib

# ================= 🌐 深度网页抓取器 =================

class DeepWebScraper:
    """深度网页内容提取器"""
    
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    }
    
    @staticmethod
    def extract_article_content(url: str) -> dict:
        """深度提取文章内容"""
        print(f"  🔍 正在深度抓取: {url}")
        
        try:
            resp = requests.get(url, headers=DeepWebScraper.HEADERS, timeout=30, allow_redirects=True)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 移除干扰元素
            for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                tag.decompose()
            
            # 提取标题
            title = DeepWebScraper._extract_title(soup, url)
            
            # 提取正文
            content = DeepWebScraper._extract_main_content(soup)
            
            # 提取元数据
            metadata = DeepWebScraper._extract_metadata(soup, url)
            
            # 提取关键实体
            entities = DeepWebScraper._extract_entities(content, title)
            
            print(f"  ✅ 提取成功: 标题={title[:30]}..., 正文={len(content)}字")
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'metadata': metadata,
                'entities': entities,
                'word_count': len(content),
                'success': True
            }
            
        except Exception as e:
            print(f"  ❌ 抓取失败: {e}")
            return {
                'url': url,
                'title': '抓取失败',
                'content': '',
                'metadata': {},
                'entities': {},
                'word_count': 0,
                'success': False
            }
    
    @staticmethod
    def _extract_title(soup: BeautifulSoup, url: str) -> str:
        """智能提取标题"""
        # 尝试多种选择器
        selectors = [
            'h1',
            '.article-title',
            '.post-title',
            '.entry-title',
            '[itemprop="headline"]',
            '.title',
            'title'
        ]
        
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                title = elem.get_text().strip()
                if len(title) > 10 and len(title) < 200:
                    # 清理标题
                    title = re.sub(r'\s+', ' ', title)
                    title = re.sub(r'\|.*$', '', title)  # 移除网站名
                    title = re.sub(r' - .*$', '', title)
                    return title.strip()
        
        # 从URL推断
        if 'arxiv.org' in url:
            return "arXiv论文"
        
        return "未知标题"
    
    @staticmethod
    def _extract_main_content(soup: BeautifulSoup) -> str:
        """提取主要内容"""
        # 内容选择器优先级
        content_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main',
            '[role="main"]',
            '#content',
            '.article-body',
        ]
        
        for selector in content_selectors:
            container = soup.select_one(selector)
            if container:
                # 提取段落
                paragraphs = []
                for elem in container.find_all(['p', 'h2', 'h3', 'h4', 'li', 'blockquote']):
                    text = elem.get_text().strip()
                    if len(text) > 20:  # 过滤太短的段落
                        paragraphs.append(text)
                
                content = '\n\n'.join(paragraphs)
                if len(content) > 500:  # 确保有足够内容
                    return content[:15000]  # 限制最大长度
        
        # 备选方案：直接提取所有段落
        all_paragraphs = soup.find_all('p')
        valid_paras = [p.get_text().strip() for p in all_paragraphs if len(p.get_text().strip()) > 30]
        
        if valid_paras:
            return '\n\n'.join(valid_paras[:100])[:15000]
        
        return ""
    
    @staticmethod
    def _extract_metadata(soup: BeautifulSoup, url: str) -> dict:
        """提取元数据"""
        metadata = {
            'author': '',
            'date': '',
            'source': '',
            'organization': '',  # 新增
            'tags': []
        }
        
        # 提取作者
        author_selectors = ['.author', '.byline', '[rel="author"]', '[itemprop="author"]']
        for selector in author_selectors:
            elem = soup.select_one(selector)
            if elem:
                metadata['author'] = elem.get_text().strip()
                break
        
        # 从URL推断来源和机构
        domain = urlparse(url).netloc
        url_lower = url.lower()
        
        # 知名机构映射（扩展）
        org_mapping = {
            'arxiv.org': ('arXiv', 'arXiv'),
            'huggingface.co': ('Hugging Face', 'Hugging Face'),
            'github.com': ('GitHub', 'GitHub'),
            'openai.com': ('OpenAI', 'OpenAI'),
            'anthropic.com': ('Anthropic', 'Anthropic'),
            'deepmind.com': ('DeepMind', 'DeepMind'),
            'deepmind.google': ('DeepMind', 'DeepMind'),
            'research.google': ('Google Research', 'Google'),
            'google.com': ('Google', 'Google'),
            'ai.meta.com': ('Meta AI', 'Meta'),
            'meta.com': ('Meta', 'Meta'),
            'microsoft.com': ('Microsoft', 'Microsoft'),
            'nvidia.com': ('NVIDIA', 'NVIDIA'),
            'apple.com': ('Apple', 'Apple'),
            'techcrunch.com': ('TechCrunch', 'TechCrunch'),
            'reuters.com': ('Reuters', 'Reuters'),
            'bloomberg.com': ('Bloomberg', 'Bloomberg'),
            'theverge.com': ('The Verge', 'The Verge'),
            'wired.com': ('Wired', 'Wired'),
            'x.com': ('Twitter/X', 'Twitter/X'),
            'twitter.com': ('Twitter/X', 'Twitter/X'),
            'stanford.edu': ('Stanford', 'Stanford'),
            'mit.edu': ('MIT', 'MIT'),
            'berkeley.edu': ('Berkeley', 'Berkeley'),
            'cmu.edu': ('CMU', 'CMU'),
            'macrumors.com': ('MacRumors', 'MacRumors'),
        }
        
        # 查找匹配
        found = False
        for key, (source, org) in org_mapping.items():
            if key in url_lower:
                metadata['source'] = source
                metadata['organization'] = org
                found = True
                break
        
        if not found:
            metadata['source'] = domain.replace('www.', '')
            metadata['organization'] = domain.replace('www.', '').split('.')[0].title()
        
        # arXiv特殊处理 - 从页面提取第一作者机构
        if 'arxiv.org' in url_lower:
            org = DeepWebScraper._extract_arxiv_org(soup)
            if org and org != 'arXiv':
                metadata['organization'] = org
        
        # 提取标签
        tag_elems = soup.select('.tag, .label, [rel="tag"]')
        metadata['tags'] = [tag.get_text().strip() for tag in tag_elems[:10]]
        
        return metadata
    
    @staticmethod
    def _extract_arxiv_org(soup: BeautifulSoup) -> str:
        """从arXiv页面提取第一作者机构"""
        try:
            # 方法1: 查找authors div
            authors_div = soup.find('div', class_='authors')
            if not authors_div:
                authors_div = soup.find('div', class_='ltx_authors')
            
            if authors_div:
                # 查找第一个机构标记
                affiliation = authors_div.find('span', class_='ltx_contact ltx_role_affiliation')
                if not affiliation:
                    affiliation = authors_div.find('div', class_='ltx_author_notes')
                
                if affiliation:
                    org_text = affiliation.get_text(strip=True)
                    # 标准化知名机构名
                    org_map = {
                        'MIT': ['Massachusetts Institute', 'MIT'],
                        'Stanford': ['Stanford'],
                        'Berkeley': ['Berkeley', 'UC Berkeley'],
                        'CMU': ['Carnegie Mellon', 'CMU'],
                        'Google': ['Google'],
                        'Meta': ['Meta', 'Facebook'],
                        'OpenAI': ['OpenAI'],
                        'Microsoft': ['Microsoft'],
                        'DeepMind': ['DeepMind'],
                        'NVIDIA': ['NVIDIA'],
                        'Apple': ['Apple'],
                    }
                    
                    for standard_name, patterns in org_map.items():
                        if any(p in org_text for p in patterns):
                            return standard_name
                    
                    # 如果没有匹配，返回前30个字符
                    return org_text[:30] if len(org_text) > 0 else 'arXiv'
            
            # 方法2: 从标题提取（如果标题中有括号信息）
            title = soup.find('h1', class_='title')
            if title:
                title_text = title.get_text()
                # 查找括号中的机构信息
                import re
                match = re.search(r'\(([^)]+)\)', title_text)
                if match:
                    org = match.group(1)
                    if any(keyword in org for keyword in ['University', 'Institute', 'Lab', 'Inc', 'Corp']):
                        return org[:30]
        
        except:
            pass
        
        return 'arXiv'
    
    @staticmethod
    def _extract_entities(content: str, title: str) -> dict:
        """提取关键实体"""
        text = (title + ' ' + content).lower()
        
        entities = {
            'companies': [],
            'technologies': [],
            'people': [],
            'models': [],
            'metrics': []
        }
        
        # 公司/组织
        companies = [
            'openai', 'anthropic', 'google', 'deepmind', 'meta', 'microsoft',
            'nvidia', 'apple', 'hugging face', 'stanford', 'mit', 'berkeley',
            'cmu', 'oxford', 'cambridge'
        ]
        entities['companies'] = [c for c in companies if c in text]
        
        # 技术关键词
        technologies = [
            'transformer', 'diffusion', 'quantization', 'pruning', 'distillation',
            'rlhf', 'dpo', 'lora', 'qlora', 'peft', 'rag', 'moe',
            'cuda', 'tensorrt', 'onnx', 'pytorch', 'tensorflow'
        ]
        entities['technologies'] = [t for t in technologies if t in text]
        
        # 知名人物
        people = [
            'hinton', 'lecun', 'bengio', 'altman', 'hassabis', 'sutskever',
            'karpathy', 'ng', 'dean', 'chollet'
        ]
        entities['people'] = [p for p in people if p in text]
        
        # 模型名称
        models = [
            'gpt-4', 'gpt-5', 'claude', 'gemini', 'llama', 'mistral',
            'phi', 'gemma', 'stable diffusion', 'dall-e', 'midjourney'
        ]
        entities['models'] = [m for m in models if m in text]
        
        # 性能指标
        metrics_patterns = [
            r'(\d+\.?\d*)x\s+faster',
            r'(\d+\.?\d*)%\s+accuracy',
            r'(\d+\.?\d*)\s+(?:billion|million)\s+parameters',
            r'(\d+\.?\d*)\s+(?:flops|tflops|gflops)'
        ]
        for pattern in metrics_patterns:
            matches = re.findall(pattern, text)
            entities['metrics'].extend(matches[:3])
        
        return entities

# ================= 🧠 语义分析引擎 (增强版) =================

class SemanticAnalyzerV4:
    """增强版语义分析器 - 基于v3.1但更智能"""
    
    # 分类特征库
    CATEGORY_FEATURES = {
        "模型算法": {
            "core_keywords": [
                "model", "algorithm", "architecture", "training", "optimization",
                "transformer", "attention", "neural network", "deep learning",
                "llm", "large language model", "vision model", "multimodal",
                "parameters", "layers", "embedding", "tokenization"
            ],
            "technique_keywords": [
                "fine-tuning", "transfer learning", "few-shot", "zero-shot",
                "rlhf", "reinforcement learning", "supervised", "unsupervised",
                "self-supervised", "contrastive learning", "distillation",
                "quantization", "pruning", "compression", "sparsity"
            ],
            "benchmark_keywords": [
                "benchmark", "evaluation", "dataset", "metric", "score",
                "accuracy", "perplexity", "bleu", "rouge", "f1"
            ]
        },
        "平台底座": {
            "framework_keywords": [
                "framework", "library", "toolkit", "sdk", "api",
                "pytorch", "tensorflow", "jax", "keras", "hugging face",
                "onnx", "triton", "mlx", "executorch"
            ],
            "hardware_keywords": [
                "gpu", "tpu", "npu", "cuda", "metal", "vulkan",
                "chip", "processor", "accelerator", "hardware",
                "nvidia", "amd", "intel", "apple silicon", "arm"
            ],
            "optimization_keywords": [
                "inference", "runtime", "engine", "optimization",
                "compilation", "kernel", "operator", "fusion",
                "memory", "latency", "throughput", "performance"
            ]
        },
        "行业动态": {
            "company_keywords": [
                "apple", "google", "microsoft", "amazon", "meta",
                "samsung", "huawei", "xiaomi", "oppo", "vivo"
            ],
            "product_keywords": [
                "product", "device", "smartphone", "iphone", "android",
                "release", "launch", "announcement", "unveil",
                "feature", "update", "version", "upgrade"
            ],
            "consumer_keywords": [
                "user", "consumer", "customer", "experience",
                "interface", "app", "application", "service"
            ]
        },
        "大V访谈": {
            "interview_keywords": [
                "interview", "conversation", "podcast", "talk", "discussion",
                "qa", "q&a", "ama", "fireside", "chat", "dialogue"
            ],
            "expert_keywords": [
                "hinton", "lecun", "bengio", "altman", "hassabis",
                "sutskever", "karpathy", "ng", "dean", "chollet",
                "chief", "ceo", "founder", "professor", "researcher"
            ],
            "opinion_keywords": [
                "opinion", "perspective", "view", "insight", "thought",
                "believe", "think", "predict", "future", "trend"
            ]
        }
    }
    
    @classmethod
    def analyze_content(cls, article_data: dict) -> dict:
        """深度分析文章内容"""
        title = article_data.get('title', '')
        content = article_data.get('content', '')
        entities = article_data.get('entities', {})
        metadata = article_data.get('metadata', {})
        
        text = (title + ' ' + content).lower()
        
        # 1. 分类
        category = cls._classify(text, entities, metadata)
        
        # 2. 提取关键主题
        themes = cls._extract_themes(text, category)
        
        # 3. 识别创新点
        innovations = cls._identify_innovations(text, entities, category)
        
        # 4. 分析技术深度
        tech_depth = cls._analyze_technical_depth(text, entities)
        
        # 5. 提取数据指标
        metrics = cls._extract_metrics(text, entities)
        
        # 6. 评估影响力
        impact = cls._assess_impact(text, entities, metadata, category)
        
        return {
            'category': category,
            'themes': themes,
            'innovations': innovations,
            'tech_depth': tech_depth,
            'metrics': metrics,
            'impact': impact,
            'entities': entities,
            'metadata': metadata
        }
    
    @classmethod
    def _classify(cls, text: str, entities: dict, metadata: dict) -> str:
        """智能分类"""
        scores = {}
        
        for category, features in cls.CATEGORY_FEATURES.items():
            score = 0
            
            # 计算各类关键词匹配度
            for feature_type, keywords in features.items():
                matches = sum(1 for kw in keywords if kw in text)
                
                # 不同特征类型权重不同
                if 'core' in feature_type:
                    score += matches * 3
                elif 'technique' in feature_type or 'framework' in feature_type:
                    score += matches * 2
                else:
                    score += matches
            
            scores[category] = score
        
        # 特殊规则 - 优先判断
        source = metadata.get('source', '').lower()
        url = metadata.get('url', '').lower()
        title_lower = metadata.get('title', '').lower()
        
        # 大V访谈识别 - Twitter/X链接或知名人物访谈
        if 'x.com' in url or 'twitter.com' in url:
            if any(name in text.lower() for name in ['karpathy', 'hinton', 'lecun', 'bengio', 'ng', 'altman']):
                return '大V访谈'
        
        if entities.get('people') and any(word in text for word in ['interview', 'podcast', 'conversation', 'talk', 'says', 'discusses']):
            return '大V访谈'
        
        if any(name in text.lower() for name in ['hinton', 'lecun', 'bengio', 'karpathy', 'altman']):
            if any(word in text for word in ['interview', 'conversation', 'talk', 'podcast']):
                return '大V访谈'
        
        # 平台底座识别 - 框架/工具/硬件/基础设施
        platform_keywords = [
            'pytorch', 'tensorflow', 'framework', 'library', 'toolkit',
            'gpu', 'hardware', 'chip', 'processor', 'accelerator',
            'infrastructure', 'deployment', 'serving', 'inference engine',
            'compiler', 'runtime', 'api', 'sdk', 'cloud platform',
            'h100', 'h200', 'a100', 'cuda', 'nvlink'
        ]
        if any(kw in text.lower() for kw in platform_keywords):
            # 进一步判断是否真的是平台底座
            if any(kw in text.lower() for kw in ['release', 'update', 'launch', 'version', 'announced']):
                scores['平台底座'] += 15
            if 'nvidia' in text.lower() or 'amd' in text.lower() or 'intel' in text.lower():
                scores['平台底座'] += 10
        
        # arXiv论文 - 通常是模型算法
        if 'arxiv' in source or 'arxiv.org' in url:
            scores['模型算法'] += 10
        
        # 检查标题和内容的算法特征
        algo_patterns = ['algorithm', 'model', 'learning', 'training', 'architecture', 'neural']
        if any(pattern in title_lower or pattern in text.lower() for pattern in algo_patterns):
            scores['模型算法'] += 5
        
        # 公司产品发布 - 行业动态
        companies = ['apple', 'google', 'microsoft', 'meta', 'amazon', 'openai', 'anthropic']
        if any(comp in source or comp in text.lower() for comp in companies):
            if any(word in text.lower() for word in ['announces', 'launches', 'releases', 'unveils', 'introduces']):
                if 'product' in text.lower() or 'app' in text.lower() or 'service' in text.lower():
                    scores['行业动态'] += 10
        
        # 返回得分最高的分类
        return max(scores, key=scores.get) if max(scores.values()) > 0 else '行业动态'
    
    @classmethod
    def _extract_themes(cls, text: str, category: str) -> list:
        """提取关键主题"""
        themes = []
        
        # 端侧AI主题
        if any(kw in text for kw in ['on-device', 'edge', 'mobile', 'local']):
            if any(kw in text for kw in ['inference', 'deployment', 'optimization']):
                themes.append('端侧AI部署')
        
        # 模型优化主题
        if any(kw in text for kw in ['quantization', 'compression', 'pruning']):
            themes.append('模型压缩优化')
        
        # 训练技术主题
        if any(kw in text for kw in ['training', 'fine-tuning', 'rlhf', 'dpo']):
            themes.append('模型训练技术')
        
        # 硬件加速主题
        if any(kw in text for kw in ['gpu', 'tpu', 'npu', 'accelerator']):
            if any(kw in text for kw in ['optimization', 'performance', 'speed']):
                themes.append('硬件加速')
        
        # 开源工具主题
        if any(kw in text for kw in ['open source', 'github', 'release']):
            if any(kw in text for kw in ['framework', 'library', 'tool']):
                themes.append('开源工具')
        
        # 行业应用主题
        if category == '行业动态':
            if any(kw in text for kw in ['product', 'launch', 'release']):
                themes.append('产品发布')
        
        return themes[:3]  # 最多3个主题
    
    @classmethod
    def _identify_innovations(cls, text: str, entities: dict, category: str) -> list:
        """识别创新点"""
        innovations = []
        
        # 架构创新
        if any(kw in text for kw in ['novel', 'new architecture', 'innovative']):
            if entities.get('technologies'):
                innovations.append(f"提出新型{entities['technologies'][0]}架构")
        
        # 性能提升
        speed_matches = re.findall(r'(\d+\.?\d*)x\s+faster', text)
        if speed_matches:
            innovations.append(f"性能提升{speed_matches[0]}倍")
        
        accuracy_matches = re.findall(r'(\d+\.?\d*)%\s+(?:accuracy|improvement)', text)
        if accuracy_matches:
            innovations.append(f"精度提升{accuracy_matches[0]}%")
        
        # 效率创新
        if any(kw in text for kw in ['efficient', 'lightweight', 'compact']):
            if any(kw in text for kw in ['inference', 'deployment', 'edge']):
                innovations.append("实现高效推理部署")
        
        # 方法论创新
        if any(kw in text for kw in ['approach', 'method', 'technique']):
            if any(kw in text for kw in ['novel', 'new', 'innovative']):
                innovations.append("创新的方法论")
        
        return innovations[:3]
    
    @classmethod
    def _analyze_technical_depth(cls, text: str, entities: dict) -> str:
        """分析技术深度"""
        depth_score = 0
        
        # 技术关键词密度
        tech_kws = len(entities.get('technologies', []))
        depth_score += tech_kws * 2
        
        # 专业术语
        advanced_terms = [
            'architecture', 'optimization', 'algorithm', 'implementation',
            'kernel', 'operator', 'compilation', 'inference engine'
        ]
        depth_score += sum(1 for term in advanced_terms if term in text)
        
        # 数学/技术细节
        if any(kw in text for kw in ['equation', 'formula', 'theorem', 'proof']):
            depth_score += 5
        
        # 实验验证
        if any(kw in text for kw in ['experiment', 'evaluation', 'benchmark']):
            depth_score += 3
        
        if depth_score >= 15:
            return "深度技术"
        elif depth_score >= 8:
            return "中等技术"
        else:
            return "应用介绍"
    
    @classmethod
    def _extract_metrics(cls, text: str, entities: dict) -> dict:
        """提取关键指标"""
        metrics = {
            'performance': [],
            'scale': [],
            'efficiency': []
        }
        
        # 性能指标
        perf_patterns = [
            (r'(\d+\.?\d*)x\s+faster', 'speed'),
            (r'(\d+\.?\d*)%\s+accuracy', 'accuracy'),
            (r'(\d+\.?\d*)%\s+improvement', 'improvement')
        ]
        
        for pattern, label in perf_patterns:
            matches = re.findall(pattern, text)
            if matches:
                metrics['performance'].append(f"{label}: {matches[0]}")
        
        # 规模指标
        scale_patterns = [
            (r'(\d+\.?\d*)\s*(?:billion|B)\s+parameters', 'params'),
            (r'(\d+\.?\d*)\s*(?:million|M)\s+parameters', 'params'),
            (r'(\d+\.?\d*)\s*(?:trillion|T)\s+tokens', 'tokens')
        ]
        
        for pattern, label in scale_patterns:
            matches = re.findall(pattern, text)
            if matches:
                metrics['scale'].append(f"{label}: {matches[0]}")
        
        return metrics
    
    @classmethod
    def _assess_impact(cls, text: str, entities: dict, metadata: dict, category: str) -> str:
        """评估影响力"""
        impact_score = 0
        
        # 来源权威度
        source = metadata.get('source', '').lower()
        if source in ['arxiv', 'openai', 'anthropic', 'deepmind']:
            impact_score += 10
        elif source in ['stanford', 'mit', 'berkeley']:
            impact_score += 8
        
        # 知名人物/机构
        if entities.get('people'):
            impact_score += len(entities['people']) * 3
        
        if entities.get('companies'):
            impact_score += len(entities['companies']) * 2
        
        # 技术前沿性
        if any(kw in text for kw in ['breakthrough', 'novel', 'first', 'pioneer']):
            impact_score += 5
        
        # 实用性
        if any(kw in text for kw in ['open source', 'available', 'release']):
            impact_score += 4
        
        if impact_score >= 20:
            return "重大突破"
        elif impact_score >= 12:
            return "显著进展"
        else:
            return "渐进改进"

# ================= 📝 智能模板生成引擎 =================

class TemplateEngine:
    """基于分析结果生成高质量文本"""
    
    @staticmethod
    def generate_chinese_title(analysis: dict, original_title: str) -> str:
        """生成精炼的中文标题 - 增加多样性"""
        try:
            category = analysis['category']
            themes = analysis['themes']
            innovations = analysis['innovations']
            entities = analysis['entities']
            impact = analysis['impact']
            
            # 提取关键元素
            tech_list = entities.get('technologies', [])
            tech = tech_list[0] if tech_list else ''
            
            company_list = entities.get('companies', [])
            company = company_list[0] if company_list else ''
            
            models_list = entities.get('models', [])
            model = models_list[0] if models_list else ''
            
            # 从原标题提取关键信息（增强具体性）
            original_lower = original_title.lower() if original_title else ''
            
            # 提取数字/性能指标
            perf_match = re.search(r'(\d+\.?\d*)\s*(x|times|percent|%)', original_lower)
            perf_info = f"{perf_match.group(1)}{perf_match.group(2)}" if perf_match else None
            
            # 提取具体技术名词（大写开头的词）
            specific_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', original_title) if original_title else []
            specific_term = specific_terms[0] if specific_terms and len(specific_terms[0]) > 3 else None
            
            # 根据类别和主题生成标题
            if category == '模型算法':
                # 优先使用具体模型名（首字母大写）
                if model:
                    model_title = model.title() if len(model) > 3 else model.upper()
                    if perf_info:
                        return f"{model_title}模型性能提升{perf_info}"
                    elif '压缩' in ''.join(themes):
                        return f"{model_title}轻量化压缩方案"
                    else:
                        return f"{model_title}架构优化研究"
                
                # 使用具体技术名
                if specific_term and specific_term.lower() not in ['the', 'and', 'for']:
                    if '端侧' in ''.join(themes) or 'edge' in original_lower:
                        return f"{specific_term}端侧部署方法"
                    elif '压缩' in ''.join(themes):
                        return f"{specific_term}模型压缩技术"
                    else:
                        return f"{specific_term}算法创新"
                
                # 使用技术词
                if tech:
                    tech_upper = tech.upper() if len(tech) <= 6 else tech.title()
                    if '端侧' in ''.join(themes):
                        return f"{tech_upper}端侧优化突破"
                    elif '压缩' in ''.join(themes):
                        return f"基于{tech_upper}的模型压缩"
                    elif perf_info:
                        return f"{tech_upper}性能提升{perf_info}"
                    else:
                        return f"{tech_upper}新型架构设计"
                
                # 通用但有区分度的标题
                if '端侧' in ''.join(themes):
                    return f"边缘设备AI模型优化方案"
                elif '压缩' in ''.join(themes):
                    return f"神经网络压缩最新进展"
                elif '训练' in ''.join(themes):
                    return f"大模型训练效率优化"
                else:
                    return f"深度学习算法创新研究"
            
            elif category == '平台底座':
                if company:
                    comp_title = company.title()
                    if '硬件' in ''.join(themes):
                        return f"{comp_title}发布AI硬件加速方案"
                    elif '开源' in ''.join(themes):
                        return f"{comp_title}开源AI框架更新"
                    else:
                        return f"{comp_title}推理引擎性能突破"
                
                if tech:
                    if '开源' in ''.join(themes):
                        return f"{tech.title()}框架重大版本发布"
                    elif specific_term:
                        return f"{specific_term}推理加速工具链"
                    else:
                        return f"{tech.title()}运行时优化升级"
                
                # 通用标题
                if '硬件' in ''.join(themes):
                    return f"新一代AI加速硬件架构"
                elif '开源' in ''.join(themes):
                    return f"AI基础框架开源生态建设"
                else:
                    return f"高性能推理引擎技术进展"
            
            elif category == '行业动态':
                if company:
                    comp = company.title()
                    if '产品' in ''.join(themes):
                        if specific_term:
                            return f"{comp}推出{specific_term}智能产品"
                        else:
                            return f"{comp}发布AI增强新品"
                    else:
                        if specific_term:
                            return f"{comp}{specific_term}战略布局"
                        else:
                            return f"{comp}深化AI产业投入"
                else:
                    if specific_term:
                        return f"科技巨头布局{specific_term}领域"
                    else:
                        return "AI产业竞争格局分析"
            
            else:  # 大V访谈
                if entities.get('people'):
                    names = entities['people']
                    expert = names[0].title()
                    
                    # 提取主题关键词
                    if 'safety' in original_lower or 'safe' in original_lower:
                        return f"{expert}论AI安全与伦理"
                    elif 'future' in original_lower or 'trend' in original_lower:
                        return f"{expert}展望AI未来趋势"
                    elif 'research' in original_lower:
                        return f"{expert}分享最新研究成果"
                    else:
                        return f"{expert}谈AI技术发展"
                else:
                    if specific_term:
                        return f"行业专家解读{specific_term}前景"
                    else:
                        return "AI领域资深专家访谈"
                        
        except Exception as e:
            print(f"  ⚠️ 标题生成异常: {e}, 使用通用标题")
            # 返回通用标题
            category = analysis.get('category', '未知')
            return f"AI{category}最新动态"
    
    @staticmethod
    def generate_key_points(analysis: dict, article_data: dict) -> dict:
        """生成三要点摘要"""
        category = analysis['category']
        themes = analysis['themes']
        innovations = analysis['innovations']
        metrics = analysis['metrics']
        impact = analysis['impact']
        tech_depth = analysis['tech_depth']
        entities = analysis['entities']
        
        content_sample = article_data.get('content', '')[:1000]
        
        # 内容简述
        brief = TemplateEngine._generate_content_brief(
            category, themes, entities, content_sample
        )
        
        # 关键创新
        innovation = TemplateEngine._generate_key_innovation(
            innovations, metrics, tech_depth, entities
        )
        
        # 洞察启示
        insight = TemplateEngine._generate_insight(
            impact, category, themes, entities
        )
        
        return {
            'content_brief': brief,
            'key_innovation': innovation,
            'insight': insight
        }
    
    @staticmethod
    def _generate_content_brief(category: str, themes: list, entities: dict, sample: str) -> str:
        """生成内容简述"""
        tech_list = entities.get('technologies', [])
        tech = tech_list[0] if tech_list else 'AI'
        
        company_list = entities.get('companies', [])
        company = company_list[0] if company_list else None
        
        models_list = entities.get('models', [])
        models = models_list[0] if models_list else None
        
        if category == '模型算法':
            if '端侧AI部署' in themes:
                return f"本文介绍了一种针对端侧设备的{tech}模型优化方案，通过创新的压缩和加速技术，实现了在资源受限环境下的高效推理。研究团队详细阐述了算法设计思路、实现细节以及在多个基准测试上的性能表现。"
            elif '模型压缩优化' in themes:
                return f"研究团队提出了一种新型的{tech}模型压缩方法，通过结合量化、剪枝和知识蒸馏技术，在保持精度的同时显著降低了模型规模和计算复杂度。实验结果表明该方法在多个任务上都取得了优异的性能。"
            elif models:
                return f"本文详细介绍了{models}模型的最新改进，包括架构优化、训练策略创新以及推理加速方案。研究团队通过大量实验验证了所提方法的有效性，为大规模模型的实用化部署提供了新思路。"
            else:
                return "本文深入探讨了AI模型算法的最新进展，从理论基础到工程实践，全面分析了当前技术路线的优势与挑战。研究工作涵盖了模型设计、训练优化和性能评估等多个方面，为领域发展提供了宝贵的参考。"
        
        elif category == '平台底座':
            if '硬件加速' in themes:
                comp = company if company else '研究团队'
                return f"{comp}发布了新一代AI加速解决方案，通过软硬件协同优化，大幅提升了模型推理性能。该方案支持多种主流框架，为开发者提供了统一的接口和高效的运行时环境，显著降低了AI应用的部署门槛。"
            elif '开源工具' in themes:
                fw = tech
                return f"{fw}发布重大更新，新增多项实用功能和性能优化。此次更新重点改进了推理引擎的效率，优化了内存管理机制，并扩展了对新型硬件的支持。开源社区对此反响热烈，认为这将加速AI技术的落地应用。"
            else:
                return "本文介绍了AI基础设施领域的最新进展，涵盖框架优化、硬件适配和工具链完善等多个维度。通过技术创新和工程实践的结合，为构建高效的AI系统提供了坚实的底层支撑。"
        
        elif category == '行业动态':
            comp = company.title() if company else '科技企业'
            if '产品发布' in themes:
                return f"{comp}正式发布搭载先进AI功能的全新产品，为用户带来智能化升级体验。产品集成了最新的机器学习技术，实现了本地化处理与云端协同的完美结合，在保护隐私的同时提供强大的AI能力。"
            else:
                return f"{comp}公布了AI领域的最新战略布局，展示了在模型研发、应用落地和生态建设方面的深入探索。此举标志着企业在AI赛道的持续投入和长远规划，有望推动行业整体发展。"
        
        else:  # 大V访谈
            if entities.get('people'):
                names = entities['people']
                expert = names[0].title()
                return f"业界知名专家{expert}在访谈中分享了对AI技术发展趋势的独到见解，探讨了当前面临的技术挑战、伦理考量以及未来机遇。专家强调了安全性和可控性在AI研究中的核心地位，为行业发展指明了方向。"
            else:
                return "本次访谈汇集了AI领域多位专家的深度对话，围绕技术前沿、产业应用和社会影响展开讨论。嘉宾们分享了各自的研究心得和实践经验，为理解AI发展脉络提供了多元视角。"
    
    @staticmethod
    def _generate_key_innovation(innovations: list, metrics: dict, tech_depth: str, entities: dict) -> str:
        """生成关键创新"""
        tech_list = entities.get('technologies', [])
        tech = tech_list[0] if tech_list else None
        
        innovation_text = ""
        
        # 如果有明确的创新点
        if innovations:
            innovation_text = f"核心创新在于{innovations[0]}，"
            
            if metrics['performance']:
                perf = metrics['performance'][0]
                innovation_text += f"实测显示{perf}，"
            
            if tech:
                innovation_text += f"该方案基于{tech}技术栈，"
            
            innovation_text += "通过系统性的优化策略，实现了性能与效率的最优平衡。"
        
        # 否则根据技术深度生成
        elif tech_depth == "深度技术":
            innovation_text = "研究工作在理论层面取得突破，提出了全新的技术范式。通过严谨的数学推导和大量实验验证，证明了方法的有效性和普适性，为后续研究奠定了坚实基础。"
        
        elif tech_depth == "中等技术":
            innovation_text = "团队采用了创新的工程实践方案，巧妙地结合了多种技术手段。通过精细的参数调优和系统优化，在保持实用性的同时提升了整体性能表现。"
        
        else:
            innovation_text = "方案注重实际应用价值，通过用户友好的设计和稳定的性能表现，降低了技术使用门槛。开放的接口和完善的文档支持，便于开发者快速集成和部署。"
        
        return innovation_text
    
    @staticmethod
    def _generate_insight(impact: str, category: str, themes: list, entities: dict) -> str:
        """生成洞察启示"""
        
        if impact == "重大突破":
            insight = "这项工作代表了领域内的重大突破，其影响不仅限于技术本身，更可能引发研究范式的转变。"
        elif impact == "显著进展":
            insight = "该研究标志着领域内的显著进展，为解决长期存在的技术挑战提供了新思路。"
        else:
            insight = "这项工作体现了技术的持续迭代和优化，虽属渐进改进，但在实用价值上不容小觑。"
        
        # 根据分类添加特定洞察
        if category == '模型算法':
            if '端侧AI部署' in themes:
                insight += "端侧AI的成熟将推动AI技术从云端走向边缘，实现更广泛的应用场景覆盖，同时在隐私保护和响应速度上带来质的飞跃。"
            else:
                insight += "算法创新始终是AI进步的核心驱动力，优秀的模型设计不仅能提升性能，更能启发新的研究方向，推动整个领域向前发展。"
        
        elif category == '平台底座':
            insight += "底层基础设施的完善对AI生态至关重要，高效的工具链和硬件支持将大幅降低AI应用的开发成本，加速技术的普及和落地。"
        
        elif category == '行业动态':
            insight += "科技巨头的战略布局往往预示着行业趋势，其在AI领域的持续投入和创新实践，将深刻影响技术发展方向和市场格局演变。"
        
        else:  # 大V访谈
            insight += "顶尖专家的思考为行业提供了宝贵的导向，他们的洞见帮助我们在技术狂飙突进的同时，保持对AI安全、伦理和社会影响的理性审视。"
        
        return insight
    
    @staticmethod
    def generate_detailed_explanation(analysis: dict, article_data: dict, key_points: dict) -> str:
        """生成600-1000字详细说明"""
        category = analysis['category']
        themes = analysis['themes']
        tech_depth = analysis['tech_depth']
        entities = analysis['entities']
        impact = analysis['impact']
        content = article_data.get('content', '')[:5000]
        url = article_data.get('url', '')
        
        # 提取关键段落
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 100][:8]
        
        # 构建详细说明
        sections = []
        
        # 第一部分：研究背景和动机（150-200字）
        sections.append(TemplateEngine._generate_background_section(category, themes, entities))
        
        # 第二部分：技术方案详解（250-350字）
        sections.append(TemplateEngine._generate_technical_section(
            category, tech_depth, entities, paragraphs
        ))
        
        # 第三部分：实验验证和效果（150-200字）
        sections.append(TemplateEngine._generate_results_section(
            analysis, paragraphs
        ))
        
        # 第四部分：行业影响和未来展望（150-200字）
        sections.append(TemplateEngine._generate_impact_section(
            impact, category, themes, entities
        ))
        
        full_text = '。'.join(sections) + f"。原文链接：{url}"
        
        return full_text
    
    @staticmethod
    def _generate_background_section(category: str, themes: list, entities: dict) -> str:
        """生成背景部分"""
        tech_list = entities.get('technologies', [])
        tech = tech_list[0] if tech_list else 'AI'
        
        if category == '模型算法':
            if '端侧AI部署' in themes:
                return f"随着{tech}模型规模的不断增长，如何在资源受限的端侧设备上实现高效推理成为亟待解决的关键问题。传统的云端推理方案虽然性能强大，但在隐私保护、响应延迟和网络依赖等方面存在明显短板。本研究正是在这一背景下展开，旨在通过创新的优化技术，突破端侧部署的性能瓶颈"
            elif '模型压缩优化' in themes:
                return f"大规模{tech}模型在多项任务上展现出卓越性能，但庞大的参数量和计算开销限制了其实际应用。模型压缩技术作为解决这一矛盾的关键手段，一直是学术界和工业界的研究热点。本工作提出了一种系统性的压缩方案，在保持模型能力的前提下大幅降低资源消耗"
            else:
                return f"AI领域的快速发展离不开算法层面的持续创新。近年来，{tech}技术在理论和实践两个层面都取得了重要突破，为复杂任务的解决提供了新的可能。本研究深入探索了模型设计的核心问题，通过巧妙的架构创新和训练策略优化，推动了技术边界的进一步拓展"
        
        elif category == '平台底座':
            return f"AI应用的大规模落地离不开高效稳定的基础设施支撑。当前，{tech}生态虽已相对成熟，但在性能优化、硬件适配和开发体验等方面仍有提升空间。本次更新聚焦于底层引擎的全面升级，通过技术创新和工程优化的深度结合，为开发者提供更加强大的工具链"
        
        elif category == '行业动态':
            comp = entities.get('companies', ['科技企业'])[0].title()
            return f"在AI技术快速演进的大背景下，{comp}作为行业领军企业，始终保持着对前沿技术的敏锐洞察和战略布局。此次动作不仅体现了企业的技术实力，更反映出其对AI未来发展趋势的深刻理解。通过产品创新和生态建设的双轮驱动，企业正在构建面向智能时代的全新竞争优势"
        
        else:  # 大V访谈
            return "AI技术的蓬勃发展引发了学术界和产业界的广泛讨论，如何在追求技术进步的同时兼顾安全性和社会责任，成为当下最为重要的议题之一。本次访谈邀请到了领域内的资深专家，围绕技术趋势、挑战应对和未来展望等核心话题展开深入对话，为理解AI发展脉络提供了宝贵视角"
    
    @staticmethod
    def _generate_technical_section(category: str, tech_depth: str, entities: dict, paragraphs: list) -> str:
        """生成技术方案部分"""
        tech_list = entities.get('technologies', [])
        tech = tech_list[0] if tech_list else None
        
        if tech_depth == "深度技术":
            base = "从技术实现角度看，该方案采用了多层次的优化策略。在算法设计上，研究团队提出了创新的数学模型，通过理论分析证明了方法的收敛性和最优性"
            
            if tech:
                base += f"。具体而言，核心技术基于{tech}框架，"
            else:
                base += "。"
            
            base += "通过精心设计的损失函数和正则化项，在保证模型表达能力的同时有效控制了过拟合风险。在工程实现层面，团队针对关键计算模块进行了深度优化，通过算子融合、内存池化和并行调度等手段，显著提升了运行效率。此外，完善的工具链和可视化界面降低了使用门槛，便于研究者快速复现和扩展"
        
        elif tech_depth == "中等技术":
            base = "方案的核心在于将多种成熟技术进行有机整合，形成系统性的解决方案。在模型层面，采用了经过验证的架构设计，并针对特定任务进行了定制化调整"
            
            if '端侧' in ''.join(paragraphs[:2]).lower():
                base += "。针对端侧部署场景，团队重点优化了模型的推理流程，通过量化和剪枝技术大幅降低了计算复杂度和内存占用"
            
            base += "。在训练策略上，结合了数据增强、学习率调度和早停等常用技巧，确保模型能够稳定收敛到理想状态。工程实现注重代码的模块化和可维护性，清晰的接口设计和详实的文档支持，使得方案易于理解和部署"
        
        else:  # 应用介绍
            base = "该方案注重实用性和易用性，通过简洁的设计理念和稳定的性能表现，为用户提供了开箱即用的解决方案。系统采用了主流的技术栈，保证了良好的兼容性和可扩展性"
            
            if category == '行业动态':
                base += "。产品集成了经过充分验证的AI能力，通过友好的交互界面和流畅的使用体验，让技术真正服务于日常应用"
            
            base += "。在部署方面，提供了详细的配置指南和示例代码，开发者可以快速完成集成工作。同时，活跃的社区支持和持续的版本迭代，确保了方案的长期可用性"
        
        return base
    
    @staticmethod
    def _generate_results_section(analysis: dict, paragraphs: list) -> str:
        """生成实验结果部分"""
        metrics = analysis['metrics']
        category = analysis['category']
        
        base = "为验证方案的有效性，研究团队开展了全面的实验评估"
        
        if metrics['performance']:
            perf_str = '、'.join(metrics['performance'][:2])
            base += f"。在标准基准测试上，方案取得了显著的性能提升，具体表现为{perf_str}"
        
        if metrics['scale']:
            scale_str = metrics['scale'][0]
            base += f"。模型规模方面，{scale_str}，"
        
        if category == '模型算法':
            base += "。消融实验进一步揭示了各个组件的贡献度，证明了设计选择的合理性。与现有主流方法的对比显示，本方案在多个评估维度上都具有明显优势，特别是在效率和精度的权衡上找到了更优的解决方案"
        elif category == '平台底座':
            base += "。性能测试涵盖了多种硬件平台和使用场景，结果表明系统具有良好的通用性和稳定性。用户反馈积极，认为新版本在易用性和功能完整性上都有质的飞跃"
        else:
            base += "。实际应用效果超出预期，在真实业务场景中展现出强大的适应能力和鲁棒性"
        
        return base
    
    @staticmethod
    def _generate_impact_section(impact: str, category: str, themes: list, entities: dict) -> str:
        """生成影响展望部分"""
        
        if impact == "重大突破":
            base = "这项工作的意义远超技术本身，可能引发领域研究范式的深刻变革"
        elif impact == "显著进展":
            base = "该研究为解决长期存在的技术难题提供了新的视角和方法"
        else:
            base = "尽管属于渐进式改进，但工作在实用性和可落地性上具有重要价值"
        
        if category == '模型算法':
            base += "。从行业影响看，优秀的算法创新往往能够催生新的应用场景，推动AI技术向更广阔的领域拓展。特别是在当前算力成本高企、效率需求迫切的背景下，这类聚焦于优化和压缩的研究工作显得尤为重要"
        
        elif category == '平台底座':
            base += "。基础设施的完善对整个AI生态具有乘数效应，优秀的框架和工具能够降低技术门槛，让更多开发者参与到AI应用的创新中来。这种底层能力的提升，最终将转化为上层应用的繁荣"
        
        elif category == '行业动态':
            comp_list = entities.get('companies', [])
            comp = comp_list[0].title() if comp_list else '龙头企业'
            base += f"。{comp}的战略选择往往具有风向标意义，其在AI领域的布局和投入力度，反映了行业对技术未来的集体判断。随着越来越多的企业加入竞争，AI技术的进步将持续加速"
        
        else:  # 大V访谈
            base += "。顶尖专家的洞见帮助我们站在更高的视角审视技术发展，他们对安全性、可控性和社会影响的关注，提醒整个行业在追求进步的同时必须保持警醒和理性"
        
        base += "。展望未来，AI技术将继续沿着高效化、实用化和安全化的方向演进，而每一项扎实的研究工作都是通向这一目标的重要阶梯"
        
        return base

# ================= 📧 EML 生成器 =================

class EMLGenerator:
    """EML邮件报告生成器"""
    
    @staticmethod
    def generate_eml(articles: list, date_str: str) -> str:
        """生成完整EML报告"""
        
        # ========== 🚨 标题去重：严格过滤相似标题 ==========
        print("\n🔍 检查标题相似度...")
        unique_articles = []
        seen_titles = []
        
        for article in articles:
            title = article.get('chinese_title', '')
            is_duplicate = False
            
            # 检查与已有标题的相似度
            for seen_title in seen_titles:
                # 去除空格和标点后比较
                clean_title = ''.join(c.lower() for c in title if c.isalnum())
                clean_seen = ''.join(c.lower() for c in seen_title if c.isalnum())
                
                similarity = SequenceMatcher(None, clean_title, clean_seen).ratio()
                
                # 相似度超过60%视为重复（更严格）
                # 或者核心词完全相同也视为重复
                core_words_title = set([w for w in title.split() if len(w) > 1])
                core_words_seen = set([w for w in seen_title.split() if len(w) > 1])
                word_overlap = len(core_words_title & core_words_seen) / max(len(core_words_title), len(core_words_seen), 1)
                
                if similarity > 0.60 or word_overlap > 0.7:
                    print(f"  ❌ 相似标题 (文字{similarity:.0%}/词汇{word_overlap:.0%}): '{title}'")
                    print(f"     与已有标题: '{seen_title}'")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
                seen_titles.append(title)
                print(f"  ✅ 保留: '{title}'")
        
        print(f"\n📊 去重结果: {len(articles)} → {len(unique_articles)} 篇")
        
        if not unique_articles:
            print("⚠️ 警告: 去重后没有文章了！")
            return ""
        
        # 使用去重后的文章列表
        articles = unique_articles
        # ========== 去重结束 ==========
        
        # 按分类排序
        category_order = ["模型算法", "平台底座", "行业动态", "大V访谈"]
        articles_sorted = sorted(
            articles,
            key=lambda x: category_order.index(x.get('category', '行业动态'))
        )
        
        # 生成目录表格HTML
        toc_html = EMLGenerator._generate_toc_table(articles_sorted)
        
        # 生成详情HTML
        details_html = EMLGenerator._generate_details_section(articles_sorted)
        
        # 组装完整HTML
        html_body = EMLGenerator._assemble_html(toc_html, details_html, date_str)
        
        # 生成EML头部
        subject = f"北美AI洞察快讯~{date_str}"
        subject_encoded = base64.b64encode(subject.encode('utf-8')).decode('ascii')
        
        now = datetime.now()
        # 格式化日期时间（RFC 2822格式）
        date_formatted = now.strftime("%a, %d %b %Y %H:%M:%S") + " -0500"  # EST时区
        
        eml_content = f"""From: "北美AI洞察快讯" <insight@moore-institute.ca>
To: subscriber@example.com
Subject: =?UTF-8?B?{subject_encoded}?=
Date: {date_formatted}
MIME-Version: 1.0
Content-Type: text/html; charset=UTF-8
Content-Transfer-Encoding: 8bit

{html_body}"""
        
        return eml_content
    
    @staticmethod
    def _generate_toc_table(articles: list) -> str:
        """生成目录表格"""
        rows = []
        category_counts = {}
        
        # 统计每个分类的文章数
        for article in articles:
            cat = article.get('category', '行业动态')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        current_category = None
        article_index = 1
        
        for article in articles:
            category = article.get('category', '行业动态')
            title = article.get('chinese_title', '未命名')
            author = article.get('author', '未知')
            
            if category != current_category:
                # 新分类，需要合并单元格
                rowspan = category_counts[category]
                rows.append(f'''
                <tr>
                    <td class="category-cell" rowspan="{rowspan}">{category}</td>
                    <td><a href="#article-{article_index}"><strong>{title}</strong></a></td>
                    <td>{author}</td>
                </tr>''')
                current_category = category
            else:
                # 同一分类
                rows.append(f'''
                <tr>
                    <td><a href="#article-{article_index}"><strong>{title}</strong></a></td>
                    <td>{author}</td>
                </tr>''')
            
            article_index += 1
        
        return '\n'.join(rows)
    
    @staticmethod
    def _generate_details_section(articles: list) -> str:
        """生成详情部分"""
        details = []
        article_index = 1
        
        for article in articles:
            title = article.get('chinese_title', '未命名')
            category = article.get('category', '行业动态')
            author = article.get('author', '未知')
            key_points = article.get('key_points', {})
            detail = article.get('detailed_explanation', '')
            
            # 锚点必须在article-box之前！
            article_html = f'''
        <a id="article-{article_index}" name="article-{article_index}"></a>
        <div class="article-box">
            <div class="article-title">{title}</div>
            <div class="article-meta">分类: {category} | 机构: {author}</div>
            
            <div class="summary-section">
                <div class="summary-title">📌 内容简述</div>
                <div class="summary-content">{key_points.get('content_brief', '')}</div>
            </div>
            
            <div class="summary-section">
                <div class="summary-title">💡 关键创新</div>
                <div class="summary-content">{key_points.get('key_innovation', '')}</div>
            </div>
            
            <div class="summary-section">
                <div class="summary-title">🎯 洞察启示</div>
                <div class="summary-content">{key_points.get('insight', '')}</div>
            </div>
            
            <div class="detail-section">
                <strong>详细说明：</strong><br><br>
                {detail}
            </div>
            
            <a href="#toc" class="back-button">↑ 返回目录</a>
        </div>'''
            
            details.append(article_html)
            article_index += 1
        
        return '\n'.join(details)
    
    @staticmethod
    def _assemble_html(toc_html: str, details_html: str, date_str: str) -> str:
        """组装完整HTML"""
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif;
            line-height: 1.8;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #667eea 100%);
            color: white;
            padding: 40px 30px;
            border-radius: 12px 12px 0 0;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
            font-weight: 700;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            color: #000000;
        }}
        .header p {{
            margin: 15px 0 0 0;
            opacity: 0.95;
            font-size: 16px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }}
        .container {{
            background: white;
            padding: 40px 35px;
            border-radius: 0 0 12px 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        }}
        h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 12px;
            margin-top: 0;
            margin-bottom: 25px;
            font-size: 26px;
            font-weight: 700;
        }}
        .toc-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-radius: 8px;
            overflow: hidden;
        }}
        .toc-table th {{
            background: linear-gradient(135deg, #667eea 0%, #667eea 100%);
            color: white;
            padding: 16px 14px;
            text-align: left;
            font-weight: 700;
            font-size: 15px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }}
        .toc-table td {{
            padding: 14px;
            border-bottom: 1px solid #e8e8e8;
            font-size: 14px;
        }}
        .toc-table tr:last-child td {{
            border-bottom: none;
        }}
        .toc-table tr:hover {{
            background-color: #f8f9fe;
        }}
        .toc-table a {{
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
            transition: color 0.2s;
        }}
        .toc-table a:hover {{
            color: #5568d3;
            text-decoration: underline;
        }}
        .category-cell {{
            background-color: #f8f9fe;
            font-weight: 700;
            color: #667eea;
            vertical-align: middle;
            text-align: center;
        }}
        .article-box {{
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 30px;
            margin: 30px 0;
            background: linear-gradient(to bottom, #f8f9fe 0%, #ffffff 100%);
            box-shadow: 0 2px 8px rgba(217, 119, 87, 0.08);
        }}
        .article-title {{
            font-size: 24px;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 12px;
            line-height: 1.4;
        }}
        .article-meta {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }}
        .summary-section {{
            background: white;
            padding: 18px;
            border-radius: 8px;
            margin: 18px 0;
            border-left: 4px solid #667eea;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .summary-title {{
            font-weight: 700;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        .summary-content {{
            color: #555;
            line-height: 1.9;
            font-size: 15px;
            text-align: justify;
        }}
        .detail-section {{
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fe;
            border-radius: 8px;
            text-align: justify;
            line-height: 1.9;
            color: #444;
            font-size: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .detail-section strong {{
            color: #667eea;
            font-size: 16px;
        }}
        .back-button {{
            display: inline-block;
            margin-top: 20px;
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #667eea 100%);
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 2px 4px rgba(204, 85, 51, 0.3);
        }}
        .back-button:hover {{
            background: linear-gradient(135deg, #667eea 0%, #5568d3 100%);
            box-shadow: 0 4px 8px rgba(204, 85, 51, 0.4);
            transform: translateY(-1px);
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 25px;
            border-top: 2px solid #e0e0e0;
            color: #7f8c8d;
            font-size: 14px;
        }}
        .footer p {{
            margin: 8px 0;
        }}
        .footer strong {{
            color: #667eea;
            font-size: 16px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 北美AI洞察快讯</h1>
        <p>加拿大研究院摩尔研究所 | {date_str}</p>
    </div>
    
    <div class="container">
        <h2 id="toc">📋 内容速览</h2>
        
        <table class="toc-table">
            <thead>
                <tr>
                    <th width="15%">分类</th>
                    <th width="60%">标题</th>
                    <th width="25%">机构</th>
                </tr>
            </thead>
            <tbody>
                {toc_html}
            </tbody>
        </table>
        
        <h2>📰 内容详情</h2>
        
        {details_html}
        
        <div class="footer">
            <p><strong>加拿大研究院摩尔研究所</strong></p>
            <p>Moore Institute of Canadian Research Academy</p>
            <p>专注于AI技术洞察与前沿研究</p>
            <p style="margin-top: 15px; font-size: 12px; color: #999;">
                本报告由智能分析系统自动生成 | 内容仅供参考
            </p>
        </div>
    </div>
</body>
</html>'''
        
        return html

# ================= 🚀 主流程编排 =================

def process_rss_to_eml(rss_file: str, output_dir: str = '.') -> str:
    """
    主流程：从RSS文件生成EML报告
    
    Args:
        rss_file: RSS XML文件路径
        output_dir: 输出目录
    
    Returns:
        生成的EML文件路径
    """
    print("\n" + "="*70)
    print("🚀 混合架构 RSS 智能分析系统 v4.0")
    print("="*70 + "\n")
    
    # 1. 解析RSS文件，提取链接
    print("📄 解析RSS文件...")
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(rss_file)
        root = tree.getroot()
        
        links = []
        for item in root.findall('.//item'):
            link_elem = item.find('link')
            if link_elem is not None and link_elem.text:
                links.append(link_elem.text.strip())
        
        print(f"✅ 找到 {len(links)} 个链接\n")
    except Exception as e:
        print(f"❌ RSS解析失败: {e}")
        return ""
    
    # 2. 深度抓取和分析每个链接
    articles = []
    
    for idx, url in enumerate(links, 1):
        print(f"[{idx}/{len(links)}] 处理链接...")
        
        # 深度抓取
        article_data = DeepWebScraper.extract_article_content(url)
        
        if not article_data['success'] or article_data['word_count'] < 100:
            print("  ⚠️ 内容不足，跳过\n")
            continue
        
        # 语义分析
        print("  🧠 语义分析中...")
        analysis = SemanticAnalyzerV4.analyze_content(article_data)
        
        # 生成中文标题
        chinese_title = TemplateEngine.generate_chinese_title(
            analysis, article_data['title']
        )
        print(f"  📝 标题: {chinese_title}")
        
        # 生成三要点
        key_points = TemplateEngine.generate_key_points(analysis, article_data)
        
        # 生成详细说明
        detailed_explanation = TemplateEngine.generate_detailed_explanation(
            analysis, article_data, key_points
        )
        
        # 汇总
        articles.append({
            'url': url,
            'original_title': article_data['title'],
            'chinese_title': chinese_title,
            'category': analysis['category'],
            'author': article_data['metadata'].get('organization', article_data['metadata'].get('source', '未知')),
            'key_points': key_points,
            'detailed_explanation': detailed_explanation,
            'analysis': analysis
        })
        
        print(f"  ✅ 完成 [分类: {analysis['category']}]\n")
    
    # 3. 生成EML报告
    print("="*70)
    print("📧 生成EML报告...")
    
    if not articles:
        print("❌ 没有有效文章，无法生成报告")
        return ""
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    eml_content = EMLGenerator.generate_eml(articles, date_str)
    
    # 4. 保存文件
    output_file = f"{output_dir}/{date_str}.eml"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(eml_content)
    
    print(f"✅ 报告已生成: {output_file}")
    print(f"📊 共处理 {len(articles)} 篇文章")
    
    # 统计
    category_counts = {}
    for article in articles:
        cat = article['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\n📂 分类统计:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}篇")
    
    print("\n" + "="*70)
    
    return output_file

# ================= 🧪 测试入口 =================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        rss_file = sys.argv[1]
        output_file = process_rss_to_eml(rss_file)
        print(f"\n✅ 完成！输出文件: {output_file}")
    else:
        print("用法: python hybrid_rss_v4.py <rss_file.xml>")
        print("\n系统已就绪，等待RSS文件输入...")
