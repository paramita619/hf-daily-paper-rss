"""
⚙️ 配置文件 - 在这里快速调整系统行为
==========================================
修改这个文件后，重新运行 intelligent_rss_aggregator.py 即可生效
"""

# ================= 🎯 评分权重配置 =================

SCORING_WEIGHTS = {
    # 权威级别分数
    "pioneer": 150,           # 图灵奖得主、领域奠基人
    "top_lab": 120,           # 顶级研究机构
    "current_leader": 80,     # 现任CEO/关键决策者
    "researcher": 60,         # 知名研究者
    
    # 技术深度分数
    "hardcore_tech": 50,      # 端侧AI核心技术
    "infrastructure": 45,     # 底层基础设施
    "algorithm": 35,          # 前沿算法
    "hardware": 20,           # 硬件/芯片（需配合其他）
    "company": 5,             # 公司名（基础分）
    
    # 特殊加成
    "title_tech_authority_bonus": 30,  # 标题同时包含技术词+权威
}

# ================= 📏 来源配置 =================

SOURCE_SETTINGS = {
    # 格式: "来源名": {"基础分": int, "阈值": int}
    # 基础分：该来源天然的可信度
    # 阈值：需要达到多少分才推送
    
    "HF Papers": {
        "base_score": 100,     # 顶级论文平台，高可信度
        "threshold": 120,      # 但也要筛选，避免灌水论文
    },
    
    "AlphaXiv": {
        "base_score": 100,     # arXiv论文
        "threshold": 120,      # 同样标准
    },
    
    "Hacker News": {
        "base_score": 40,      # 聚合平台，鱼龙混杂
        "threshold": 80,       # 需要强关键词支持
    },
    
    "TechCrunch": {
        "base_score": 60,      # 知名科技媒体
        "threshold": 100,      # 需要配合技术词或权威
    },
    
    "a16z": {
        "base_score": 80,      # 顶级VC视角
        "threshold": 100,      # 行业洞察价值高
    },
}

# ================= 🗑️ 噪音过滤配置 =================

NOISE_FILTER_SETTINGS = {
    # 需要多少个噪音关键词才触发一票否决
    "noise_threshold": 2,
    
    # 噪音检测分数（命中即返回此分数）
    "noise_penalty": -100,
}

# ================= 📊 输出配置 =================

OUTPUT_SETTINGS = {
    # RSS文件名
    "output_filename": "intelligent_feed.xml",
    
    # RSS元数据
    "rss_title": "🧠 Intelligent AI & Tech Feed",
    "rss_description": "High-quality, authority-focused feed for AI research, edge computing, and technical breakthroughs.",
    
    # 每个来源最多抓取多少条
    "max_items_per_source": {
        "HF Papers": 20,
        "AlphaXiv": 30,
        "Hacker News": 40,
        "TechCrunch": 15,
        "a16z": 20,
    },
    
    # 是否显示被拒绝的高分文章（调试用）
    "show_rejected_high_score": True,
    "rejected_score_threshold": 50,  # 只显示分数>50但仍被拒的文章
}

# ================= 🎓 权威人物/机构 (快速添加) =================

# 在这里添加你关注的新权威
CUSTOM_AUTHORITIES = {
    # 格式: "姓名/机构": 分数
    # 
    # 示例:
    # "elon musk": 80,
    # "your favorite lab": 120,
}

# ================= ⚡ 技术关键词 (快速添加) =================

# 添加你特别关注的技术词
CUSTOM_TECH_KEYWORDS = {
    # 格式: "关键词": 分数
    # 
    # 示例:
    # "webgpu": 50,
    # "rust for ml": 45,
}

# ================= 🚫 自定义噪音词 =================

# 添加你想过滤的特定词汇
CUSTOM_NOISE_KEYWORDS = [
    # 示例:
    # "clickbait word",
    # "spam term",
]

# ================= 🏷️ 分类显示名称 =================

CATEGORY_DISPLAY_NAMES = {
    "authority": "🎓 权威发声",
    "top_research": "🔬 顶级研究", 
    "edge_tech": "⚡ 端侧/底层技术",
    "algorithm": "🧠 模型算法",
    "hardware": "💻 芯片硬件",
    "industry": "📰 行业动态",
}

# ================= 🔍 调试配置 =================

DEBUG_SETTINGS = {
    # 是否显示详细的评分过程
    "verbose_scoring": False,
    
    # 是否在标题中显示分数
    "show_score_in_title": True,
    
    # 是否保存被拒绝的文章到单独文件（用于分析）
    "save_rejected": False,
    "rejected_filename": "rejected_articles.json",
}

# ================= 💡 使用提示 =================
"""
快速调整指南：

1. 觉得某个来源文章太多？
   → 提高该来源的 threshold

2. 觉得某个来源文章太少？
   → 降低该来源的 threshold，或提高 base_score

3. 想关注新的技术领域？
   → 在 CUSTOM_TECH_KEYWORDS 添加关键词

4. 想过滤特定类型的内容？
   → 在 CUSTOM_NOISE_KEYWORDS 添加噪音词

5. 想调整整体数量？
   → 同时调整所有来源的 threshold（上调减少，下调增加）

建议调整幅度：
- 阈值调整: ±10分为一档
- 权重调整: ±5分为一档
- 每次只调整一个参数，观察效果后再继续
"""
