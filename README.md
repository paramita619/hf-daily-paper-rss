# 🧠 智能RSS聚合器 v2.0

## 🎯 核心改进

### 1. **多维度评分系统**
不再是简单的关键词匹配，而是综合考虑：
- **来源权威度** (40-100分基础分)
- **作者/机构声誉** (最高+150分)
- **技术深度** (按关键词重要性分级打分)
- **内容质量** (上下文分析，避免标题党)

### 2. **动态阈值**
不同来源使用不同的准入标准：
- **arXiv/HuggingFace**: 阈值120分（顶会论文，但也要筛选）
- **TechCrunch/a16z**: 阈值100分（需要强关键词支持）
- **Hacker News**: 阈值80分（聚合平台，最严格筛选）

### 3. **智能噪音过滤**
- **一票否决机制**：检测到2个以上噪音关键词直接-100分
- **分类检测**：金融炒作、消费评测、谣言、娱乐八卦、低质量聚合
- **上下文理解**：不只看单词，还看组合（如"Apple stock"会被过滤，但"Apple NPU"不会）

### 4. **权威知识库分级**

#### 🏆 先驱者 (Pioneers) - +150分
图灵奖得主、领域奠基人，**无论说什么都值得关注**：
- Geoffrey Hinton, Yann LeCun, Yoshua Bengio
- Ilya Sutskever, Andrej Karpathy, Demis Hassabis
- Andrew Ng, Fei-Fei Li, Jeff Dean

#### 🔬 顶级实验室 (Top Labs) - +120分
论文来源为这些机构时几乎必看：
- 工业界：OpenAI, DeepMind, Meta AI, Google Research, Anthropic
- 学术界：MIT, Stanford, Berkeley, CMU, ETH Zurich

#### 👔 现任领导者 (Current Leaders) - +80分
重大战略决策值得关注：
- Sam Altman (OpenAI CEO)
- Jensen Huang (NVIDIA CEO)
- Satya Nadella (Microsoft CEO)

### 5. **技术关键词分级**

#### ⚡ 核心技术 (每个+50分)
端侧AI的硬核内容：
- `on-device ai`, `edge ai`, `tinyml`, `npu`, `quantization`
- `llama.cpp`, `mlx`, `executorch`, `int4`, `awq`, `gptq`

#### 🏗️ 底层技术 (每个+45分)
基础设施和系统优化：
- `cuda kernels`, `flash attention`, `moe`, `kv cache`
- `tensor parallelism`, `continuous batching`

#### 🧠 前沿算法 (每个+35分)
模型和训练方法：
- `transformer`, `diffusion`, `rlhf`, `dpo`, `chain-of-thought`
- `retrieval augmented`, `lora`, `mechanistic interpretability`

#### 💻 硬件/芯片 (每个+20分)
**必须配合其他技术词**才有价值：
- `a18 pro`, `m4 chip`, `snapdragon 8 elite`, `h100`, `b200`

#### 🏢 公司名称 (每个+5分)
**基础分，单独不够入选**：
- Apple, Google, NVIDIA, Qualcomm, OpenAI等

## 📊 评分示例

### ✅ 高质量内容（会通过）

1. **"Geoffrey Hinton discusses reasoning in neural networks"**
   - 先驱者: +150分
   - 来源基础分: +100分
   - **总分: 250分** ✅ (远超阈值)

2. **"New paper from Stanford on on-device LLM quantization"**
   - 顶级实验室: +120分
   - 核心技术(quantization): +50分
   - 核心技术(on-device): +50分
   - 来源基础分: +100分
   - **总分: 320分** ✅

3. **"Apple's M4 chip enables 4-bit quantization for local AI"**
   - 硬件(M4): +20分
   - 核心技术(quantization): +50分
   - 核心技术(4-bit): +50分
   - 公司名(Apple): +5分
   - 来源基础分: +60分
   - **总分: 185分** ✅

### ❌ 低质量内容（会被过滤）

1. **"Apple stock surges after iPhone 16 sales beat expectations"**
   - 公司名(Apple): +5分
   - 来源基础分: +60分
   - **噪音检测**: "stock", "sales" → -100分
   - **总分: -100分** ❌ (一票否决)

2. **"Top 10 AI apps for productivity in 2024"**
   - 来源基础分: +60分
   - **噪音检测**: "top 10 apps", "聚合内容" → -100分
   - **总分: -100分** ❌

3. **"Rumor: Samsung developing new AI chip"**
   - 公司名(Samsung): +5分
   - 来源基础分: +60分
   - **噪音检测**: "rumor" → -100分
   - **总分: -100分** ❌

4. **"Google announces new AI feature"** (仅此标题)
   - 公司名(Google): +5分
   - 来源基础分: +60分
   - **总分: 65分** ❌ (低于阈值80分)

## 🚀 使用方法

### 安装依赖
```bash
pip install requests beautifulsoup4 PyRSS2Gen --break-system-packages
```

### 运行
```bash
python intelligent_rss_aggregator.py
```

### 输出
生成 `intelligent_feed.xml` 文件，可导入任何RSS阅读器。

## 📈 预期效果

### 原系统问题
- ❌ "Apple" → 推送（太宽泛）
- ❌ 股市新闻、促销信息混入
- ❌ 谣言和八卦难以过滤
- ❌ 小V访谈和权威论文混在一起

### 新系统优势
- ✅ "Apple NPU optimization" → 推送（技术相关）
- ✅ "Apple stock" → 过滤（金融噪音）
- ✅ Geoffrey Hinton的任何言论 → 推送（绝对权威）
- ✅ 普通访谈 → 过滤（除非是顶级实验室或先驱者）
- ✅ 自动发现新的重要机构（通过技术词+实验室名组合）

## 🎯 关键设计理念

### 1. **信号vs噪音比**
不追求数量，追求每一条都值得阅读。宁可漏掉边缘内容，也不推送垃圾。

### 2. **权威优先**
人类的时间有限，优先看最权威的声音。一篇Hinton的访谈 > 100篇普通博客。

### 3. **技术深度判断**
不看"AI"这种泛泛的词，看"quantization"、"NPU"这种硬核词。

### 4. **上下文理解**
"Apple" 单独出现 → 可能是任何新闻
"Apple" + "NPU" → 技术新闻
"Apple" + "stock" → 金融新闻 → 过滤

### 5. **动态阈值**
arXiv论文天然可信度高，可以放宽；Hacker News聚合平台，必须严格。

## 🔧 自定义配置

### 添加新的权威人物
在 `AuthorityDatabase.PIONEERS` 中添加：
```python
PIONEERS = {
    "your favorite researcher",
    # ...
}
```

### 调整技术关键词
在 `TechnicalKeywords.HARDCORE_EDGE_AI` 中添加：
```python
HARDCORE_EDGE_AI = {
    "your specific tech term",
    # ...
}
```

### 修改来源阈值
在 `IntelligentScorer.SOURCE_CONFIG` 中调整：
```python
"Your Source": {"base": 70, "tier": SourceTier.TIER_B, "threshold": 90}
```

## 📊 统计输出

运行后会显示：
- 总抓取数 vs 通过数
- 通过率
- 分类统计（权威发声、顶级研究、端侧技术等）
- 来源统计（各平台贡献度）
- 最高分文章

## 🎓 学习价值

这个系统展示了：
1. **如何设计多维度评分系统**
2. **如何平衡精确率和召回率**
3. **如何用代码模拟人类的判断逻辑**
4. **如何处理噪音数据**
5. **如何构建可扩展的知识库**

---

**建议**: 运行一周后，查看输出的RSS，根据你的实际需求调整权重和阈值。这是一个活的系统，应该持续优化。
