# 🎯 快速参考卡 - v3.0终极版

## 📊 评分系统速查表

### 来源基础分
```
HF Papers:      +25分
AlphaXiv:       +25分
a16z:           +20分
TechCrunch:     +15分
Hacker News:    +10分
```

### 域名信誉分
```
S级 (学术/官方):  +50分
  arxiv.org, openai.com, pytorch.org...

A级 (知名媒体):   +30分
  techcrunch.com, wired.com, github.com...

B级 (社区):       +15分
  news.ycombinator.com, reddit.com...

黑名单:           -200分
  pinterest, facebook, tiktok...
```

### 权威人物/机构分
```
S级权威:  +100分
  Hinton, LeCun, Bengio, Hassabis, Ilya, Sam Altman...

A级权威:  +70分
  Karpathy, Andrew Ng, Fei-Fei Li, OpenAI, DeepMind...

B级权威:  +40分
  知名研究者, Google Research, Apple ML...
```

### 技术关键词分
```
L1核心技术:  前3个各+40分, 之后+10分
  on-device, edge ai, quantization, llama.cpp, MLX,
  int4, int8, gguf, npu, tinyml...

L2底层技术:  前2个各+25分
  tensorrt, onnxruntime, webgpu, cuda kernel,
  flash attention, kv cache...

L3相关技术:  前2个各+15分
  transformer, lora, qlora, rag, moe...
```

### 语义组合加分
```
边缘优化组合:     +60分
  (on-device/edge) + (quantization/compression)

硬件性能组合:     +50分
  (硬件名) + (benchmark/acceleration)

开源工具组合:     +45分
  (open-source) + (tool/library/framework)

顶级研究组合:     +80分
  (paper/arxiv) + (S/A级权威)

真实发布组合:     +35分
  (公司名) + (release/launch/ship)
```

### 行业门控
```
公司名/硬件名 + 技术词/强动作:  +20分
公司名/硬件名 无实质内容:       -30分
```

### 噪音惩罚
```
硬噪音 (一票否决):  -500分
  stock price, discount, phone case...

中等噪音:
  1个命中:  -50分
  2+命中:   -300分
  (rumor, leak, top 10, listicle...)

软噪音 (上下文):
  review 无技术:  -150分
  review 有技术:  -10分
```

### 时效衰减
```
0-24h:    ×1.0
24-48h:   ×0.9
2-7天:    ×0.8
>7天:     ×0.7
```

---

## 🎯 阈值速查表

```
来源           阈值    说明
─────────────────────────────────────
HF Papers      60分    顶级论文也要筛选
AlphaXiv       60分    同上
a16z           65分    VC视角稍宽松
TechCrunch     70分    媒体需强关键词
Hacker News    75分    聚合平台最严格
```

---

## 🔧 常见调整场景

### 场景1: 文章太少（每天<5篇）

**方案A**: 降低阈值
```python
threshold = {
    "HF Papers": 55,      # -5
    "TechCrunch": 65,     # -5
    "Hacker News": 70     # -5
}
```

**方案B**: 提高技术词权重
```python
# 在 score_technical_depth() 中
points = 45 if l1_count <= 3 else 15  # 原40/10
```

**方案C**: 放宽行业门控
```python
# 在 score_industry_gate() 中
return 25, ["..."]  # 原20分
```

---

### 场景2: 质量不够（垃圾太多）

**方案A**: 提高阈值
```python
threshold = {
    "HF Papers": 65,      # +5
    "TechCrunch": 75,     # +5
    "Hacker News": 80     # +5
}
```

**方案B**: 加强噪音过滤
```python
# 添加自定义噪音词
HARD_NOISE.add("你想过滤的词")
```

**方案C**: 提高域名要求
```python
# 只接受S级域名
if domain not in TIER_S_DOMAINS:
    return -100, ["Only S-tier domains"]
```

---

### 场景3: 重复内容太多

**方案A**: 降低相似度阈值
```python
# 在 SmartDeduplicator 中
if text_similarity(...) > 0.80:  # 原0.85
```

**方案B**: 检查指纹长度
```python
# 缩短指纹前缀
seen_fingerprints[fingerprint[:6]]  # 原[:8]
```

---

### 场景4: 想关注特定领域

**方案A**: 添加自定义权威
```python
TIER_A_AUTHORITIES.add("你关注的实验室/人物")
```

**方案B**: 添加L1关键词
```python
CORE_TECH_L1.add("你关注的技术词")
```

**方案C**: 添加语义组合
```python
# 在 detect_tech_combos() 中添加
if "你的关键词1" in text and "关键词2" in text:
    combos.append(("custom_combo", 70))
```

---

### 场景5: 过滤特定类型

**方案A**: 添加噪音词
```python
HARD_NOISE.add("你不想看的词")
```

**方案B**: 域名黑名单
```python
BLOCKED_DOMAINS.add("spam-site.com")
```

**方案C**: 负面语义组合
```python
# 在 detect_negative_combos() 中
if "不想要的模式" in text:
    negatives.append(("custom_negative", -100))
```

---

## 📊 调试技巧

### 1. 查看被拒文章
在main()中添加:
```python
for item in unique_items:
    score, reasons = AdvancedScorer.comprehensive_score(...)
    if 40 <= score < threshold:  # 接近但未通过
        print(f"⚠️ [{score}] {item['title']}")
        print(f"   Reasons: {reasons}")
```

### 2. 分析分数分布
```python
scores = [item['score'] for item in scored_items]
print(f"平均分: {sum(scores)/len(scores):.1f}")
print(f"中位数: {sorted(scores)[len(scores)//2]}")
print(f"最高分: {max(scores)}")
print(f"最低分: {min(scores)}")
```

### 3. 追踪评分原因
```python
# 查看某个类别的评分分布
for cat in ["模型算法", "平台底座", "行业动态"]:
    items = [i for i in scored_items if i['category'] == cat]
    avg = sum(i['score'] for i in items) / len(items) if items else 0
    print(f"{cat}: {len(items)}篇, 平均{avg:.1f}分")
```

---

## 🎓 最佳实践

### DO ✅
1. **渐进调整**: 每次改一个参数，观察3天
2. **记录数据**: 保存每次的通过率、分数分布
3. **周期审查**: 每周检查一次被拒的高分文章
4. **平衡多样性**: 不要只看最高分，也要看各类别

### DON'T ❌
1. **过度降低阈值**: 会导致质量崩塌
2. **忽略时效性**: 过时内容要降权
3. **盲目相信分数**: 分数是工具，不是目标
4. **一成不变**: 技术快速变化，配置也要更新

---

## 🔍 快速诊断表

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| 每天<3篇 | 阈值太高 | 降低5-10分 |
| 垃圾文章多 | 噪音过滤不够 | 添加噪音词 |
| 重复内容多 | 去重太宽松 | 降低相似度阈值 |
| 错过重要内容 | 关键词库不全 | 添加权威/技术词 |
| 分数虚高 | 关键词堆砌 | 已有递减机制 |
| 来源单一 | 某来源阈值太低 | 调整各来源阈值 |

---

## 💻 命令行快速测试

```bash
# 运行主程序
python ultimate_rss_aggregator.py

# 只看通过的文章标题
python ultimate_rss_aggregator.py | grep "✅"

# 统计各类别数量
python ultimate_rss_aggregator.py | grep "By Category:" -A 10

# 查看最高分文章
python ultimate_rss_aggregator.py | grep "Top article"
```

---

## 📝 配置模板

### 保守模式（质量优先）
```python
threshold = {"HF Papers": 70, "AlphaXiv": 70, "TechCrunch": 80, 
             "a16z": 75, "Hacker News": 85}
max_total = 5  # 每天只推5篇
```

### 平衡模式（推荐）
```python
threshold = {"HF Papers": 60, "AlphaXiv": 60, "TechCrunch": 70,
             "a16z": 65, "Hacker News": 75}
max_total = 10  # 每天10篇
```

### 探索模式（数量优先）
```python
threshold = {"HF Papers": 50, "AlphaXiv": 50, "TechCrunch": 60,
             "a16z": 55, "Hacker News": 65}
max_total = 15  # 每天15篇
```

---

**快速开始**: 使用"平衡模式"运行3天，根据结果调整到"保守"或"探索"模式。
