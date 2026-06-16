# SFP 内容解析 + 与 Storm Slivkoff 博士论文的对比

> 记录这次对 SFP 仓库的逐节解析,以及它与 Paradigm 数据科学家 Storm
> Slivkoff(`@sslivkoff` / `@notnotstorm`)神经科学博士工作的思想对比。
>
> ⚠️ 诚实声明:Storm 博士论文的**确切标题与全文尚未坐实**(见 §4)。本文中对他
> 论文内容的描述基于公开资料(Paradigm 团队页、LinkedIn、Gallant Lab 相关工作),
> 属于**合理推断**,不是逐字引用。SFP 部分则全部来自本仓库源码与 `paper/`。

---

## 1. 这个仓库是什么

**Sparse Feature Preservation (SFP)** —— Paradigm 的一个开源项目,研究大语言模型的
**灾难性遗忘**(学新任务时退化旧能力),包含三样东西:

1. **一套 benchmark**(考遗忘):5 个顺序任务 math → code → ifeval → safety → domain,
   两个对立指标 retention(记性)/ plasticity(可塑),线上按 `0.6·retention + 0.4·plasticity` 排名。
2. **一个方法 SFP**:只保住少数"重要激活方向"防漂移。
3. **一个 Lean 4 形式化证明**(0 `sorry`):严格验证"若假设成立则遗忘有界"。

论文:**《Sparse Feature Preservation: Formally Verified Forgetting Mitigation via
Low-Dimensional Activation Drift》**,作者 Georgios Konstantopoulos(Paradigm),
NeurIPS 2025 preprint 格式(源码在 `paper/`,产物 `sfp.pdf`)。

---

## 2. 论文逐节内容

### 引言
- **问题**:现有遗忘缓解(replay / 蒸馏 / EWC / 架构约束)缺一个"可测量 ↔ 遗忘上界"的**严格联系**。
- **核心假设 H1**:遗忘主要由**少数激活维度的漂移**造成,不是整个激活空间弥散变化。
- **三步**:① 中间层激活提 PCA 基;② 按**消融重要性**排名维度;③ 训练时只惩罚 top-r 重要维度的漂移。
- **卖点**:Lean 4 形式化,0 `sorry`。

### 方法(假设 + 算法 + 证明)
- **H1 正式化**:`遗忘 ≤ C·‖P_W(Δa)‖`,且 top-r 漂移比 total/random/bottom 更能预测遗忘。
  **预注册**判据:必须 `top > total > random > bottom`。
- **消融重要性**:把某方向从激活投影减掉,旧任务损失涨多少 = 它的重要性。
- **算法两阶段**:训练前在 L/4、L/2、3L/4 采激活 → PCA(k=128)→ 消融排名 → 选 top-r(r=32)→ 记锚点;
  训练时 `L = L_new + λ·Σ‖U_r^T a(θ) − U_r^T a(θ*)‖²`,补空间随便变。还有**梯度版变体**。
- **Lean 证明(~1200 行,0 sorry)**:保留损失非负、勾股分解(保子空间严格比禁所有漂移容易)、
  梯度正交、主定理 `sfp_reduces_forgetting`(H1 ⟹ 遗忘 ≤ C·√δ)、稳定-可塑权衡。
  **明确写了范围:Lean 只证 "H1 ⟹ 上界",不证 H1 本身。**

### 实验(好坏都报)
- **设置**:SmolLM2-135M(CPU)+ Qwen2.5-1.5B(A10G);LoRA rank32;主场景 math→code(GSM8K→MBPP,500 步)。
- **相关性(1.5B)**:

  | 预测子 | 维度 | R² |
  |---|---|---|
  | **Top-r(消融)** | 32 | **0.933** |
  | 总漂移 | 128 | 0.888 |
  | 随机 | 32 | 0.850 |
  | Bottom-r | 32 | 0.827 |

  排序符合 H1,但 top 仅领先 random 0.083 → 论文自判 **"部分通过 2/3"**。
- **反面结果(135M)**:H1 **不成立、排序反转**,论文坦白"无确定解释",并说 H1 **依赖规模**。
- **因果证据**:推理时注噪声,1σ 时打 top-r 损害是随机的 **2.5×**、bottom 的 **28×**;
  **135M 上因果测试照样成立**(即使相关性测试在 135M 失败)。
- **维度扫描**:**单 1 个主成分就占 96.2% 方差、R²=0.854** → 遗忘信号极度集中。
- **五科全量表**:`\todo`,**还没跑**(等 GPU / 等 benchmark 标签)。

---

## 3. "降维"到底怎么做(SFP 引擎)

对应 `features.py`:

1. `collect_activations`:跑小抄句子,层钩子取激活,按词平均 → 矩阵 `X = [样本, 1536]`。
2. `build_pca_basis`:对 X 做 **SVD**,取前 k=128 主成分 `U`(点云里"最被拉开"的方向)。
3. `rank_importance`:**逐个抹掉**主成分,看旧任务掉多少分 → 重要性。这步是 SFP 比纯 PCA 多的关键
   ("方差大" ≠ "任务重要")。
4. `select_top_r`:按重要性留 top-r=32 → `U_r = [1536, 32]`。
5. 投影 `anchor = X @ U_r`:**1536 维 → 32 维**。训练时只盯这 32 维别漂移。

核心一句:**SVD/PCA 找主方向(降维)+ 消融测试挑任务相关方向(选重点)。**

---

## 4. Storm Slivkoff 是谁(已核实 / 存疑)

**已核实**:
- `@sslivkoff` = **Storm Slivkoff**(推特 `@notnotstorm`),**Paradigm 数据科学家**。
- 开源加密数据工具:**Cryo、checkthechain(ctc)、flood、Paradigm Data Portal**。
- **UC Berkeley 神经科学博士(约 2020)**;本科 Rice(生物工程 + 应用数学)。
- 方向:人脑 **fMRI 体素编码模型(voxelwise encoding)**、**视觉注意力**、把高维脑活动压进**低维状态空间**;
  与 **Gallant Lab** 圈子相关。

**存疑(未坐实)**:
- 博士论文**确切标题与全文**未找到(Berkeley eScholarship / Google Scholar 未直接命中)。
- 某搜索摘要提到 "MILP for Experimental Design / Neuron" —— **存疑,疑似 AI 摘要错配,未采信**。
- 待办:去 eScholarship 按 "Slivkoff" 精确检索学位论文 PDF 以核实本文 §5 的描述。

---

## 5. 内容对比:SFP ↔ Storm 的神经科学 PhD

> 注意:**SFP 论文不是 Storm 写的**(作者是 Konstantopoulos)。这是**思想/方法谱系**对比,
> 不是同一作者的传承 —— 是"相似直觉在不同领域的回响"。

**共同内核**:高维神经活动里,真正有用的信息住在一个**低维子空间**;能把它找出来并**选择性地下手**。

| 维度 | Storm(神经科学 fMRI) | SFP(LLM) |
|---|---|---|
| 高维原始信号 | 几万个体素响应 | 1536 维激活 |
| 降维工具 | PCA / SVD / 状态空间建模 | PCA / SVD(`build_pca_basis`) |
| "找重点" | 注意力调制 / 任务子空间可分性 | 消融排序找重要方向 |
| 探测方式 | 跨脑区读体素响应 | 在 L/4、L/2、3L/4 层挂钩子读激活 |
| 结果 | 低维认知状态空间 | 32 维重要子空间 |

**呼应**:SFP 仓库里的 `forgetting_curve` 投稿直接引用 **Ebbinghaus 遗忘曲线 / 间隔重复 /
Maximally Interfered Retrieval** —— 全是认知科学概念。"用脑科学的记忆/注意力直觉治 LLM 遗忘"
这条线,与 Storm 的"神经科学 + 数据科学"底子是**同一种思维方式**。

**差别(别夸大相似)**:
- **目标**:他为**理解大脑**(科学解释);SFP 为**改进模型**(工程干预 + 形式化证明)。
- **因果**:神经科学多为观测/相关;SFP 还做了**主动干预**(注噪声看因果)并想证因果界。
- **关系**:非直接传承,而是 Paradigm 内"神经科学/数据 → 加密 → AI"人才谱系下相似直觉的回响。

---

## 6. 一句话总结

> 大模型"学新忘旧";SFP 提出"遗忘住在少数重要激活方向里"这个**可证伪假设**,
> 用 **PCA 降维 + 消融排序挑重点**只保住那几个方向(~40 行),并用 **Lean 严格证明
> "若假设成立则遗忘 ≤ C√δ"**;实测在 math→code 上 **1.5B 部分成立、135M 不成立、
> 因果两个尺度都过、遗忘信号集中到 1 个方向就够**,五科全量表尚缺。
> 这套"高维激活 → 低维重要子空间 → 选择性保留"的思路,与 Storm Slivkoff
> "高维体素 → 低维认知状态 → 注意力挑重点"的神经科学博士工作**惊人同构** ——
> 一个为了懂大脑,一个为了治模型。

---

## 来源

- 本仓库:`paper/sections/*.tex`、`features.py`、`methods.py`、`data.py`、`README.md`、`ITERATION.md`
- [Storm Slivkoff — Paradigm team](https://www.paradigm.xyz/team/storm-slivkoff)
- [sslivkoff — GitHub](https://github.com/sslivkoff)
- [Gallant Lab publications](https://gallantlab.org/publications/)
- [Voxel-Based State Space Modeling Recovers Task-Related Cognitive States (Frontiers, 2020)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.565976/full)
- [Centaur — centaur.run](https://centaur.run/) · [Paradigm: Open Sourcing Centaur](https://www.paradigm.xyz/2026/05/open-sourcing-centaur-multiplayer-self-hosted-secure-agents)
