# 硕士毕业论文工作二：实施方案报告

## 1. 论文题目与定位

* **建议题目**：**《面向眼底影像长尾分布的多标签证据不确定性建模研究》**
* *(Research on Multi-label Evidential Uncertainty Modeling for Long-tailed Fundus Image Distribution)*

* **与工作一（C2Net 基础篇）的关系**：
    * **工作一（追求极致性能）**：侧重于特征提取，利用共现关系（Co-occurrence）和 Focal Loss 解决“分类准不准”的问题，建立了高性能基准。
    * **工作二（追求诊断可靠）**：侧重于风险评估。针对长尾分布中**罕见病（Tail classes）容易产生的“盲目自信”现象**，引入证据深度学习（EDL），让模型在输出诊断结果的同时，量化“我不知道（Uncertainty）”的程度，实现安全预警。

---

## 2. 核心痛点与动机 (Motivation)

在工作一的基础上，我们发现了两个进阶难题：
1. **长尾分布的陷阱**：ODIR 数据集中，健康样本和常见病占统治地位。模型在训练几十个 Epoch 后，F1 分数往往会卡在某个固定数值（如 0.628），这反映了模型陷入了“只敢预测常见病，对罕见病消极对待”的局部最优。
2. **决策风险控制**：医疗场景对误诊零容忍。传统的 Sigmoid 输出只能给出点估计概率，无法区分“因为特征明显而确信”还是“因为没见过而瞎猜”。

**本研究的破局点**：
通过 **Beta 证据神经网络 (Beta-ENN)**，将单一概率预测提升为分布预测，并针对多标签长尾场景提出**非对称证据权重（Asymmetric Weighting）**和**证据缩放（Evidence Scaling）**策略，打破性能瓶颈。

---

## 3. 方法论详解 (Methodology)

### 3.1 理论模型：Beta 分布证据建模
对于多标签任务中的第 $c$ 种疾病，我们弃用 Sigmoid，假设其服从 **Beta 分布** $	ext{Beta}(p_c; \alpha_c, \beta_c)$：
*   **$\alpha_c$ (Positive Evidence)**：支持该疾病存在的证据量。
*   **$\beta_c$ (Negative Evidence)**：支持该疾病不存在的证据量。
*   **$S_c = \alpha_c + \beta_c$**：该标签的总证据量。

**关键指标定义：**
1.  **预测概率**：$\hat{p}_c = \alpha_c / S_c$。
2.  **不确定性 (Uncertainty)**：$u_c = 2 / S_c$。当模型对罕见病（Tail）缺乏识别经验时，$S_c$ 较小，$u_c$ 趋于 1。

### 3.2 网络架构改进 (Architecture)
在 ViT/ResNet 主干网络之后，我们将分类头升级为 **EDLHead**：
*   **输出维度**：从 `num_classes` 扩展为 `2 * num_classes`（分别对应 $\alpha$ 和 $\beta$ 的原始 Logits）。
*   **证据缩放 (Evidence Scaling)**：引入缩放因子 $\sigma$（如 10.0），对 Softplus 后的结果进行放大：
    $$\alpha_c = 	ext{Softplus}(f(x)_{c,\alpha}) \cdot \sigma + 1$$
    $$\beta_c = 	ext{Softplus}(f(x)_{c,\beta}) \cdot \sigma + 1$$
    *注：此缩放解决了 Backbone 输出 Logits 动态范围过小导致的“证据贫瘠”问题，是提升 F1 的关键工程手段。*

### 3.3 改进版损失函数设计 (The Innovation)

总 Loss 采用多项式结构：$\mathcal{L} = \mathcal{L}_{risk} + \lambda(t) \cdot \mathcal{L}_{KL} + \gamma \cdot \mathcal{L}_{contra}$

#### (1) 非对称风险损失 ($\mathcal{L}_{risk}$) —— 解决 F1=0.628 的杀手锏
针对多标签 ODIR 任务中“负样本极多、正样本极少”的特点，引入**非对称权重系数 $w_{neg}$**（如 0.2）：
$$\mathcal{L}_{risk} = y \cdot (\psi(S) - \psi(\alpha)) + (1-y) \cdot (\psi(S) - \psi(\beta)) \cdot w_{neg}$$
*   **作用**：大幅压低负样本在损失函数中的统治地位，强制模型产生更多正向证据，显著提升尾部类别的 Recall。

#### (2) 周期性 KL 退火 ($\lambda(t)$) —— 跳出局部最优
摒弃单一线性退火，采用 **周期性退火 (Cyclical Annealing)**。
*   **逻辑**：将训练分为 4 个周期，每个周期开始时 KL 权重从 0 重启。
*   **优势**：在每个周期的开始阶段给模型“松绑”，允许其重新探索 Likelihood 空间，避免过早被 KL 正则项压制为“全 0 预测”。

#### (3) 不确定性感知加权 (UW-Weight)
利用预测的不确定性 $u$ 对 Likelihood 进行加权：$W_{uw} = (1 - u)^\gamma$。
*   让模型更关注那些“虽然有证据但还不够确信”的样本，通过不确定性反馈实现自适应学习。

---

## 4. 实验设计与预期成果

### 4.1 数据集策略
*   **保留所有长尾标签**：不再粗暴合并 Others。
*   **阈值过滤**：仅保留样本数 $\ge 10$ 的独立标签，保持医疗分类的特异性。

### 4.2 核心评价指标
1.  **ECE (Expected Calibration Error)**：证明 EDL 模型的置信度比传统工作一的 Sigmoid 更接近真实准确率。
2.  **不确定性分离度 (Uncertainty Analysis)**：通过直方图展示，对于误诊（Incorrect）样本，模型输出的 $u$ 显著高于正确诊断（Correct）样本。

### 4.3 必须展示的图表
*   **Reliability Diagram**：对比 Sigmoid 与 EDL，展示后者在校准度上的降维打击。
*   **Tail Class Recall Curve**：展示引入非对称权重后，长尾类别的召回率变化。

---

## 5. 修改记录附录 (What's Changed)

为了支持上述方案，我对代码库进行了以下核心修改：

1.  **`losses/edl_loss.py`**：
    *   **新增**：`log_likelihood_loss` 引入 `neg_weight` 参数。
    *   **新增**：`EDLLoss` 类实现 `get_annealing_coef` 周期性退火函数。
2.  **`models/head.py`**：
    *   **新增**：`EDLHead` 支持 `evidence_scale` 参数（默认 10.0），放大证据动态范围。
3.  **`config.py`**：
    *   **新增**：`evidence_scale` (10.0), `neg_weight` (0.2), `annealing_type` ('cyclical') 三个超参数配置及命令行支持。
4.  **`train.py`**：
    *   **更新**：训练循环中将 `total_epochs` 传回损失函数以支持周期性步进。
    *   **更新**：实例化 Head 和 Loss 时自动读取最新的长尾平衡配置。

---
**方案评估结论**：该方案在继承工作一性能优势的基础上，通过 EDL 引入了安全维度，且针对性地解决了 F1 分数停滞的工程问题，具备极强的学术和应用说服力。
