# 第六章重写材料包：NOVA V2 生命周期增强式生成跟踪

本文档用于交给 GPT 重写 `论文-1.pdf` 的第六章。写作目标是：基于当前工程实现，将第六章改写为一个完整、可复现实验的“基于大语言模型的在线 3D 多目标跟踪”章节。请优先采用论文写法，避免像开发日志。所有方法、参数、Prompt、训练与验证流程均以当前工程为准。

## 1. 给 GPT 的总指令

请根据本文档重写论文第六章。第六章主题为：NOVA V2，一种在 V1 pairwise association + Hungarian 跟踪框架上，引入大语言模型进行 Birth/Suppress 和 Keep/End 生命周期决策的在线 3D 多目标跟踪方法。

写作要求：

1. 保持学术论文语气，结构完整，公式清晰。
2. 明确说明该方法是 online tracking：推理时第 t 帧只使用历史轨迹和当前帧检测，不使用未来帧或验证集 GT。
3. 明确说明训练阶段的生命周期标签是 offline supervised label construction：只在训练集内利用完整 GT 序列构造 Keep/End 标签，不造成 train/val 泄漏。
4. 重点写清楚 Prompt Formulator，包括三类任务的 prompt 模板、`<box>` token 与几何 embedding 注入规则。
5. 章节中要能让读者按参数复现系统，包括数据集、模型、loss、训练、推理、评估命令、超参数和消融方案。
6. 实验部分要包含：V1/V2 对比、score threshold sweep、association threshold sweep、max_lost_frames sweep、Birth/Dead 诊断指标、可视化设计。

## 2. 第六章建议标题与摘要

建议标题：

```text
第六章 基于大语言模型生命周期决策的在线 3D 多目标跟踪
```

本章核心摘要可写为：

```text
本章在前述检测器与 V1 关联式跟踪框架的基础上，提出 NOVA V2。该方法保留 V1 中 track-detection pairwise association 与 Hungarian matching 的主体流程，同时将传统启发式轨迹出生与终止策略改写为两类显式语言决策任务：对未匹配检测进行 Birth/Suppress 判断，对未匹配轨迹进行 Keep/End 判断。模型以 Qwen2.5-0.5B 为语言骨干，通过 LoRA 进行轻量微调，并使用几何编码器将三维框、类别、置信度和时间位置编码为 `<box>` token 的连续 embedding。推理阶段严格在线，仅基于历史轨迹、当前帧检测和当前关联分数做决策；训练阶段利用完整训练序列构造监督标签。实验表明，V2 在保持 V1 检测关联能力的同时显著降低 ID switch，并为低检测阈值下的召回提升提供可学习的生命周期控制机制。
```

## 3. 第六章详细结构

### 6.1 研究动机

需要写的核心点：

1. 3D MOT 常见 pipeline 是 tracking-by-detection：先检测，再关联。
2. V1 已经验证 pairwise association + Hungarian 有效，但轨迹生命周期仍依赖启发式规则。
3. V1 隐式生命周期策略：

```text
unmatched detection -> 直接 birth
unmatched track     -> lost_frames += 1，直到超过 max_lost_frames 删除
```

4. 这种策略的问题：
   - 低检测阈值下，FP detection 会被直接生成为新轨迹，precision 降低。
   - 检测漏检或短时遮挡时，固定 `max_lost_frames` 不能适应不同场景。
   - 轨迹是否应继续保留不仅取决于丢失帧数，也和历史运动、类别、最近匹配质量有关。
5. V2 的思想：让 LLM 通过结构化 prompt 对生命周期进行 action-token 决策。

### 6.2 问题定义与符号

定义单类 Car 的在线 3D MOT。

第 t 帧检测集合：

```math
\mathcal{D}_t = \{ d_t^j \}_{j=1}^{N_t}
```

其中每个检测：

```math
d_t^j = (b_t^j, c_t^j, s_t^j)
```

`b_t^j` 是 3D box：

```math
b = (x, y, z, l, w, h, \theta)
```

`c` 是类别，本章第一版只使用 `Car`；`s` 是 detector score。

第 t 帧之前的 active tracks：

```math
\mathcal{T}_{t-1} = \{ \tau_i \}_{i=1}^{M_t}
```

每条轨迹维护：

```math
\tau_i = (id_i, H_i, lost_i)
```

其中 `H_i` 是最近 K 帧历史观测，`lost_i` 是连续未匹配帧数。工程默认：

```text
history_len = 3
history_stride = 1
max_lost_frames = 2
```

### 6.3 V1 基线：关联式在线跟踪

V1 做法：

1. 对 active track 和当前 detection 构造 pair。
2. 对每个 pair 构造 association prompt。
3. 模型输出是否同一目标的概率：

```math
p_{ij}^{A} = P(y_{ij}^{A}=1 \mid \tau_i, d_t^j)
```

其中：

```text
y^A = 1 -> Yes
y^A = 0 -> No
```

4. 得到关联分数矩阵：

```math
S_t(i,j) = p_{ij}^{A}
```

5. Hungarian 求解：

```math
\pi^* = \arg\max_{\pi} \sum_{(i,j)\in \pi} S_t(i,j)
```

6. 接受匹配条件：

```math
S_t(i,j) \ge \tau_{assoc}
```

工程默认：

```text
association_threshold = 0.5
```

7. V1 生命周期启发式：

```text
matched pair        -> update track
unmatched detection -> create new track
unmatched track     -> lost_frames += 1; delete if lost_frames > max_lost_frames
```

V1 代表性验证结果，用户已有 baseline：

```text
eval_score_thresh = 0.6
association_threshold = 0.5
MOTA = 0.6245
IDS = 93
sAMOTA = 0.8650
AMOTA = 0.4167
```

### 6.4 NOVA V2 总体框架

V2 保留 V1 的 association 模块和 Hungarian 匹配流程，不改变 pairwise association 的主体逻辑。在 Hungarian 之后增加两个 LLM 决策阶段：

```text
unmatched detections -> Birth / Suppress
unmatched tracks     -> Keep / End
```

V2 online inference 流程：

```text
1. 读取当前帧 cached detections，并按 eval.score_thresh 过滤
2. active tracks × detections 构造 association prompt batch
3. Qwen 输出 P(Yes)
4. Hungarian + association_threshold 得到 matched / unmatched
5. matched pairs 更新轨迹
6. unmatched detections 构造 birth prompt batch
   - argmax(Birth, Suppress)
   - Birth 才新建 track
   - Suppress 直接丢弃
7. unmatched tracks 构造 lifecycle prompt batch
   - argmax(Keep, End)
   - Keep: lost_frames += 1，保留
   - End: 删除
   - lost_frames > max_lost_frames 强制删除
```

需要强调：

```text
V2 没有新增 birth threshold。
V2 没有新增 dead threshold。
Birth/Suppress 和 Keep/End 都由 action-token argmax 决定。
max_lost_frames 只是 hard cap，不是主要 dead 决策。
```

### 6.5 Prompt Formulator

本节要详细写。Prompt 不是提前离线生成的文件，而是在训练和推理时由 Formulator 动态构造。

#### 6.5.1 `<box>` token 与几何注入规则

文本 prompt 中不直接展开 3D box 数值，而使用特殊占位符：

```text
<box>
```

每出现一个有效 `<box>`，模型都会用 Geometry Encoder 输出的连续 embedding 替换该 token 的输入 embedding。

规则：

1. 有效历史 box 才写 `<box>`。
2. 缺失历史帧只写：

```text
Frame -k: Observation: Missing
```

不写 `<box>`。

3. association prompt 包含：
   - 有效历史 box 若干个
   - 当前 candidate box 一个
4. birth prompt 只包含当前 candidate box 一个。
5. lifecycle prompt 只包含有效历史 box，不包含 candidate box。
6. prompt 中 `<box>` 数量必须等于有效 box embedding 数量。

#### 6.5.2 Association Prompt

模板：

```text
Task: 3D Association
History:
Frame -3: ID: {track_id}, Class: Car, Box: <box>
Frame -2: Observation: Missing
Frame -1: ID: {track_id}, Class: Car, Box: <box>
Candidate:
Class: Car, Box: <box>
Question: Is this the same object?
Answer:
```

实际帧数由 `history_len=3` 控制。若历史某帧缺失，不写 `<box>`。

输出 token 对：

```text
[No, Yes]
```

标签：

```text
0 = No
1 = Yes
```

#### 6.5.3 Birth Prompt

模板：

```text
Task: 3D Track Birth Decision

Candidate:
Class: Car, Box: <box>
Detector score: {score}
Best same-object probability: {best_assoc_score}

Question:
What should happen to this unmatched candidate?

Options:
Birth
Suppress

Answer:
```

解释：

1. `score` 是 detector confidence。
2. `best_assoc_score` 是该 detection 与当前 active tracks 的最大同目标概率或近似关联质量。
3. 该 prompt 用于 unmatched detection。

输出 token 对：

```text
[Suppress, Birth]
```

标签：

```text
0 = Suppress
1 = Birth
```

#### 6.5.4 Lifecycle Prompt

模板：

```text
Task: 3D Track Lifecycle Decision

Track:
ID: {track_id}
Lost frames: {lost_frames}
Best same-object probability: {best_assoc_score}

History:
Frame -3: ID: {track_id}, Class: Car, Box: <box>
Frame -2: Observation: Missing
Frame -1: ID: {track_id}, Class: Car, Box: <box>

Question:
What should happen to this unmatched track?

Options:
Keep
End

Answer:
```

输出 token 对：

```text
[End, Keep]
```

标签：

```text
0 = End
1 = Keep
```

### 6.6 训练数据构造

V2 dataset 混合三类样本：

```text
association: label = Yes / No
birth:       label = Birth / Suppress
lifecycle:   label = Keep / End
```

#### 6.6.1 Association 标签

Detection 与 GT 使用 3D IoU 匹配。阈值：

```text
det_gt_iou_threshold = 0.5
```

对 track `i` 和 detection `j`：

```math
y_{ij}^{A} =
\begin{cases}
1, & \text{if } det_j \text{ matches GT track } id_i \\
0, & \text{otherwise}
\end{cases}
```

负样本采样：

```text
negative_positive_ratio = 3.0
```

hard negative 优先保留距离 track 最近的 negative detection。

#### 6.6.2 Birth 标签

对每个 detection `d_t^j`，先由 GT 匹配得到其对应 GT identity：

```math
g(d_t^j) \in \{-1, id\}
```

当前 active tracks 的 GT identity 集合：

```math
\mathcal{A}_t = \{ id(\tau_i) \mid \tau_i \in \mathcal{T}_{t-1} \}
```

Birth 标签：

```math
y_j^B =
\begin{cases}
1, & g(d_t^j) \ne -1 \land g(d_t^j) \notin \mathcal{A}_t \\
0, & \text{otherwise}
\end{cases}
```

也就是：

```text
Birth:
  detection 匹配到 GT，并且该 GT track_id 当前不在 active tracks 中

Suppress:
  detection 是 FP，或 detection 属于已有 active track
```

Birth 负样本采样：

```text
正样本全部保留
Suppress 负样本最多保留 negative_positive_ratio × positive
Suppress 负样本优先保留 detector score 高的 hard negatives
```

#### 6.6.3 Lifecycle 标签

对当前 unmatched track `τ_i`，判断该 track 的 GT identity 在当前帧之后是否还会出现：

```math
y_i^L =
\begin{cases}
1, & id_i \in \bigcup_{k>t} \mathcal{G}_k \\
0, & \text{otherwise}
\end{cases}
```

其中：

```text
1 = Keep
0 = End
```

注意：

```text
不引入 future_horizon。
直接使用整条 train sequence 的未来 GT 判断 Keep/End。
该未来信息只用于训练标签构造，推理时不使用。
```

Lifecycle 类别采样：

```text
不新增 ratio。
复用 negative_positive_ratio = 3.0。
minority class 全保留。
majority class 最多保留 ratio × minority。
```

### 6.7 模型结构

模型名称可写为：

```text
NOVA Lifecycle Model
```

组成：

1. Qwen2.5-0.5B language model
2. LoRA 参数高效微调
3. Geometry Encoder
4. Adapter
5. action-token classification
6. association quality head

工程配置：

```yaml
model:
  llm_model_name_or_path: /media/ana-4090/LY/LLM_models/Qwen2.5-0.5B
  freeze_llm: true
  llm_torch_dtype: float16

nova:
  geometry_hidden_size: 128
  use_lora: true
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  lora_target_modules: [q_proj, v_proj]
```

#### 6.7.1 Geometry Encoder

对每个 box token，构造输入：

```math
u = [\phi(b), s, e_c(c), e_\tau(type), e_t(\Delta t)]
```

其中：

1. `φ(b)` 是归一化后的 3D box 编码。
2. `s` 是检测或历史观测置信度。
3. `e_c(c)` 是类别 embedding。
4. `e_\tau(type)` 是 token type embedding：

```text
0 = history box
1 = candidate box
```

5. `e_t(Δt)` 是时间位置 MLP embedding。

几何编码器：

```math
z = \mathrm{MLP}(u)
```

Adapter 将几何特征映射到 LLM hidden size：

```math
\tilde{z} = \mathrm{Adapter}(z)
```

#### 6.7.2 `<box>` embedding 注入

设 tokenizer 后的 prompt token embedding 为：

```math
E = [e_1, e_2, \ldots, e_L]
```

若第 `p_k` 个 token 是 `<box>`，则替换为对应几何 embedding：

```math
e'_{p_k} = \tilde{z}_k
```

其他 token 保持原语言 embedding：

```math
e'_l = e_l,\quad l \notin \{p_k\}
```

最终输入 LLM：

```math
H = \mathrm{LLM}(E')
```

取 `Answer:` 位置的下一 token logits 作为动作 logits。

#### 6.7.3 Action Token 决策

三类任务均转化为二分类：

```text
association -> [No, Yes]
birth       -> [Suppress, Birth]
lifecycle   -> [End, Keep]
```

对任务 `q`：

```math
\ell^q = [\ell^q_0, \ell^q_1]
```

概率：

```math
p^q = \mathrm{softmax}(\ell^q)
```

推理：

```math
\hat{y}^q = \arg\max_{c\in\{0,1\}} p^q_c
```

### 6.8 损失函数

V2 总损失：

```math
\mathcal{L} = \mathcal{L}_{action} + \lambda_q \mathcal{L}_{quality}
```

配置：

```text
lambda_quality = 1.0
```

#### 6.8.1 Action Loss

对 batch 内所有 association / birth / lifecycle 样本统一计算 cross entropy：

```math
\mathcal{L}_{action}
= - \frac{1}{B}\sum_{n=1}^{B}
\log
\frac{\exp(\ell_{n,y_n})}
{\exp(\ell_{n,0})+\exp(\ell_{n,1})}
```

标签含义：

```text
association: 0 = No,       1 = Yes
birth:       0 = Suppress, 1 = Birth
lifecycle:   0 = End,      1 = Keep
```

#### 6.8.2 Quality Loss

quality head 只用于 association 样本：

```math
\hat{q}_{ij} = \sigma(f_q(h_{ij}))
```

目标是 detection 与 GT 的 IoU：

```math
q_{ij}^{*} = IoU(b_j, b_{gt})
```

仅当 `quality_valid=True` 时生效。Birth 和 lifecycle 不参与 quality loss。

```math
\mathcal{L}_{quality}
= \frac{1}{|\Omega|}
\sum_{(i,j)\in\Omega}
\mathrm{SmoothL1}(\hat{q}_{ij}, q_{ij}^{*})
```

其中 `Ω` 是有效 association 样本集合。

### 6.9 在线推理算法

可以在论文中给出伪代码：

```text
Algorithm: NOVA V2 Online Tracking
Input: detections D_t, active tracks T_{t-1}, model F
Output: tracks at frame t

1: filter D_t by eval.score_thresh
2: build association prompts for T_{t-1} × D_t
3: compute S(i,j)=P(Yes | τ_i, d_j)
4: run Hungarian on S
5: accept matches with S(i,j) >= association_threshold
6: update matched tracks
7: for each unmatched detection d_j:
8:     build birth prompt
9:     if argmax([Suppress, Birth]) == Birth:
10:        create new track
11:    else:
12:        discard d_j
13: for each unmatched track τ_i:
14:    build lifecycle prompt
15:    if argmax([End, Keep]) == Keep:
16:        lost_i += 1
17:        if lost_i > max_lost_frames: delete τ_i
18:    else:
19:        delete τ_i
```

工程默认推理参数：

```yaml
eval:
  score_thresh: 0.3

nova:
  association_threshold: 0.5
  max_lost_frames: 2
```

用户当前正式对比常用：

```text
eval_score_thresh = 0.6
association_threshold = 0.5
```

### 6.10 数据集与实现细节

数据集：

```text
dataset.name = xian
class_names = [Car]
训练集规模：不到 6000 帧
验证集规模：2119 帧
```

检测器与 cache：

```text
VoxelNeXt 检测器输出 cached detections。
V2 复用 V1 detection cache。
cache root = outputs/nova_qwen05b_a1/detection_cache
detection_cache.score_thresh = 0.05
detection_cache.max_dets_per_frame = 100
```

重要说明：

```text
detection_cache.score_thresh 是生成 cache 时的低阈值。
eval.score_thresh 是训练验证/推理时二次过滤阈值。
```

训练配置：

```text
optimizer = AdamW
lr = 1e-4
batch_size = 32 （用户实际训练）
epochs = 30 （用户实际训练）
validation interval = 3 epochs
save interval = 3 epochs
best metric = MOTA
```

用户实际训练日志摘要：

```text
lifecycle_samples = 75005
steps_per_epoch = 2344
epochs = 30
best_mota during training validation = 0.4850 at epoch 27
best checkpoint iter = 63288
```

注意：训练过程中的验证默认使用配置里的 `eval.score_thresh=0.3`；用户后续正式对比使用 `--eval_score_thresh 0.6`。

### 6.11 实验设计

#### 6.11.1 主对比实验：V1 vs V2

V1 命令：

```bash
python tools/eval_nova_tracking.py \
  --cfg_file configs/generative_tracking/nova_qwen05b_a1.yaml \
  --ckpt outputs/nova_qwen05b_a1/checkpoints/best.pth \
  --split val \
  --eval_score_thresh 0.6 \
  --association_threshold 0.5 \
  --max_lost_frames 2 \
  --eval_metrics \
  --eval_ab3dmot
```

V2 命令：

```bash
python tools/eval_nova_lifecycle_tracking.py \
  --cfg_file configs/generative_tracking/nova_qwen05b_a1_v2_lifecycle.yaml \
  --ckpt outputs/nova_qwen05b_a1_v2_lifecycle/checkpoints/best.pth \
  --split val \
  --eval_score_thresh 0.6 \
  --association_threshold 0.5 \
  --max_lost_frames 2 \
  --eval_metrics \
  --eval_ab3dmot
```

已有结果：

```text
V1 @ score=0.6, assoc=0.5:
MOTA   = 0.6245
IDS    = 93
sAMOTA = 0.8650
AMOTA  = 0.4167

V2 @ score=0.6, assoc=0.5:
precision = 0.8357
recall    = 0.7782
AP_3D@0.5 = 0.7214
MOTA      = 0.6194
IDS       = 44
FRAG      = 169
sAMOTA    = 0.8651
AMOTA     = 0.4163
AMOTP     = 0.7352
MOTP      = 0.7199
```

论文分析重点：

```text
V2 与 V1 在 MOTA/AMOTA 上基本持平。
V2 将 IDS 从 93 降到 44，说明生命周期决策和保守 Birth/Keep 机制提升了身份稳定性。
当前 MOTA 没明显超过 V1 的主要原因不是 IDS，而是 FN 与 FP 仍占主要误差。
```

MOTA 分析公式：

```math
MOTA = 1 - \frac{FN + FP + IDS}{GT}
```

因此当 IDS 已明显下降时，MOTA 的进一步提升主要依赖：

```text
降低 FN，提高 recall；
控制 FP，保持 precision。
```

#### 6.11.2 Score Threshold Sweep

目的：分析 Birth 模块是否能在较低检测阈值下抑制 FP，同时提升 recall。

命令：

```bash
for s in 0.40 0.45 0.50 0.55 0.60 0.65; do
  python tools/eval_nova_lifecycle_tracking.py \
    --cfg_file configs/generative_tracking/nova_qwen05b_a1_v2_lifecycle.yaml \
    --ckpt outputs/nova_qwen05b_a1_v2_lifecycle/checkpoints/best.pth \
    --split val \
    --eval_score_thresh $s \
    --association_threshold 0.5 \
    --max_lost_frames 2 \
    --output outputs/nova_qwen05b_a1_v2_lifecycle/sweep_score_${s}.json \
    --eval_metrics \
    --eval_ab3dmot \
    --ab3dmot_output outputs/nova_qwen05b_a1_v2_lifecycle/sweep_score_${s}_ab3dmot.json
done
```

需要报告：

```text
score_thresh | precision | recall | FP | FN | MOTA | IDS | AMOTA
```

预期现象：

```text
降低 score_thresh 通常 recall 上升、FN 下降，但 FP 可能上升。
如果 V2 Birth 有效，则在较低 score_thresh 下 FP 不会爆炸，MOTA 可能超过 V1。
```

#### 6.11.3 Association Threshold Sweep

目的：分析关联阈值对 IDS、FN 和 Birth 触发频率的影响。

命令：

```bash
for a in 0.40 0.45 0.50 0.55 0.60; do
  python tools/eval_nova_lifecycle_tracking.py \
    --cfg_file configs/generative_tracking/nova_qwen05b_a1_v2_lifecycle.yaml \
    --ckpt outputs/nova_qwen05b_a1_v2_lifecycle/checkpoints/best.pth \
    --split val \
    --eval_score_thresh 0.6 \
    --association_threshold $a \
    --max_lost_frames 2 \
    --output outputs/nova_qwen05b_a1_v2_lifecycle/sweep_assoc_${a}.json \
    --eval_metrics \
    --eval_ab3dmot \
    --ab3dmot_output outputs/nova_qwen05b_a1_v2_lifecycle/sweep_assoc_${a}_ab3dmot.json
done
```

分析逻辑：

```text
association_threshold 降低：更容易匹配，可能降低 FN，但可能增加 IDS。
association_threshold 提高：匹配更保守，IDS 可能降低，但更多 unmatched detection/track 交给 Birth/Lifecycle。
```

#### 6.11.4 max_lost_frames Sweep

目的：公平比较 V1 和 V2 的轨迹终止 hard cap。

V1：

```bash
for m in 0 1 2 3 4; do
  python tools/eval_nova_tracking.py \
    --cfg_file configs/generative_tracking/nova_qwen05b_a1.yaml \
    --ckpt outputs/nova_qwen05b_a1/checkpoints/best.pth \
    --split val \
    --eval_score_thresh 0.6 \
    --association_threshold 0.5 \
    --max_lost_frames $m \
    --output outputs/nova_qwen05b_a1/sweep_lost_${m}.json \
    --eval_metrics \
    --eval_ab3dmot \
    --ab3dmot_output outputs/nova_qwen05b_a1/sweep_lost_${m}_ab3dmot.json
done
```

V2：

```bash
for m in 0 1 2 3 4; do
  python tools/eval_nova_lifecycle_tracking.py \
    --cfg_file configs/generative_tracking/nova_qwen05b_a1_v2_lifecycle.yaml \
    --ckpt outputs/nova_qwen05b_a1_v2_lifecycle/checkpoints/best.pth \
    --split val \
    --eval_score_thresh 0.6 \
    --association_threshold 0.5 \
    --max_lost_frames $m \
    --output outputs/nova_qwen05b_a1_v2_lifecycle/sweep_lost_${m}.json \
    --eval_metrics \
    --eval_ab3dmot \
    --ab3dmot_output outputs/nova_qwen05b_a1_v2_lifecycle/sweep_lost_${m}_ab3dmot.json
done
```

分析重点：

```text
V1 完全依赖 max_lost_frames 控制 dead。
V2 主要由 Keep/End 决策控制 dead，max_lost_frames 只是 hard cap。
如果 V2 对 max_lost_frames 不那么敏感，说明模型学到了更自适应的生命周期策略。
```

#### 6.11.5 Birth/Dead 事件级诊断指标

建议新增或作为未来工作/补充实验。该指标可以公平比较 V1 和 V2。

V1 的隐式策略：

```text
V1 Birth:
  unmatched detection 一律 Birth

V1 Dead:
  unmatched track 一律 Keep
  lost_frames > max_lost_frames 时 End
```

V2 的显式策略：

```text
V2 Birth:
  LLM argmax([Suppress, Birth])

V2 Dead:
  LLM argmax([End, Keep])
  lost_frames > max_lost_frames 时 hard cap End
```

Birth GT 标签：

```text
GT Birth:
  unmatched detection 匹配到 GT，且该 GT track_id 当前不在 active tracks 中

GT Suppress:
  unmatched detection 是 FP，或属于已有 active track
```

Birth 指标：

```math
BirthPrecision = \frac{TP_B}{TP_B + FP_B}
```

```math
BirthRecall = \frac{TP_B}{TP_B + FN_B}
```

```math
BirthF1 = \frac{2 \cdot BirthPrecision \cdot BirthRecall}
{BirthPrecision + BirthRecall}
```

Lifecycle GT 标签：

```text
GT Keep:
  unmatched track 的 GT identity 未来还会出现

GT End:
  unmatched track 的 GT identity 未来不再出现
```

Lifecycle 指标：

```text
Life_Acc
Keep_Recall
End_Precision
EarlyEnd = 未来还会出现但预测 End
StaleKeep = 未来不出现但预测 Keep
```

推荐表格：

```text
Method | Birth_P | Birth_R | Birth_F1 | Life_Acc | Keep_R | End_P | EarlyEnd | StaleKeep | MOTA | IDS
V1     |         |         |          |          |        |       |          |           |      |
V2     |         |         |          |          |        |       |          |           |      |
```

这些指标对应的解释：

```text
Birth_R 低 -> 新目标没被生出，FN 高。
Birth_P 低 -> FP detection 被生太多，FP 高。
Keep_R 低 -> 轨迹过早结束，IDS/FRAG 高。
End_P 低 -> 消失轨迹保留太久，FP 高。
```

#### 6.11.6 消融实验设计

建议表格：

```text
Variant | Association | Birth | Lifecycle | Geometry | Prompt extras | MOTA | IDS | FP | FN | AMOTA
V1      | LLM         | all unmatched det birth | max_lost hard cap | yes | no lifecycle prompt |      |     |    |    |
V2-Full | LLM         | LLM Birth/Suppress      | LLM Keep/End      | yes | score + best assoc   |      |     |    |    |
V2-B    | LLM         | LLM Birth/Suppress      | V1 hard cap       | yes | score + best assoc   |      |     |    |    |
V2-L    | LLM         | V1 birth all            | LLM Keep/End      | yes | best assoc           |      |     |    |    |
NoGeom  | LLM text    | LLM                     | LLM               | no  | text only            |      |     |    |    |
NoBest  | LLM         | LLM                     | LLM               | yes | remove best_assoc    |      |     |    |    |
```

说明：

```text
V2-B 和 V2-L 需要通过代码开关或临时替换 runtime 策略实现。
NoGeom/NoBest 属于 prompt/architecture 消融，若当前工程未实现，可作为可选补充实验或未来工作，不要假称已经完成。
```

### 6.12 可视化设计

建议至少做三类图。

#### 6.12.1 BEV 跟踪可视化

每帧绘制：

```text
GT box: 灰色
matched track: 绿色
Birth accepted: 蓝色
Suppress detection: 红色叉号
Keep unmatched track: 黄色虚线
End track: 紫色终止标记
```

用途：

```text
展示 V2 在低分检测、遮挡、短时漏检情况下如何做生命周期决策。
```

#### 6.12.2 轨迹时间轴可视化

横轴为 frame，纵轴为 track id。标注：

```text
match
birth
suppress
keep
end
hard_cap_end
id_switch
fragment
```

用途：

```text
对比 V1 与 V2 在同一 sequence 上的轨迹连续性和 ID switch。
```

#### 6.12.3 阈值曲线

画：

```text
score_thresh vs MOTA
score_thresh vs recall
score_thresh vs precision
score_thresh vs IDS
association_threshold vs MOTA / IDS
max_lost_frames vs MOTA / FRAG / FP
```

用途：

```text
说明 V2 的生命周期模块是否扩大了低阈值检测的可用范围。
```

### 6.13 结论写法

建议结论：

```text
本章提出 NOVA V2，将传统 3D MOT 中启发式的出生和终止规则改写为 LLM action-token 决策。该方法保留 V1 的 pairwise association 与 Hungarian matching 框架，在未匹配检测和未匹配轨迹上分别引入 Birth/Suppress 与 Keep/End 两类 prompt。通过 `<box>` token 的几何 embedding 注入，语言模型能够在结构化文本上下文中感知 3D 空间信息。实验表明，V2 在 MOTA/AMOTA 基本保持 V1 水平的同时显著降低 ID switch，说明显式生命周期建模有助于提升身份稳定性。进一步的阈值与事件级诊断实验可用于分析 Birth 与 Lifecycle 模块对 FN、FP 和轨迹碎片的具体贡献。
```

需要避免夸大：

```text
不要写 V2 已经显著提升 MOTA，因为当前正式对比中 V2 MOTA=0.6194，V1 MOTA=0.6245，基本持平。
可以强调 IDS 从 93 降到 44，身份稳定性提升明显。
可以说明 MOTA 进一步提升需要改善 FN/FP 平衡，尤其是低 score threshold 下 Birth 模块的 precision-recall。
```

## 4. 当前工程文件索引

V1：

```text
configs/generative_tracking/nova_qwen05b_a1.yaml
tools/train_nova_association.py
tools/eval_nova_tracking.py
generative_tracking/nova_data.py
generative_tracking/nova_model.py
generative_tracking/nova_runtime.py
```

V2：

```text
configs/generative_tracking/nova_qwen05b_a1_v2_lifecycle.yaml
tools/train_nova_lifecycle.py
tools/eval_nova_lifecycle_tracking.py
generative_tracking/nova_data.py
generative_tracking/nova_model.py
generative_tracking/nova_runtime.py
```

输出目录：

```text
V1: outputs/nova_qwen05b_a1/
V2: outputs/nova_qwen05b_a1_v2_lifecycle/
```

Detection cache：

```text
outputs/nova_qwen05b_a1/detection_cache/
```

## 5. 最后给 GPT 的写作提醒

请按照“动机 -> 方法 -> 数据构造 -> 模型 -> 损失 -> 推理 -> 实验 -> 分析”的顺序写。第六章应自洽，不要假设读者看过代码。所有 Prompt 模板必须完整给出。公式要服务于工程实现，不要引入当前工程没有的 birth threshold、dead threshold、future horizon 或额外 loss 权重。实验分析要如实说明：V2 目前主要改善 IDS，MOTA 与 V1 接近；下一步应通过阈值 sweep 和事件级诊断分析 Birth/Dead 的贡献。
