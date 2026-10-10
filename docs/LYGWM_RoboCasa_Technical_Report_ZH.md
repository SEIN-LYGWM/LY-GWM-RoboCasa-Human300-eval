# LY-GWM-RoboCasa-Human300

## 基于图动力学模型的预测引导动作选择

**Tianliang Ma**

北京市密网信息科技有限公司，中国北京

技术报告 v1.3｜2026 年 10 月 10 日｜本地实验与官方复测

### 摘要

本文介绍 LY-GWM-RoboCasa-Human300：一个由经过 Human300 微调的 GR00T-N1.5 动作生成器、动作条件图动力学模型和学习式二元状态奖励评分器组成的联合策略。推理时，策略采样四个动作块，分别预测执行后的潜在特征和机器人状态，并执行预测状态奖励 logit 最高的候选。因此，预测影响传给环境的动作。被评测系统采用 16 步预测跨度，三个组件在评测期间全部冻结。在 RoboCasa 1.0.1 的 pretrain 厨房划分中，对 50 个任务各进行 50 回合本地评测，系统记录了 207/2,500 次成功（8.28%）；历史自训练 GR00T-N1.5 基线为 181/2,500（7.24%）。观察到的差值为 26 次成功，即 1.04 个百分点。原子任务由 141/900 增至 150/900。另外，RoboCasa 维护者对提交系统进行了复测，在 50 个任务中报告了 354/1,500 次成功（23.6%）；Atomic-Seen、Composite-Seen 和 Composite-Unseen 的榜单成功率分别为 48.3%、16.3% 和 3.1% [9]。本地结果和官方结果分别报告，两者差异不归因于模型改进。记录的本地审计识别了 212,700 个发生变化的环境输入动作块，其中原子任务为 28,223 个。本文说明实际训练阶段、已实现的推理机制、评测来源与公开复现入口。这些结果记录了一个实际运行的预测与选择系统，以及正向的历史结果差值。

**关键词：** 机器人操作；视觉语言动作策略；图动力学；动作选择；RoboCasa；可复现性

### 1. 引言

动作生成策略可以针对同一观测产生多个合理动作，而从中选择需要评估其后果。本系统将动作提议、未来状态预测和候选评分分开：GR00T 提出少量候选动作，LY-GWM 预测候选相关的结果，学习式评分器选择要执行的动作。本报告说明具体的 RoboCasa 实现及被评测的预测与选择流程。

本文的贡献是推理时组合：一次共享的 GR00T 骨干计算支持四次动作头采样，候选经过训练好的潜在图模型与奖励头显式评估。我们提供训练规格、精确接口、分组及逐任务结果、动作选择证据与局限，以支持评估这一组合。当前报告关注联合策略评测（S42；标识见第 2.1 节）。公开实现记录了实际运行的四候选预测与选择机制 [5]。

<!-- PAGEBREAK -->

### 2. 背景与范围

GR00T 提供预训练策略基础与动作生成架构 [2,3]。RoboCasa365 提供家庭操作基准与 Human300 任务设置 [1]。此前 LY-GWM 预印本提出将结构化世界表示与动作条件转移分离 [4]。本文将这一思路实例化为用于候选选择的潜在图预测。被评测表示包含 16 个池化骨干特征节点与一个机器人状态节点，在 LY-GWM 更广泛的设计方向下实现了动作条件潜在动力学与学习式预测状态评分。

PRTS 是一个公开基准模型，提供方法论文、代码、检查点和评测详情 [8]。其对比式推理与本系统的同帧奖励评分不同。本文采用常规的方法与实验报告方式，明确标注组件归属。

#### 2.1 组件名称与实验标识

R5、W5、S38、S31 和 S42 是公开文件使用的内部工作流标识，不是标准模型名称、参数数量或单一模型的连续版本。本报告将每个标识与其功能名称对应，使读者能够将方法与相应配置及评测记录关联起来。

| 标识 | 功能名称 | 本报告中的含义 |
| --- | --- | --- |
| R5 | GR00T-N1.5 策略微调 | 在 Human300 上训练动作生成器，生成 step-120,000 策略检查点。 |
| W5 | LY-GWM 图动力学训练 | 训练根据候选动作块预测未来特征和机器人状态的模型。 |
| S38 | 学习式状态奖励评分器训练 | 训练联合推理时用于预测状态评分的二元奖励 MLP。 |
| S31 | 自训练基线评测 | 仅评测 R5 策略，在 pretrain 上记录 181/2,500 次成功。 |
| S42 | 预测引导联合策略评测 | 组合评测 R5 策略、W5 动力学与 S38 评分器，在 pretrain 上记录 207/2,500 次成功。 |
| S30 | 公开检查点参考评测 | 记录公开 GR00T-N1.5 检查点在 target 上的独立本地评测。 |
| S32 | 历史影子预测评测 | 在策略运行时并行计算图预测；当前实际动作选择结果指 S42。 |
| W3 / W4 / S37 | 训练数据准备阶段 | W3 选取转移对；W4 缓存冻结策略特征；S37 准备记录的奖励标签。 |

训练与评测是不同阶段。策略微调（R5）、图动力学训练（W5）与奖励评分器训练（S38）产生三个分别训练的组件。基线评测（S31）仅使用微调后的动作策略。联合策略评测（S42）将同一策略架构与训练好的动力学模型及评分器组合；选择和执行动作时三个组件全部冻结。标识用于贯通文件来源，功能名称则说明系统实际执行的工作。

<!-- PAGEBREAK -->

### 3. 已实现的方法

#### 3.1 输入与候选生成

记相机观测为 o、归一化后的 20 维机器人状态为 s、任务指令为 l。冻结的 GR00T 主干处理策略输入，将有效主干 token 自适应平均池化为 16 个、每个宽度 2,048 的特征节点，并逐节点归一化，得到 F。策略补齐后的状态接口为 B x 1 x 64，动作输出为 B x 16 x 32。图模型使用前 20 个状态分量及 16 x 12 的有效归一化动作分量 [5]。

候选零保留原始动作头输出；其余三个候选使用同一份缓存主干输出，通过分别设定随机流的动作头调用生成。缓存发生在动作头可能修改其输入之前。首次请求恢复随机状态并重放缓存动作头，与原始输出比较，最大绝对误差要求不超过 1e-6。主干不会重复计算四次。每个候选均采用四步流匹配积分。额外候选的随机种子由记录的任务、请求上下文和候选身份确定。

#### 3.2 动作条件图动力学

16 个特征节点与 1 个状态节点投影至宽度 384，叠加可学习节点嵌入。两层 MLP 将展平后的 192 维动作块编码为宽度 384 的控制向量 u。四个图计算块执行有向全连接消息传递，屏蔽节点对自身的消息。节点数 N = 17，更新为：

```text
u = action_MLP(vec(A))
m_i = (1 / 16) sum_{j != i} message_MLP([h_i, h_j, u])
h_i_next = LayerNorm(h_i + update_MLP([h_i, m_i]))       (1)
```

消息 MLP 的维度为 1,152 -> 768 -> 384；更新 MLP 为 768 -> 768 -> 384，均使用 SiLU。独立输出头预测残差，并加回输入特征节点和状态。动力学模型 D 共含 10,237,204 个参数，根据当前表示及完整候选动作块预测 t + 16 的状态。

#### 3.3 预测状态评分与选择

对于候选 k，对预测特征节点取均值，再与预测机器人状态拼接，形成 2,068 维输入。使用保存的训练集均值和尺度归一化，再由 2,068 -> 128 -> 1 的 SiLU MLP 生成奖励 logit：

```text
(F_hat_k, s_hat_k) = D(F, s, A_k)
z_k = reward_MLP(normalize([mean_nodes(F_hat_k), s_hat_k]))
k_star = argmax_k z_k,  k in {0, 1, 2, 3}              (2)
```

完全相同的分数选择索引最小的候选。评分器有 264,961 个可训练参数，没有显式目标输入。任务信息通过 GR00T 进入候选生成，评分器处理其预测潜在特征和状态。训练目标是对应帧记录的二元奖励，不是折扣回报，也不是经过校准的整回合成功概率。

**算法 1：一次策略请求。** 编码当前观测一次；保留候选零；额外采样三个动作块；预测四组未来特征及状态；评分并取最大 logit；将选中的补齐动作输出经过 GR00T 原有逆变换；执行至多 16 步动作；获取新观测后重复。所有模型权重冻结。环境奖励和观测到的未来状态诊断均不进入选择规则。源代码中的 medoid 计算仅用于诊断，本次评测的决策模式为 learned_reward。

<!-- PAGEBREAK -->

### 4. 训练与数据

训练分为三个独立阶段。图动力学和奖励训练不更新 GR00T，S42 评测期间不更新任何模型。公开配置和模型实现固定在 [5,6]；公开资源见第 8 节。

| 阶段 | 数据与监督 | 训练组件 | 训练规模 |
| --- | --- | --- | --- |
| R5 策略 | Human300；30,000 条示范；专家动作 | GR00T 投影层与动作头 | 120,000 次更新；全局批量 128 |
| W5 动力学 | 108,000 训练 / 12,000 验证转移对 | 仅图动力学 | 10 轮；33,750 次更新；批量 32 |
| S38 奖励 | 216,000 训练 / 24,000 验证观测状态 | 奖励 MLP；独立的仅状态诊断头 | 5 轮；批量 512 |

#### 4.1 GR00T-N1.5 策略微调（R5）

R5 从 nvidia/GR00T-N1.5-3B 的 869830fc749c35f34771aa5209f923ac57e4564e 版本开始训练，使用 RoboCasa GR00T 源码 9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10。数据选择为 pretrain_human300，split 为 pretrain，过滤键为 100_demos：300 个任务各 100 条示范。数据配置为 panda_omron，embodiment 为 new_embodiment。冻结语言与视觉主干，训练投影层和动作模型。

使用 16 张 GPU，每设备批量 2、梯度累积 4，全局批量 128。优化器 AdamW：学习率 3e-5、权重衰减 1e-5、betas=(0.95,0.999)、epsilon=1e-8、梯度范数上限 1.0、余弦调度、6,000 步预热。本阶段开启 BF16 与 TF32。最终权重为第 120,000 步。全局批量乘更新次数对应 15,360,000 次样本呈现，并非同等数量的不同示范。

继承的流匹配目标将专家动作 A 与高斯噪声 e 插值为 X_t=(1-t)e+tA，动作头对速度 A-e 进行有效动作维度上的掩码均方误差学习。推理从噪声开始，以 1/4 步长执行四次 Euler 更新。配置名 tune_diffusion_model 指向这一动作模型，不能据名称将其误写为另一个噪声预测或 DDIM 目标。

#### 4.2 LY-GWM 图动力学训练（W5）

下游模型拟合时，每个任务的 100 条示范划为 90 条训练和 10 条验证。每条示范采样四个跨度为 16 步的转移对，得到 108,000 个训练对和 12,000 个验证对。示范长度为 L 时，采样位置为 floor(i(L-17)/3)，i=0,1,2,3；未来位置为 t+16。监督数据包含冻结 R5 特征、20 维状态及示范的 16 x 12 动作。该划分仅对 W5 和 S38 保留验证数据；R5 已使用全部 30,000 条示范。

W5 的损失是平均特征 MSE 与平均状态 MSE 等权相加。模型从新初始化参数开始，残差输出层初始化为零。使用 AdamW（权重衰减 1e-4、梯度范数上限 1.0），余弦学习率从 1e-4 降至 1e-6，200 步预热、种子 4892026、FP32 且禁用 TF32、最多 10 轮、早停耐心值 3。实际完成全部 33,750 次更新，按验证损失选择第 10 轮。

#### 4.3 状态奖励评分器训练（S38）

S38 使用缓存转移对中的当前和未来观测状态，标签为对应帧记录的二元奖励。输入按训练集均值与总体标准差归一化，尺度下限为 0.05。另训练 20 -> 64 -> 1 的仅状态头作为离线诊断。两个 BCE 损失相加，以同一 AdamW 更新组合参数：学习率 3e-4、权重衰减 1e-4、组合梯度范数上限 1.0、种子 4892038。输出偏置以训练集奖励先验的 logit 初始化。按观测验证集 BCE 选择权重。W5 在 12,000 个验证示范动作对上的预测仅用于最后诊断，不参与评分器训练或权重选择。这些细节已对照作者存档的 S38 训练 worker 核实，其 SHA-256 与公开配置一致（附录 B）。

<!-- PAGEBREAK -->

### 5. 本地评测协议与结果

#### 5.1 基准设置

S42 使用 RoboCasa 1.0.1、pretrain 厨房划分，覆盖 18 个 Atomic-Seen、16 个 Composite-Seen、16 个 Composite-Unseen 任务，每任务 50 回合，三个分母分别为 900、800、800。官方基准在预训练厨房中评测 50 个目标任务；“目标任务”不等同于名为 target 的环境划分 [1,7]。逐任务最大步数列于附录 A。

每任务使用五个环境槽位。构造种子为 489000 + 1000 x task_index + episode_index，策略进程种子为 489 + task_index。动作块长度 16，动作头积分步数 4。成功判定为 worker 对每个动作块结束时 info.success 的逻辑或。按照记录的任务步数上限与完成逻辑运行，不以仅成功回合或仅动作输入未越界回合代替完整分母。

原评测使用 NVIDIA A100 40GB、Linux/Python 3.10、PyTorch 2.5.1+cu121 与 EGL 渲染。策略环境包含 NumPy 1.26.4、Transformers 4.51.3；独立仿真环境使用 NumPy 2.2.5 和 RoboCasa 1.0.1。推理禁用 TF32。发布包保留观察到的环境记录，但并非已经跨架构验证的通用安装锁定文件 [5]。

#### 5.2 与自训练基线比较

| 组别 | 历史 S31 基线 | S42 联合策略 | 成功数差值 | 成功率差值（百分点） |
| --- | --- | --- | --- | --- |
| Atomic-Seen | 141/900 (15.6667%) | 150/900 (16.6667%) | +9 | +1.000 |
| Composite-Seen | 25/800 (3.125%) | 38/800 (4.750%) | +13 | +1.625 |
| Composite-Unseen | 15/800 (1.875%) | 19/800 (2.375%) | +4 | +0.500 |
| 总体 | 181/2,500 (7.24%) | 207/2,500 (8.28%) | +26 | +1.040 |

**表 1。** 两个策略均包含全部 2,500 回合。总体按回合数加权，在本设置中等价于 50 个任务成功率的平均：(18 r_atomic + 16 r_seen + 16 r_unseen)/50。“pp”表示百分点。较高的分组汇总不代表每个任务都提高。

S31 与 S42 的任务清单和记录的检查点路径一致。导出的回合标识在任务、构造器 seed、步数上限和动作块长度上也一致。此比较以历史检查点路径及记录的元数据说明来源，并报告这些记录设置下观察到的结果。

#### 5.3 动力学离线验证

| 12,000 个 W5 留出转移对上的误差 | 保持当前值 | W5 预测 |
| --- | --- | --- |
| 特征 MSE | 0.02689410 | 0.01713613 |
| 状态 MSE | 0.01001973 | 0.00071013 |
| 两项均值之和 | 0.03691383 | 0.01784625 |

表 2. 持续性基线将当前特征与状态直接复制到未来。验证数据用于 W5 检查点选择，包含示范动作。两项误差度量量化了这一离线示范分布上的未来特征与机器人状态预测。

<!-- PAGEBREAK -->

#### 5.4 官方独立复测

在第 5.1—5.3 节所述本地实验之后，RoboCasa 维护者独立复测了提交系统。维护者于 2026 年 9 月 28 日在榜单 PR #19 中公布复现的逐任务成功率和分组成功次数。该提交于 2026 年 9 月 30 日合并，并以 LY-GWM-RoboCasa 名称、SEIN-LYGWM 提交者身份出现在官方榜单 [9]。

官方报告涵盖相同的 50 个任务名称，包括 18 个 Atomic-Seen、16 个 Composite-Seen 和 16 个 Composite-Unseen 任务。三组分别报告 540、480 和 480 回合。这些总数与公布的逐任务成功率对应于每任务 30 回合，共计 1,500 回合，与本地评测每任务 50 回合不同。

| 组别 | 任务数 | 成功次数 / 回合数 | 榜单成功率 |
| --- | --- | --- | --- |
| Atomic-Seen | 18 | 261 / 540 | 48.3% |
| Composite-Seen | 16 | 78 / 480 | 16.3% |
| Composite-Unseen | 16 | 15 / 480 | 3.1% |
| 总体 | 50 | 354 / 1,500 | 23.6% |

表 1a. RoboCasa 官方复测 [9]。分组百分比沿用官方榜单的一位小数显示。总体按精确成功次数计算：(261 + 78 + 15) / (540 + 480 + 480) = 354/1,500 = 23.6%。

#### 5.4.1 与本地实验的关系

本地 S31 基线（181/2,500；7.24%）和 S42 联合策略（207/2,500；8.28%）仍是表 1 报告的历史比较。+1.04 个百分点的差值、配对结果分析、bootstrap 区间和动作输入诊断仅适用于这些本地记录。S30 公开检查点参考（212/2,500；8.48%）仍是 target 厨房划分上的独立本地评测，见第 7 节。

官方 23.6% 是独立复测结果，不替换本地实验的分母，也不是新的本地消融实验。它与 8.28% 的差异不得解释为模型更新带来的改进，或图动力学模型的独立贡献。公开复测评论未说明全部运行设置，因此不能声称存档的本地 seed、步数上限、依赖版本和诊断计数描述了官方运行。

#### 5.4.2 官方逐任务结果与来源

附录 A.1 转录维护者评论中的全部 50 项任务成功率。附录 A 保留原始本地成功次数、策略比较和记录的步数上限。官方来源与榜单见 [9]。

<!-- PAGEBREAK -->

### 6. 回合结果与动作选择分析

#### 6.1 探索性配对结果统计

本报告按任务索引和回合索引连接公开 S31 结果表与 S42 回合记录，并检查种子、步数上限、任务分组和动作块长度。分析于 2026 年 9 月 25 日补充，不是新增仿真评测，也不是预先注册的实验检验。

| 组别 | 两者成功 | 仅 S31 成功 | 仅 S42 成功 | 两者失败 |
| --- | --- | --- | --- | --- |
| Atomic-Seen | 93 | 48 | 57 | 702 |
| Composite-Seen | 7 | 18 | 31 | 744 |
| Composite-Unseen | 5 | 10 | 14 | 771 |
| 总体 | 105 | 76 | 102 | 2,217 |

**表 3。** 净增 26 次成功等于新增 102 次、损失 76 次。逐任务比较中，20 个任务成功数增加、13 个减少、17 个不变（附录 A）。构造种子匹配可用于识别记录，但不同策略采取不同动作后，后续状态不会因此保持一致。

在“不一致配对独立”的零假设下，对 178 个不一致配对进行双侧精确二项计算得到 p = 0.06065。这是描述性计算：任务内聚类与历史设计限制其解释。另一个探索性百分位 bootstrap 在每个组内重采样任务级成功次数差值，保留 18/16/16 个任务，使用 20,000 次重复和 NumPy default_rng seed 20260925。其 95% 区间为 [0.00, 2.12] 个百分点。这是在重采样任务组合，而非重复策略 seed。这些计算概括记录的历史比较中的不确定性。补充材料提供精确计算脚本和派生计数。

#### 6.2 选择确实改变环境输入的证据

| 组别 | 动作块总数 | 选中非首个候选 | 环境输入变化块数 |
| --- | --- | --- | --- |
| Atomic-Seen | 37,550 | 28,223 | 28,223 |
| Composite-Seen | 116,100 | 86,862 | 86,862 |
| Composite-Unseen | 130,150 | 97,615 | 97,615 |
| 总体 | 283,800 | 212,700 | 212,700 |

**表 4。** 变化相对于同一次策略请求中的候选零计算，指传入环境的动作块，不是控制器处理后的电机指令。记录的审计还显示全部 283,800 个动作块存在预测和分数差异。原子与复合任务均启用选择器，代码不会因任务所属分组而将其关闭。

非首候选选择比例为 74.947%。源代码与选择/执行报告共同记录了实际运行的候选选择。动作块计数描述机制的使用；任务成功则在表 1 和表 3 中按回合单独测量。

#### 6.3 结果汇总与动作输入诊断

报告的总数采用 S42_ACCEPTANCE_V2_BOUNDS_DIAGNOSTIC_ONLY。该汇总规则将动作边界计数作为诊断，并保留回合、报告与证据检查。规则在 rollout 后应用，替换原始零计数验收条件，但不改变记录的动作、seed、奖励或结果。原始汇总器与修订差异仍可获取。

记录包含被选动作块中 116,968 个越界标量元素，以及已消费输入前缀中 116,292 个越界标量元素，涉及 1,822 回合和全部 50 个任务，其中包括 149 个成功回合。这些是环境输入测量。工作进程采用原始动作逆变换，没有额外裁剪或反归一化，报告的成绩包含全部 2,500 回合。

<!-- PAGEBREAK -->

### 7. 公开检查点参考结果

#### 7.1 评测设置与记录结果

历史 S30 对公开 GR00T-N1.5 检查点的本地评测记录为 212/2,500 次成功（8.48%）。存档任务清单明确设定 split = target。相比之下，S31 与 S42 使用 split = pretrain，官方多任务榜单在预训练厨房中评测 [7]。因此，公开检查点结果作为单独参考报告，并明确标注其划分。

| 历史 S30，target 划分 | Atomic-Seen | Composite-Seen | Composite-Unseen | 总体 |
| --- | --- | --- | --- | --- |
| 成功次数 / 回合数 | 162/900 | 36/800 | 14/800 | 212/2,500 |
| 成功率 | 18.00% | 4.50% | 1.75% | 8.48% |

**表 5。** 这是独立的历史参考，不能作为 S42 的同协议基线。分组名称描述任务身份，不能消除厨房划分差异。作者提供的原始只读审计记录标明了附录 B 中的 S30 协议摘要。

截至 2026 年 9 月 25 日核查，官方榜单显示 GR00T N1.5 总体为 23.9%（三组分别为 50.7%、14.8%、2.7%），并注明相对论文结果使用了 1.5 倍更长步数上限重新评测 [7]。

### 8. 公开资产与复现资源

公开发布提供推理实现、评审入口、GR00T 上游训练源代码、阶段配置、策略与动力学训练摘要、推理权重、回合记录、逐任务审计摘要及记录的轨迹哈希 [5,6]。这些材料将方法描述与固定实现及结果记录联系起来。

训练文档涵盖策略微调（R5）、图动力学训练（W5）和奖励评分器训练（S38）。公开执行入口主要用于推理与评测；完整重训还需使用定制数据准备与训练编排、缓存及奖励标签。新评测需提供兼容的本地环境与仿真器资产，见第 9 节。历史核验使用导出的回合记录和审计摘要，新评测则生成自身完整的选择/执行轨迹。

<!-- PAGEBREAK -->

### 9. 复现步骤

请使用以下固定代码与模型版本，避免假定可变的 main 分支等同于原评测系统。完整链接及来源映射见附录 B-C。

| 组件 | 不可变版本 |
| --- | --- |
| 评测代码 | 79b247c854df116614d9ec5db29406c329a18603 |
| Hugging Face 推理发布包 | 180883d9af9ee02edc3c53fe20babdc7658d1127 |
| 上游 GR00T 训练源码 | 9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10 |
| NVIDIA 基础模型 | 869830fc749c35f34771aa5209f923ac57e4564e |

**第一步：核验导出记录。** 在完整的固定版本仓库中运行：

```text
python3 -B reviewer_s42/verify_historical_results.py
```

本报告编写时已对下载的固定代码快照执行该检查，验证了导出文件哈希、全部 2,500 条回合记录、207 次成功、回执摘要及记录的动作证据总数。这是针对导出结果的 CPU 记录完整性检查。新 GPU 评测按照以下准备与执行流程进行。

**第二步：准备新运行。** 按 reviewer_s42/README.md 提供已有策略环境、仿真环境、固定模型快照，以及 RoboCasa/robosuite/robomimic 源码与仿真资产。下列路径是评审机器安装路径的明确占位示例：

```text
python3 -B reviewer_s42/prepare_run.py \
  --weights /absolute/weights_snapshot \
  --policy-env /absolute/policy_env \
  --sim-env /absolute/simulation_env \
  --third-party /absolute/third_party \
  --output /absolute/new_s42_review_run
```

准备程序检查源码、配置和权重哈希，仅迁移路径，并验证 worker 的其余语法树和协议内容不变。新输出目录必须在代码仓库之外且此前不存在。路径迁移产生新的协议摘要，同时保留原摘要，不会复制历史回合成绩。

**第三步：在已分配的 GPU 上运行所有任务并汇总。** 以下第一条命令只运行任务 0；完整评测须覆盖索引 0 至 49。脚本不负责分配 GPU 或提交调度作业。

```text
python3 -B reviewer_s42/run_task.py \
  --run /absolute/new_s42_review_run --task-index 0
python3 -B reviewer_s42/collect_results.py \
  --run /absolute/new_s42_review_run
```

新运行汇总器核验完整覆盖及本次新生成的选择/执行轨迹。不支持任务内部分续跑，也不支持混合不同作业的回合。不得用 -O 禁用 Python 断言。存档集群作业脚本包含历史路径，不是评审运行入口。新运行可能得到不同成功数。

### 10. 结论

LY-GWM-RoboCasa-Human300 在经过 Human300 微调的 GR00T-N1.5 之上实现了显式预测引导动作选择。S42 中预测用于选择实际执行的动作，原子任务同样如此。本地记录结果为 207/2,500 次成功，相比 S31 的历史差值为正，净增 26 次成功。另外，官方复测记录为 354/1,500 次成功（23.6%）[9]。本报告使实际训练和推理设计可核查，标明公开检查点评测的划分，并将结果与可复现记录关联。公开检查点与评审入口支持在独立基准环境中评测该系统。

<!-- PAGEBREAK -->

### 附录 A. 本地逐任务结果与步数上限

每行每个策略包含 50 回合。H 为记录的最大环境步数上限。计数为成功次数，不是百分比。S31 表示基线策略评测，S42 表示联合策略评测；差值为联合策略减去基线。索引与 seed 公式及评审入口一致。A = Atomic-Seen；C = Composite-Seen；U = Composite-Unseen。

| 索引 | 任务 | 组别 | H | S31 | S42 | 差值 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | CloseBlenderLid | A | 900 | 1 | 1 | +0 |
| 1 | CloseFridge | A | 900 | 10 | 11 | +1 |
| 2 | CloseToasterOvenDoor | A | 450 | 4 | 6 | +2 |
| 3 | CoffeeSetupMug | A | 600 | 1 | 1 | +0 |
| 4 | NavigateKitchen | A | 450 | 3 | 2 | -1 |
| 5 | OpenCabinet | A | 1050 | 12 | 17 | +5 |
| 6 | OpenDrawer | A | 750 | 10 | 10 | +0 |
| 7 | OpenStandMixerHead | A | 450 | 11 | 16 | +5 |
| 8 | PickPlaceCounterToCabinet | A | 750 | 11 | 14 | +3 |
| 9 | PickPlaceCounterToStove | A | 600 | 16 | 11 | -5 |
| 10 | PickPlaceDrawerToCounter | A | 750 | 8 | 11 | +3 |
| 11 | PickPlaceSinkToCounter | A | 900 | 10 | 10 | +0 |
| 12 | PickPlaceToasterToCounter | A | 600 | 15 | 13 | -2 |
| 13 | SlideDishwasherRack | A | 450 | 7 | 8 | +1 |
| 14 | TurnOffStove | A | 750 | 3 | 0 | -3 |
| 15 | TurnOnElectricKettle | A | 450 | 6 | 9 | +3 |
| 16 | TurnOnMicrowave | A | 450 | 3 | 2 | -1 |
| 17 | TurnOnSinkFaucet | A | 600 | 10 | 8 | -2 |
| 18 | DeliverStraw | C | 2550 | 0 | 0 | +0 |
| 19 | GetToastedBread | C | 3000 | 0 | 0 | +0 |
| 20 | KettleBoiling | C | 1500 | 1 | 3 | +2 |
| 21 | LoadDishwasher | C | 1800 | 3 | 1 | -2 |
| 22 | PackIdenticalLunches | C | 3900 | 0 | 5 | +5 |
| 23 | PreSoakPan | C | 2400 | 3 | 5 | +2 |
| 24 | PrepareCoffee | C | 1800 | 0 | 0 | +0 |
| 25 | RinseSinkBasin | C | 1350 | 3 | 6 | +3 |
| 26 | ScrubCuttingBoard | C | 1200 | 4 | 5 | +1 |
| 27 | SearingMeat | C | 4350 | 1 | 0 | -1 |
| 28 | SetUpCuttingStation | C | 2400 | 2 | 1 | -1 |
| 29 | StackBowlsCabinet | C | 2100 | 1 | 2 | +1 |
| 30 | SteamInMicrowave | C | 2100 | 0 | 0 | +0 |
| 31 | StirVegetables | C | 2400 | 0 | 3 | +3 |
| 32 | StoreLeftoversInBowl | C | 2550 | 6 | 5 | -1 |
| 33 | WashLettuce | C | 1650 | 1 | 2 | +1 |
| 34 | ArrangeBreadBasket | U | 4350 | 1 | 0 | -1 |
| 35 | ArrangeTea | U | 2250 | 1 | 0 | -1 |
| 36 | BreadSelection | U | 1950 | 2 | 5 | +3 |
| 37 | CategorizeCondiments | U | 1650 | 0 | 1 | +1 |
| 38 | CuttingToolSelection | U | 1200 | 0 | 0 | +0 |
| 39 | GarnishPancake | U | 2700 | 4 | 3 | -1 |
| 40 | GatherTableware | U | 2250 | 0 | 0 | +0 |
| 41 | HeatKebabSandwich | U | 2700 | 0 | 0 | +0 |
| 42 | MakeIceLemonade | U | 3000 | 0 | 0 | +0 |
| 43 | PanTransfer | U | 1800 | 0 | 0 | +0 |
| 44 | PortionHotDogs | U | 2250 | 0 | 2 | +2 |
| 45 | RecycleBottlesByType | U | 2850 | 4 | 4 | +0 |
| 46 | SeparateFreezerRack | U | 2400 | 0 | 0 | +0 |
| 47 | WaffleReheat | U | 4050 | 1 | 2 | +1 |
| 48 | WashFruitColander | U | 3150 | 2 | 2 | +0 |
| 49 | WeighIngredients | U | 3000 | 0 | 0 | +0 |
| 总体 | 50 个任务 × 50 回合 | A/C/U | - | 181 | 207 | +26 |

<!-- PAGEBREAK -->

### 附录 A.1. 官方复测逐任务结果

以下成功率转录自 RoboCasa 维护者的复测 [9]。分组总回合数对应每任务 30 回合。A = Atomic-Seen；C = Composite-Seen；U = Composite-Unseen。索引对应本地附录中的任务名称；不会用本地步数上限或成功次数替代官方测量。

| 索引 | 任务 | 组别 | 回合数 | 官方成功率 |
| --- | --- | --- | --- | --- |
| 0 | CloseBlenderLid | A | 30 | 13.33% |
| 1 | CloseFridge | A | 30 | 53.33% |
| 2 | CloseToasterOvenDoor | A | 30 | 36.67% |
| 3 | CoffeeSetupMug | A | 30 | 16.67% |
| 4 | NavigateKitchen | A | 30 | 26.67% |
| 5 | OpenCabinet | A | 30 | 73.33% |
| 6 | OpenDrawer | A | 30 | 73.33% |
| 7 | OpenStandMixerHead | A | 30 | 80.00% |
| 8 | PickPlaceCounterToCabinet | A | 30 | 70.00% |
| 9 | PickPlaceCounterToStove | A | 30 | 66.67% |
| 10 | PickPlaceDrawerToCounter | A | 30 | 60.00% |
| 11 | PickPlaceSinkToCounter | A | 30 | 96.67% |
| 12 | PickPlaceToasterToCounter | A | 30 | 36.67% |
| 13 | SlideDishwasherRack | A | 30 | 56.67% |
| 14 | TurnOffStove | A | 30 | 13.33% |
| 15 | TurnOnElectricKettle | A | 30 | 23.33% |
| 16 | TurnOnMicrowave | A | 30 | 40.00% |
| 17 | TurnOnSinkFaucet | A | 30 | 33.33% |
| 18 | DeliverStraw | C | 30 | 3.33% |
| 19 | GetToastedBread | C | 30 | 0.00% |
| 20 | KettleBoiling | C | 30 | 23.33% |
| 21 | LoadDishwasher | C | 30 | 23.33% |
| 22 | PackIdenticalLunches | C | 30 | 10.00% |
| 23 | PreSoakPan | C | 30 | 40.00% |
| 24 | PrepareCoffee | C | 30 | 0.00% |
| 25 | RinseSinkBasin | C | 30 | 30.00% |
| 26 | ScrubCuttingBoard | C | 30 | 46.67% |
| 27 | SearingMeat | C | 30 | 6.67% |
| 28 | SetUpCuttingStation | C | 30 | 13.33% |
| 29 | StackBowlsCabinet | C | 30 | 3.33% |
| 30 | SteamInMicrowave | C | 30 | 6.67% |
| 31 | StirVegetables | C | 30 | 10.00% |
| 32 | StoreLeftoversInBowl | C | 30 | 13.33% |
| 33 | WashLettuce | C | 30 | 30.00% |
| 34 | ArrangeBreadBasket | U | 30 | 0.00% |
| 35 | ArrangeTea | U | 30 | 0.00% |
| 36 | BreadSelection | U | 30 | 6.67% |
| 37 | CategorizeCondiments | U | 30 | 0.00% |
| 38 | CuttingToolSelection | U | 30 | 0.00% |
| 39 | GarnishPancake | U | 30 | 13.33% |
| 40 | GatherTableware | U | 30 | 0.00% |
| 41 | HeatKebabSandwich | U | 30 | 0.00% |
| 42 | MakeIceLemonade | U | 30 | 3.33% |
| 43 | PanTransfer | U | 30 | 0.00% |
| 44 | PortionHotDogs | U | 30 | 0.00% |
| 45 | RecycleBottlesByType | U | 30 | 10.00% |
| 46 | SeparateFreezerRack | U | 30 | 0.00% |
| 47 | WaffleReheat | U | 30 | 6.67% |
| 48 | WashFruitColander | U | 30 | 3.33% |
| 49 | WeighIngredients | U | 30 | 6.67% |
| 总体 | 50 个任务；354 次成功 | A/C/U | 1,500 | 23.6% |

<!-- PAGEBREAK -->

### 附录 B. 来源、摘要与代码映射

补充材料包含逐任务 CSV、配对计数、bootstrap 设置、记录核验结果、去除私有路径的 S30 来源摘要，以及从固定仓库重算新增分析的脚本。下列摘要保留其原始含义：文件字节 SHA-256 与规范化 JSON 对象 SHA-256 不能混用。

| 对象 / 摘要种类 | SHA-256 |
| --- | --- |
| S42 协议 / 规范化 JSON | 4bdd5abfabe2674caba78c1f7b637f3c141d5550a09b55e065020c8a9c069e71 |
| S31 协议 / 记录的摘要 | ee15b8a67247a3f37f6e137c222e0b79cc1801c6d0ab539ec6db247e32bbc943 |
| S30 协议 / 记录的摘要 | 7af171be0a1bafad91e6ffe5a90fe2032a89f8e1428ce4f9f6e59cb27f9bed2e |
| R5 权重分片 1 / 文件字节 | a734b821ddf1a145536baeeb5b144e0e8435f1746ee6aced523d2ff0a19d5d4a |
| R5 权重分片 2 / 文件字节 | 0b713131a2e4627710a57fdf65887f44629d505e15ec66df18ac32a47c7485d3 |
| W5 权重 / 文件字节 | a964d19a0078cc6282d5da79f06f7d7ead3137799b8f6e969ccff5e0643be9f1 |
| S38 权重 / 文件字节 | 5f9ad4b52eddfc2ca3f1ac06efcae3f20a02a764857987fe447810d5cde68a8b |
| W5 存档 worker / 文件字节 | aa6bb5eece2eca77b2a23625a138336ef02132852f24e9498882ffa1fb4d1a2c |
| S38 存档 worker / 文件字节 | 52ef967efff2747bb11e9a0d34fa8a76a212203d87bcdb431ff04dfbecbe78b3 |

| 说明内容 | 固定评测仓库内的来源 |
| --- | --- |
| R5 数据、微调及训练计划 | evaluated_snapshot/gr00t_training/formal_config.json; formal_results.json |
| W5 架构与验证结果 | evaluated_snapshot/w5_training/graph_model.py; config.json; training_results.json |
| 评分器输入、目标与来源 | evaluated_snapshot/s42/s42_original/scorer-training-config.json; full/learned_reward/reward_model.py |
| 候选采样与执行 | evaluated_snapshot/s42/s42_original/full/learned_reward/decision_adapter.py; selection_rule.py; worker.py |
| 协议与逐任务步数上限 | evaluated_snapshot/s42/s42_original/full/learned_reward/protocol.json |
| 修订结果与回执 | evaluated_snapshot/s42/acceptance_v2/overall_results.json; per_task_results.json; evaluation_receipt.json |
| 历史基线回合 | evaluated_snapshot/joint_s32/baseline_episode_outcomes.json |
| 新运行与记录核验 | reviewer_s42/README.md; verify_historical_results.py; prepare_run.py; run_task.py; collect_results.py |

**作者存档证据。** W5 和 S38 训练 worker 均从作者安装脚本中恢复，其字节摘要与公开配置匹配，从而核对第 4.2 节的优化器与调度，以及第 4.3 节的归一化和 BCE 实现。固定推理仓库不包含这些完整训练 worker。S30 的划分与结果来自作者 2026 年 9 月 23 日的只读评测审计。本文区分这些作者存档与公开下载文件；补充材料提供相关非私有事实及记录摘要。

**组件归属。** NVIDIA 提供 GR00T 基础模型，RoboCasa 团队提供基准资产与 GR00T 适配。我们训练 R5 微调策略、W5 动力学模型与 S38 评分器；S42 期间三者全部冻结。访问及复用遵循各仓库保留的组件声明和条款，本文不替代这些条款。

<!-- PAGEBREAK -->

### 附录 C. 参考文献

[1] Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, Yuke Zhu. *RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots.* ICLR 2026. arXiv:2603.04356. https://arxiv.org/abs/2603.04356

[2] NVIDIA et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.* 2025. arXiv:2503.14734。该文为基础模型论文；本系统采用 [3] 的 N1.5 发布版。https://arxiv.org/abs/2503.14734

[3] NVIDIA. *GR00T-N1.5-3B*，基础模型版本 869830fc749c35f34771aa5209f923ac57e4564e。https://huggingface.co/nvidia/GR00T-N1.5-3B/tree/869830fc749c35f34771aa5209f923ac57e4564e

[4] Tianliang Ma. *Why world models fail under intervention: Ontological-causal separation as a necessary structure.* Research Square 预印本，第 1 版，2025 年 12 月 19 日。DOI:10.21203/rs.3.rs-8377767/v1。作为概念基础，本文另行说明具体 RoboCasa 实现与结果。https://doi.org/10.21203/rs.3.rs-8377767/v1

[5] SEIN-LYGWM. *LY-GWM-RoboCasa-Human300-eval*，评测代码与记录发布包，固定版本 79b247c854df116614d9ec5db29406c329a18603。https://github.com/SEIN-LYGWM/LY-GWM-RoboCasa-Human300-eval/tree/79b247c854df116614d9ec5db29406c329a18603

[6] lygwm-review. *LY-GWM-RoboCasa-Human300*，推理权重发布包，固定版本 180883d9af9ee02edc3c53fe20babdc7658d1127。https://huggingface.co/lygwm-review/LY-GWM-RoboCasa-Human300/tree/180883d9af9ee02edc3c53fe20babdc7658d1127

[7] RoboCasa Team. *RoboCasa365 leaderboard, submission requirements, and submission schema.* 网站核对日期为 2026 年 9 月 25 日；仓库版本 a228bd4b724fd5c7649802a5d2829ea1893da0ec。https://robocasa.ai/leaderboard.html ; https://github.com/robocasa-benchmark/leaderboard/tree/a228bd4b724fd5c7649802a5d2829ea1893da0ec

[8] Yang Zhang, Jiangyuan Zhao, Chenyou Fan, Fangzheng Yan, et al. *PRTS: A Primitive Reasoning and Tasking System via Contrastive Representations.* 2026. arXiv:2604.27472。参考其公开模型说明结构及方法背景。https://arxiv.org/abs/2604.27472 ; https://github.com/TeleHuman/PRTS

[9] RoboCasa Team / sepnasiriany. LY-GWM-RoboCasa 官方复测，榜单 PR #19，结果公布于 2026 年 9 月 28 日；2026 年 9 月 30 日合并。官方榜单条目及逐任务评测结果；访问日期：2026 年 10 月 10 日。

https://github.com/robocasa-benchmark/leaderboard/pull/19#issuecomment-5863555620

https://robocasa.ai/leaderboard.html

### 数据与代码可用性

固定代码与权重见 [5,6]。评审指南为 [5] 中的 reviewer_s42/README.md。随本报告发布的统计补充材料增加了对现有导出结果的分析，没有新增 GPU 实验。概念预印本 [4] 与本实现报告用途不同，描述提交系统时应将两者关联引用。
