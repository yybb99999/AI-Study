# D5–D8｜Qwen 大模型基础理论收尾总结

## 一、D5：Decoder-only Transformer 内部结构

D5 的核心目标是理解：`input_ids` 进入 Qwen 之后，中间发生了什么。

整体结构：

```text
input_ids
↓
Token Embedding
↓
Transformer Block × N
↓
Final Norm
↓
LM Head
↓
Vocabulary Logits
```

一个 Transformer Block 可以概括为：

```text
Input
↓
Normalization
↓
Self-Attention
↓
Residual Connection
↓
Normalization
↓
MLP
↓
Residual Connection
↓
Output
```

### 1. Embedding

Tokenizer 将文本转换为 Token ID，但 Token ID 只是整数，不能直接表示语义，因此需要经过 Embedding：

```text
Text
↓
Tokenizer
↓
Token ID
↓
Embedding Layer
↓
高维向量
```

因此：

```text
Tokenizer = Text → Integer
Embedding = Integer → Vector
```

Embedding 之后得到的向量才真正进入 Transformer。

### 2. Self-Attention

Self-Attention 的核心作用是：

> 让当前 Token 根据上下文中其他相关 Token 的信息更新自己的表示。

它解决的是：

```text
当前 Token 应该关注上下文中的哪些 Token？
```

### 3. Q、K、V

输入 Hidden State 记为 `X`，通过不同线性映射得到：

```text
Q = XW_Q
K = XW_K
V = XW_V
```

可以建立如下直觉：

```text
Query = 当前 Token 想寻找什么信息
Key   = 每个 Token 提供什么匹配特征
Value = 匹配后真正传递什么内容
```

### 4. Attention 核心公式

```text
Attention(Q,K,V)
=
softmax(QK^T / sqrt(d_k)) V
```

过程：

```text
Q × K^T
↓
相关性分数
↓
除以 sqrt(d_k)
↓
Softmax
↓
Attention Weights
↓
与 V 加权组合
↓
新的 Token Representation
```

### 5. 为什么除以 sqrt(d_k)

随着 Query / Key 维度增加，`QK^T` 的数值可能变大，导致 Softmax 过度尖锐，从而影响训练稳定性，因此使用 `sqrt(d_k)` 进行缩放。

### 6. Multi-Head Attention

Transformer 不只计算一个 Attention，而是并行使用多个 Attention Head：

```text
Head 1
Head 2
Head 3
...
Head H
↓
Concat
↓
Linear Projection
```

不同 Head 可以学习不同类型的关系，例如语法、语义、实体和长距离依赖。

### 7. q_proj / k_proj / v_proj / o_proj

Qwen 源码中常见：

```text
q_proj → Q
k_proj → K
v_proj → V
o_proj → Attention 输出投影
```

这些层也是后续 LoRA 常见的目标模块。

### 8. Causal Mask

Decoder-only Language Model 不能在预测当前位置时看到未来 Token。

```text
Token 1 → 只能看 Token 1
Token 2 → 可以看 Token 1,2
Token 3 → 可以看 Token 1,2,3
Token 4 → 可以看 Token 1,2,3,4
```

核心目的：

```text
禁止看到未来 Token
```

否则 Next-token Prediction 会发生信息泄露。

### 9. Causal Mask 与 attention_mask

```text
attention_mask
→ 区分真实 Token 与 Padding Token

Causal Mask
→ 区分过去 Token 与未来 Token
```

因此：

```text
Padding Mask ≠ Causal Mask
```

### 10. MLP

```text
Attention
→ 负责 Token 与 Token 之间的信息交互

MLP
→ 负责对每个 Token 的表示进行进一步非线性加工
```

### 11. Residual Connection

基本形式：

```text
y = x + F(x)
```

作用包括保留原始信息、改善梯度传播和提高深层网络训练稳定性。

### 12. Normalization

现代大语言模型经常使用 LayerNorm 或 RMSNorm，其作用是控制 Hidden State 的数值尺度，使深层网络训练更加稳定。

---

## 二、D6：Causal Language Modeling 与训练 Loss

D6 的核心问题是：模型如何通过训练学会预测下一个 Token。

### 1. Causal Language Modeling

假设文本为：

```text
我 喜欢 人工 智能
```

训练目标可以表示为：

```text
我
→ 喜欢

我 喜欢
→ 人工

我 喜欢 人工
→ 智能
```

即：

```text
P(x_t | x_1, x_2, ..., x_{t-1})
```

### 2. 一次 Forward 可以训练多个位置

训练阶段可以一次输入整个序列，并结合 Causal Mask 同时计算多个位置的预测：

```text
位置 1 → 预测位置 2
位置 2 → 预测位置 3
位置 3 → 预测位置 4
```

这使训练阶段能够并行处理序列中的多个预测位置。

### 3. input_ids 与 labels

训练时通常有：

```text
input_ids
labels
```

二者可能来源于同一个 Token 序列，但计算 Loss 时需要 Shift。

例如：

```text
Input:
A B C D

目标：
A → B
B → C
C → D
```

### 4. Shift

概念上：

```text
当前位置的 Logits
↓
预测下一个位置的 Label
```

即：

```text
logits[:-1]
vs
labels[1:]
```

### 5. Cross Entropy

模型在每个位置输出整个 Vocabulary 的 Logits。

如果正确 Token 为 `北京`，Cross Entropy 会推动：

```text
P(北京) ↑
```

可以理解为：

```text
正确 Token 概率高 → Loss 小
正确 Token 概率低 → Loss 大
```

### 6. 完整训练过程

```text
Text
↓
Tokenizer
↓
input_ids
↓
Qwen
↓
Vocabulary Logits
↓
Shift
↓
Cross Entropy
↓
Loss
↓
Backward
↓
Gradients
↓
Optimizer
↓
Parameter Update
```

### 7. Training 与 Inference

训练：

```text
完整训练序列
↓
Causal Mask
↓
并行计算多个位置的 Logits
↓
Cross Entropy
↓
Backward
↓
Update
```

推理：

```text
Prompt
↓
预测一个 Next Token
↓
Append
↓
再次 Forward
↓
预测下一个 Token
↓
...
```

因此：

> 训练阶段可以并行计算多个 Token 位置，而自回归推理阶段通常需要逐 Token 生成。

### 8. Teacher Forcing

训练阶段通常使用真实历史 Token 作为当前预测位置之前的上下文：

```text
Training  → Ground-truth History
Inference → Model-generated History
```

这种训练方式可以理解为 Teacher Forcing。

---

## 三、D7：LLM 基础链路整合

完整推理链：

```text
用户输入
↓
messages
↓
Chat Template
↓
Formatted Text
↓
Tokenizer
↓
Token IDs
↓
Embedding
↓
Decoder-only Transformer
↓
Q/K/V Self-Attention
↓
Causal Mask
↓
MLP
↓
Residual
↓
Norm
↓
Hidden State
↓
LM Head
↓
Vocabulary Logits
↓
Temperature / Top-k / Top-p
↓
Greedy / Sampling
↓
Next Token
↓
追加到 Context
↓
继续生成
↓
EOS / max_new_tokens
↓
decode
↓
最终回答
```

### Encoder-only 与 Decoder-only

RoBERTa：

```text
Encoder-only
双向上下文
Masked Language Modeling
Representation Learning
```

Qwen：

```text
Decoder-only
Causal Mask
Next-token Prediction
Autoregressive Generation
```

可以建立简化直觉：

```text
RoBERTa → 更偏向表示与理解
Qwen    → 更偏向自回归生成
```

---

## 四、D8：Instruction Dataset 与 SFT 前置知识

D8 是从基础理论进入后训练项目的过渡。

### 1. Pretraining Data

预训练数据可以只是普通文本：

```text
人工智能是计算机科学的重要研究方向……
```

训练目标依然是：

```text
Next-token Prediction
```

Base Model 能学习语言规律、知识和通用能力，但不一定天然擅长遵循聊天指令。

### 2. Instruction Data

SFT 数据通常形成：

```text
Instruction
→ Response
```

例如：

```json
{
  "instruction": "解释什么是LoRA。",
  "response": "LoRA是一种参数高效微调方法……"
}
```

也可以使用对话格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "解释什么是LoRA。"
    },
    {
      "role": "assistant",
      "content": "LoRA是一种参数高效微调方法……"
    }
  ]
}
```

### 3. SFT 本质仍然是 Causal LM

SFT 并没有完全更换训练目标，本质仍然可以是：

```text
Causal Language Modeling
+
Cross Entropy
```

区别主要来自：

```text
普通预训练文本
↓
高质量 Instruction / Response 数据
```

从而让模型学习：

```text
看到 User Instruction
↓
生成合适的 Assistant Response
```

### 4. Chat Template 与 SFT

```text
messages
↓
Chat Template
↓
Token IDs
↓
input_ids / labels
↓
Causal LM Loss
```

因此推理模板、训练模板和 Tokenizer 配置之间需要保持一致。

### 5. Assistant Loss Masking

训练时可以选择只在 Assistant Token 上计算 Loss：

```text
User Tokens
→ ignore

Assistant Tokens
→ calculate loss
```

常见实现：

```text
User Labels
→ -100

Assistant Labels
→ Token IDs
```

Cross Entropy 通常忽略 Label 为 `-100` 的位置。

这就是 Assistant Loss Masking。

### 6. Dataset Processing Pipeline

```text
Raw Dataset
↓
Cleaning
↓
Format Conversion
↓
messages
↓
Chat Template
↓
Tokenization
↓
Truncation
↓
input_ids
↓
labels
↓
Data Collator
↓
Batch
↓
Trainer
```

### 7. Train / Validation / Test

```text
Train      → Parameter Update
Validation → 调参、观察训练状态与 Overfitting
Test       → 最终性能评估
```

### 8. Epoch

Epoch 表示完整遍历一次训练集。

### 9. Batch Size

Batch Size 表示一次 Forward / Backward 使用多少个训练样本。

通常：

```text
Batch Size 增大
→ 显存占用增大
```

### 10. Gradient Accumulation

当显存不足以直接使用较大 Batch 时，可以通过多步累积梯度得到更大的 Effective Batch Size。

例如：

```text
per_device_batch_size = 2
gradient_accumulation_steps = 8

effective batch size ≈ 16
```

### 11. Learning Rate

Learning Rate 控制每次参数更新的步幅。

```text
过大 → 训练可能不稳定或发散
过小 → 收敛速度可能过慢
```

### 12. Overfitting

典型表现：

```text
Training Loss 持续下降
但 Validation Performance 开始下降
```

因此训练不能只观察 Training Loss，还需要关注 Validation Metrics 和实际 Case。

---

## 五、D5–D8 核心知识速记

```text
Tokenizer
=
Text → Token ID
```

```text
Embedding
=
Token ID → Vector
```

```text
Self-Attention
=
Token 根据上下文更新自己的表示
```

```text
Q = 寻找什么信息
K = 提供什么匹配特征
V = 真正传递什么内容
```

```text
Causal Mask
=
禁止看到未来 Token
```

```text
MLP
=
进一步加工 Token Representation
```

```text
Residual
=
保留原信息并帮助梯度传播
```

```text
Norm
=
稳定 Hidden State 数值尺度
```

```text
LM Head
=
Hidden State → Vocabulary Logits
```

```text
Causal LM
=
根据过去 Token 预测下一个 Token
```

```text
Cross Entropy
=
衡量正确 Token 的预测质量
```

```text
SFT
=
在高质量 Instruction / Response 数据上继续进行监督式 Causal LM 训练
```

---

# 六、D5–D8 最重要的最终闭环

```text
                 ┌─────────────────────┐
                 │      Dataset        │
                 │ Instruction/Response│
                 └──────────┬──────────┘
                            ↓
                       Chat Template
                            ↓
                         Tokenizer
                            ↓
                         input_ids
                            ↓
                        Embedding
                            ↓
              ┌─────────────────────────┐
              │ Decoder-only Transformer│
              │                         │
              │ Q / K / V Attention     │
              │ Causal Mask             │
              │ Residual                │
              │ Norm                    │
              │ MLP                     │
              └────────────┬────────────┘
                           ↓
                      Hidden State
                           ↓
                         LM Head
                           ↓
                         Logits
                    ┌──────┴──────┐
                    ↓             ↓
                 Training       Inference
                    ↓             ↓
               Shift Labels   Temperature
                    ↓         Top-k / Top-p
              Cross Entropy       ↓
                    ↓          Next Token
                  Loss             ↓
                    ↓          Append
                Backward           ↓
                    ↓            Repeat
                Optimizer          ↓
                    ↓             EOS
             Parameter Update
```

## 七、最终理论学习总结

这一阶段的理论学习完成了从自然语言输入到模型训练与推理输出的完整闭环：原始文本首先通过 Chat Template 被整理成符合模型训练格式的对话序列，再由 Tokenizer 转换成 Token IDs，Token IDs 经过 Embedding 映射为连续向量后进入 Decoder-only Transformer；Transformer 内部通过 Q、K、V 构造 Self-Attention，使不同 Token 之间能够进行上下文信息交互，同时使用 Causal Mask 保证当前位置只能访问过去和当前位置的信息，随后通过 MLP 对表示进行非线性加工，并结合 Residual Connection 和 Normalization 保证深层网络的信息传递与训练稳定性；经过多层 Transformer 后得到的 Hidden States 会通过 LM Head 映射到整个 Vocabulary 的 Logits，在训练阶段，这些 Logits 与经过 Shift 的 Labels 通过 Cross Entropy 计算 Loss，再经过 Backward 和 Optimizer 更新模型参数，从而让模型学习 Next-token Prediction，而在推理阶段，Logits 会经过 Temperature、Top-k、Top-p、Greedy 或 Sampling 等解码策略选择下一个 Token，并将新 Token 重新加入 Context 继续自回归生成，直到 EOS 或长度限制触发停止；在此基础上，Instruction Dataset 将普通预训练文本进一步组织成 User Instruction 与 Assistant Response，SFT 仍然基于 Causal Language Modeling 和 Cross Entropy 进行训练，只是通过高质量指令数据、Chat Template 和可选的 Assistant Loss Masking，使模型从“能够预测下一个 Token”进一步转变为“能够按照人类指令生成符合要求的回答”。至此，大语言模型从数据、输入编码、模型内部计算、训练目标到推理解码的基础理论链路已经形成完整认识，后续重点将转向代码实现，通过实际的数据处理、SFT、LoRA、QLoRA、Evaluation 和 DPO 项目进一步理解每一个模块的工程细节和实际行为。
