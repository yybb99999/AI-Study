# D1｜Qwen 大模型基础学习总结与面试复习

## 一、今天学习内容总结

 **Qwen 推理的完整链路** 

今天学习的核心流程是：

```text
用户输入
↓
messages
↓
Chat Template
↓
Tokenizer
↓
input_ids
↓
Qwen Decoder-only Transformer
↓
logits
↓
选择下一个 token
↓
自回归生成
↓
generated token ids
↓
decode
↓
最终文本
```

---

### 1. 环境配置

直接放在requirement文件中了用conda虚拟环境创建。当前暂时先用CPU

### 2. Qwen

今天使用小型 Qwen 模型作为学习对象。

Qwen 可以先理解为：CPU

```text
Qwen
↓
生成式大语言模型
↓
Transformer
↓
Decoder-only
↓
Causal Language Model
```

当前阶段最重要的是理解：

> Qwen 不是一次性生成整个回答，而是不断预测“下一个 token”。

---

### 3. Decoder-only Transformer

Qwen 属于 Decoder-only Transformer。

可以先和过去常见的 RoBERTa 区分：

| 对比项 | RoBERTa | Qwen |
|---|---|---|
| 架构 | Encoder-only | Decoder-only |
| 主要训练目标 | Masked Language Modeling | Causal Language Modeling |
| 上下文方式 | 双向 | 因果、自回归 |
| 典型任务 | 分类、表示学习 | 文本生成、对话、代码生成 |
| 典型输出 | Hidden Representation | Next-token Logits |

今天暂时不深入 Q/K/V、RoPE、GQA、KV Cache，这些后续再学。

---

### 4. Causal Language Modeling

Causal LM 的核心目标是：

> 根据前面的 token，预测下一个 token。

数学表达：

\[
P(x_t \mid x_1,x_2,\ldots,x_{t-1})
\]

例如：

```text
中国的首都是
```

模型会预测：

```text
北京
```

然后把“北京”加入上下文：

```text
中国的首都是北京
```

继续预测下一个 token。

因此整个生成过程本质上是：

```text
Predict
↓
Append
↓
Predict
↓
Append
↓
...
```

这就是“自回归生成”。

---

### 5. messages

聊天模型通常使用结构化对话：

```python
messages = [
    {
        "role": "user",
        "content": "请解释什么是大语言模型。"
    }
]
```

这里的 `messages` 并不是模型真正看到的 Tensor。

它只是一个结构化的聊天历史。

常见 role 包括：

```text
system
user
assistant
```

多轮聊天本质上就是不断扩展这个 messages 列表。

---

### 6. Chat Template

Chat Template 的作用是：

> 把结构化的 `system/user/assistant` 消息转换成模型训练时对应的聊天文本格式。

代码：

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

两个参数今天需要理解：

#### `tokenize=False`

表示：

> 暂时只生成格式化后的字符串，不立即转换成 token IDs。

所以此时：

```python
type(text)
```

通常还是：

```text
str
```

#### `add_generation_prompt=True`

可以理解为：

> 在输入末尾补上“接下来轮到 assistant 回答”的格式提示。

这对 Instruct/Chat 模型非常重要。

---

### 8. Tokenizer

Tokenizer 是文本世界和模型数字世界之间的桥梁。

过程：

```text
文本
↓
Token
↓
Token ID
```

例如：

```python
encoded = tokenizer(
    text,
    return_tensors="pt"
)
```

会产生：

```python
encoded["input_ids"]
```

即整数 Tensor。

需要注意：

> Tokenizer 不是传统意义上简单的“按词分词”。

一个 token 可能是：

- 一个汉字
- 多个汉字
- 一个英文单词
- 英文单词的一部分
- 标点
- 特殊符号

---

### 9. input_ids

`input_ids` 是：

> 输入文本经过 Tokenizer 后，每一个 token 在模型 vocabulary 中对应的整数编号。

例如：

```python
print(model_inputs["input_ids"])
```

可能得到：

```text
tensor([[151644, ..., ...]])
```

模型真正处理的是这些数字，而不是原始中文字符串。

---

### 10. input_ids.shape

假如：

```python
model_inputs["input_ids"].shape
```

得到：

```text
torch.Size([1, 27])
```

含义是：

```text
1  → batch size
27 → sequence length
```

以后训练时如果看到：

```text
[8, 512]
```

就表示：

```text
batch_size = 8
sequence_length = 512
```

---

### 11. model.generate()

生成代码：

```python
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32,
        do_sample=False,
    )
```

目前可以把 `generate()` 理解为：

```text
模型前向计算
↓
得到 next-token logits
↓
根据生成策略选择 token
↓
把 token 加到已有序列
↓
再次执行模型
↓
不断重复
```

直到：

- 达到 `max_new_tokens`
- 生成 EOS
- 或满足其他停止条件

---

### 12. 为什么使用 torch.no_grad()

推理阶段不需要计算梯度。

训练流程是：

```text
Forward
↓
Loss
↓
Backward
↓
Gradient
↓
Optimizer Update
```

而推理只有：

```text
Forward
↓
Generation
```

因此：

```python
with torch.no_grad():
```

可以避免构建梯度计算图，减少额外内存和计算开销。

---

### 13. `**model_inputs`

如果：

```python
model_inputs = {
    "input_ids": ...,
    "attention_mask": ...
}
```

那么：

```python
model.generate(**model_inputs)
```

相当于：

```python
model.generate(
    input_ids=...,
    attention_mask=...
)
```

这是 Python 的字典参数展开语法。

---

### 14. generated_ids

需要特别注意：

```python
generated_ids = model.generate(...)
```

通常返回：

```text
原始输入 token
+
模型新生成 token
```

也就是说：

```text
generated_ids ≠ 只有回答
```

假设：

```text
prompt 长度 = 25 token
生成长度 = 20 token
```

那么最终：

```text
generated_ids 长度 ≈ 45 token
```

---

### 15. 为什么要切掉 Prompt

代码：

```python
prompt_length = model_inputs["input_ids"].shape[1]

output_ids = generated_ids[0][prompt_length:]
```

原因：

> 我们最终只希望 decode 模型新生成的回答，而不是把原始用户 prompt 再输出一次。

因此：

```text
generated_ids
=
prompt tokens
+
new tokens
```

切片以后：

```text
output_ids
=
new tokens
```

---

### 16. decode

最后：

```python
response = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
)
```

把：

```text
Token IDs
```

重新转换成：

```text
自然语言字符串
```

于是完整链路完成。

---

### 17. max_new_tokens

例如：

```python
max_new_tokens=32
```

表示：

> 最多新增 32 个 token。

它不是：

- 32 个字符
- 32 个汉字
- 32 个英文单词

而是：

```text
32 tokens
```

---

## 二、D1 验收标准与标准答案

下面这些问题今天应该可以脱离代码进行解释。

---

### 验收 1：Tokenizer 和 Model 分别负责什么？

**标准答案：**

Tokenizer 负责文本和 Token ID 之间的转换。它把用户输入的字符串切分成 token，并根据 vocabulary 转换成整数 `input_ids`；模型本身不能直接处理字符串。

Model 则接收这些 Token ID 对应的 Tensor，通过 Transformer 进行前向计算，输出 vocabulary 上的 logits，并据此预测后续 token。

简化为：

```text
Tokenizer：
Text ↔ Token IDs

Model：
Token IDs → Logits → Next Token
```

---

### 验收 2：为什么模型看到的是 input_ids，而不是中文字符串？

**标准答案：**

神经网络只能进行数值计算，无法直接对 Python 字符串执行矩阵运算。

因此字符串首先经过 Tokenizer，被映射成 vocabulary 中的整数 Token ID，再通过 Embedding 层转换成向量表示，才能进入 Transformer 进行计算。

流程是：

```text
字符串
↓
Tokenizer
↓
Token IDs
↓
Embedding
↓
Transformer
```

---

### 验收 3：Qwen 为什么能够连续生成文字？

**标准答案：**

Qwen 属于 Causal Language Model，它会根据已有 token 预测下一个 token 的概率分布。

生成一个 token 后，该 token 会被追加到当前序列中，然后模型再次基于更新后的序列预测下一个 token。

因此：

```text
预测 token
↓
加入上下文
↓
再次预测
```

不断循环，就形成连续文本。

这称为：

> Autoregressive Generation，自回归生成。

---

### 验收 4：Chat Template 有什么作用？

**标准答案：**

Chat Template 用于把结构化的 `system/user/assistant` 消息转换成模型在对话训练时采用的标准文本格式。

因为聊天模型在训练阶段使用特定角色标记和消息边界，所以推理阶段最好保持相同格式。

`apply_chat_template()` 可以避免手工拼接不同模型的聊天协议。

---

### 验收 5：messages 和真正进入模型的 input_ids 有什么关系？

**标准答案：**

`messages` 是结构化的对话表示。

首先：

```text
messages
↓
apply_chat_template()
↓
formatted text
```

然后：

```text
formatted text
↓
Tokenizer
↓
input_ids
```

最终模型真正接收到的是 `input_ids`，而不是 messages 本身。

---

### 验收 6：model.generate() 大致做了什么？

**标准答案：**

`model.generate()` 会执行自回归生成流程。

模型首先根据当前输入计算 next-token logits，然后按照指定的 decoding strategy 选择下一个 token，并把该 token 追加到当前序列中。

之后模型继续预测下一个 token，直到达到停止条件。

简化为：

```text
Input
↓
Forward
↓
Logits
↓
Select next token
↓
Append
↓
Repeat
```

---

### 验收 7：为什么 generated_ids 里面还包含原始 Prompt？

**标准答案：**

对于 Decoder-only Causal LM，`generate()` 通常返回完整生成序列：

```text
原始 input_ids
+
新生成 token IDs
```

因此如果只想得到 assistant 的新回答，需要使用输入长度进行切片：

```python
prompt_length = input_ids.shape[1]
output_ids = generated_ids[0][prompt_length:]
```

---

### 验收 8：max_new_tokens 到底限制什么？

**标准答案：**

`max_new_tokens` 限制的是模型在原始输入之后最多新增多少个 token。

它不限制：

- 字符数
- 中文汉字数
- 英文单词数

例如：

```python
max_new_tokens=32
```

表示最多生成 32 个新 token。

---

### 验收 9：为什么推理时使用 torch.no_grad()？

**标准答案：**

推理阶段不需要反向传播，因此不需要保存梯度计算图。

使用：

```python
with torch.no_grad():
```

可以关闭梯度记录，从而降低内存占用和额外计算。

训练阶段需要梯度，而推理阶段不需要。

---

### 验收 10：input_ids 的 shape 如何理解？

**标准答案：**

通常：

```text
input_ids.shape
=
[batch_size, sequence_length]
```

例如：

```text
torch.Size([1, 27])
```

表示：

- batch 中有 1 个样本
- 当前输入序列长度为 27 个 token

如果训练时看到：

```text
[8, 512]
```

则表示：

- batch size = 8
- sequence length = 512

---

### 验收 11：RoBERTa 和 Qwen 的核心区别是什么？

**标准答案：**

RoBERTa 主要是 Encoder-only Transformer，主要用于理解和表示学习，预训练目标以 Masked Language Modeling 为核心。

Qwen 主要是 Decoder-only Transformer，采用 Causal Language Modeling，通过预测下一个 token 进行自回归生成。

简化为：

```text
RoBERTa：
Encoder-only
→ Representation / Classification

Qwen：
Decoder-only
→ Next-token Prediction / Generation
```

---

### 验收 12：完整讲一遍 Qwen 推理链路

**标准答案：**

用户输入首先组织成 `messages`。

然后通过 Chat Template 转换成符合 Qwen 对话协议的文本。

Tokenizer 将文本转换为整数形式的 `input_ids`。

Qwen 接收 Token IDs，通过 Decoder-only Transformer 计算 vocabulary 上的 next-token logits。

`generate()` 根据 logits 选择下一个 token，并不断把新 token 加入已有序列进行自回归生成。

生成结束后，将新增的 Token IDs 从完整序列中切出来，再通过 Tokenizer 的 `decode()` 转换为自然语言回答。

完整流程：

```text
User Input
↓
messages
↓
Chat Template
↓
Text
↓
Tokenizer
↓
input_ids
↓
Qwen
↓
logits
↓
Next Token
↓
Autoregressive Generation
↓
generated_ids
↓
slice new tokens
↓
decode
↓
Response
```

---

## 三、D1 面试题

下面按难度分为三层。

---

# A. 基础题

### Q1：什么是大语言模型？

**参考答案：**

大语言模型是一类通常基于 Transformer 架构、利用大规模文本进行训练的神经网络模型。生成式 LLM 的核心训练目标通常是根据已有 token 预测后续 token，并通过自回归方式完成文本生成。

---

### Q2：Qwen 属于什么类型的模型？

**参考答案：**

Qwen 是生成式大语言模型系列，主要采用 Decoder-only Transformer 架构，并基于 Causal Language Modeling 完成自回归文本生成。

---

### Q3：什么是 Tokenizer？

**参考答案：**

Tokenizer 负责把自然语言转换成模型能够处理的 Token ID，也负责将生成后的 Token ID 解码成人类可读文本。它连接了字符串空间和模型数字空间。

---

### Q4：Token 和 Token ID 有什么区别？

**参考答案：**

Token 是文本切分后得到的基本符号单位，而 Token ID 是该 token 在模型 vocabulary 中对应的整数编号。

---

### Q5：input_ids 是什么？

**参考答案：**

`input_ids` 是输入文本经过 Tokenizer 后得到的 Token ID 序列，通常以 PyTorch Tensor 的形式输入模型。

---

### Q6：Chat Template 是什么？

**参考答案：**

Chat Template 用于把 `system/user/assistant` 等结构化聊天消息转换成模型在对话训练时采用的特定文本格式。

---

### Q7：什么是 Causal Language Model？

**参考答案：**

Causal Language Model 根据当前位置之前的 token 预测下一个 token，其核心形式可以表示为：

\[
P(x_t|x_1,\ldots,x_{t-1})
\]

---

### Q8：什么是自回归生成？

**参考答案：**

模型每次预测一个新的 token，把它加入已有序列后，再继续预测下一个 token。这个不断“预测—追加—再预测”的过程就是自回归生成。

---

### Q9：max_new_tokens 是什么意思？

**参考答案：**

它表示模型最多可以在原始输入之后新增多少个 token，而不是限制字符数或单词数。

---

### Q10：为什么推理时使用 model.eval()？

**参考答案：**

`model.eval()` 会让模型进入推理模式，使 Dropout 等训练阶段特有行为切换到推理状态，从而得到稳定的推理结果。

---

# B. 理解题

### Q11：为什么大语言模型不能直接输入字符串？

**参考答案：**

Transformer 内部进行的是矩阵和向量计算，因此输入必须转换为数值表示。文本首先通过 Tokenizer 转换为 Token ID，再通过 Embedding 层变成向量。

---

### Q12：为什么 Tokenizer 必须和模型匹配？

**参考答案：**

因为模型训练时使用的是特定 vocabulary 和 token 编码规则。如果使用不匹配的 Tokenizer，同一段文本会被映射成错误的 Token ID，模型接收到的含义就会发生变化。

---

### Q13：为什么 Instruct 模型需要 Chat Template？

**参考答案：**

因为 Instruct 模型在训练过程中通常按照固定的角色和消息边界格式学习对话。如果推理输入格式与训练格式不一致，模型的指令遵循能力可能下降。

---

### Q14：apply_chat_template() 和 tokenizer() 有什么区别？

**参考答案：**

`apply_chat_template()` 主要负责把结构化聊天消息组织成符合模型聊天协议的文本格式。

`tokenizer()` 则负责把文本转换成 Token IDs。

因此通常是：

```text
messages
↓
apply_chat_template
↓
formatted text
↓
tokenizer
↓
input_ids
```

---

### Q15：为什么 generated_ids 不是只包含模型回答？

**参考答案：**

Decoder-only 模型的生成是在已有输入序列后继续追加 token，因此 `generate()` 返回的通常是完整序列，也就是原始 prompt 加上新生成 token。

---

### Q16：为什么需要把 generated_ids 切片？

**参考答案：**

因为最终只希望得到模型新生成的 assistant 回答，所以需要根据原始输入 token 长度切掉 prompt 对应部分，再进行 decode。

---

### Q17：torch.no_grad() 和 model.eval() 有什么区别？

**参考答案：**

二者作用不同。

`model.eval()` 用来切换模型内部模块的训练/推理行为，例如关闭 Dropout。

`torch.no_grad()` 则用于关闭 autograd 梯度记录，避免建立反向传播计算图。

推理时通常二者一起使用。

---

### Q18：`do_sample=False` 大致表示什么？

**参考答案：**

表示生成时不进行随机采样，通常使用确定性的 token 选择策略。后续学习 decoding strategy 时还会进一步区分 Greedy Search、Sampling、Top-p、Temperature 等方法。

---

# C. 追问题 / 面试官容易继续问的问题

### Q19：模型是怎么从 Token ID 变成能够计算的向量的？

**参考答案：**

Token ID 会首先经过 Embedding 层。Embedding 本质上是一个可学习的查表操作，每个 Token ID 会映射到一个高维向量，然后该向量再送入 Transformer Blocks。

---

### Q20：模型最后为什么能够预测整个 vocabulary？

**参考答案：**

Transformer 最后的 hidden state 会经过 LM Head 投影到 vocabulary 维度，从而为 vocabulary 中每个 token 得到一个 logit。

再通过生成策略从这些 logits 中决定下一个 token。

---

### Q21：Logit 和 Probability 有什么区别？

**参考答案：**

Logit 是模型直接输出的未归一化分数。

如果对 logits 使用 Softmax，可以转换成 vocabulary 上的概率分布。

---

### Q22：为什么 Qwen 是 Decoder-only，而不是 Encoder-Decoder？

**参考答案：**

Decoder-only 架构天然适合统一建模“给定已有上下文预测后续 token”的任务。Prompt 和回答可以放在同一个序列中进行自回归建模，因此特别适合通用生成式语言模型。

---

### Q23：RoBERTa 为什么不适合直接像 Qwen 一样持续生成文本？

**参考答案：**

RoBERTa 是 Encoder-only 模型，训练目标主要是 Masked Language Modeling，重点是获得双向上下文表示，而不是按照从左到右方式持续预测下一个 token。

Qwen 则直接以 Causal LM 目标学习自回归生成。

---

### Q24：如果 max_new_tokens 设置很大，模型一定会生成那么多 token 吗？

**参考答案：**

不一定。

`max_new_tokens` 是上限。如果模型提前生成 EOS 或满足其他 stopping criteria，生成过程可以提前停止。

---

### Q25：CPU 可以运行 Qwen 吗？

**参考答案：**

可以，只要模型规模和内存允许。

CPU 推理与 GPU 推理的基本计算逻辑相同，主要区别在于计算速度。对于学习 Tokenizer、Chat Template、Generation 等基础流程，小模型 CPU 推理是可行的。

---

### Q26：为什么训练大模型通常需要 GPU？

**参考答案：**

训练不仅需要前向传播，还需要保存中间激活、计算梯度并进行反向传播，因此计算量和内存开销远高于推理。

GPU 对大规模矩阵运算具有更高并行计算能力，因此更适合大模型训练。

---

### Q27：Base Model 和 Instruct Model 有什么区别？

**参考答案：**

Base Model 主要通过大规模预训练学习语言模式和知识，本质目标通常是 next-token prediction。

Instruct Model 通常在 Base Model 基础上进一步进行 SFT 和偏好对齐，使模型更擅长理解用户指令和按照聊天格式回答问题。

---

### Q28：为什么同一个 Qwen 可以回答知识问题、数学问题和生成代码？

**参考答案：**

从底层形式上看，这些任务都统一成“根据当前上下文预测后续 token”。

模型并不是简单地为每种任务单独准备一个模块，而是在大规模预训练和后训练中学习不同文本模式，然后统一通过自回归生成完成不同任务。

---