# D3｜Qwen Chat Template、多轮对话与 System Prompt 学习总结

## 一、D3 学习目标

D3 的核心是理解：**聊天模型如何把结构化对话转换成真正送入模型的输入序列**。

```text
messages
↓
Chat Template
↓
Formatted Conversation
↓
Tokenizer
↓
input_ids
↓
Qwen
↓
Assistant Response
```

今天重点掌握：messages、system/user/assistant、Chat Template、`add_generation_prompt=True`、`tokenize=False`、System Prompt、多轮对话机制、Context 增长、Prompt 与 Context 的区别，以及 Chat Template 与 Tokenizer、SFT 的关系。

## 二、messages 是什么

`messages` 是结构化的对话历史。

```python
messages = [
    {"role": "system", "content": "你是一名AI学习助手。"},
    {"role": "user", "content": "什么是Transformer？"}
]
```

它本身不是模型最终接收的 Tensor，而是聊天内容的结构化表示。

常见角色：

- `system`：定义身份、风格、规则和约束。
- `user`：表示用户输入。
- `assistant`：表示模型之前生成的回答。

多轮对话本质上就是不断把新的 user 和 assistant 消息追加到 `messages` 中。

## 三、Chat Template 是什么

Chat Template 的作用是：

> 把结构化的 `system/user/assistant` 消息转换成模型训练时使用的对话格式。

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

本质流程：

```text
messages
↓
角色标记 / 分隔符 / 结束标记
↓
模型专属聊天格式
```

不同模型可能使用不同的聊天模板，因此不能随意手工拼接 `User:`、`Assistant:`，除非明确知道模型训练时使用的就是该格式。

## 四、为什么训练和推理格式要一致

如果训练时使用 Template A，推理时却使用 Template B，就可能产生：

```text
Train-Inference Mismatch
```

可能导致：

- 指令遵循能力下降
- 角色混乱
- 输出特殊符号
- 重复生成
- 停止行为异常

因此 Chat Template 的一个核心意义是保持训练和推理格式尽可能一致。

## 五、add_generation_prompt=True

```python
add_generation_prompt=True
```

可以理解为：

> 在输入末尾加入“接下来应该由 assistant 回答”的起始标记。

概念上类似：

```text
user:
什么是Transformer？

assistant:
```

这样模型可以明确知道下一步应该生成 assistant 内容。

## 六、tokenize=False

```python
tokenize=False
```

表示 `apply_chat_template()` 暂时只生成格式化后的字符串，不直接转成 Token IDs。

学习阶段这样做的好处是可以直接观察 Chat Template 到底增加了哪些角色标记和特殊格式。

如果使用：

```python
tokenize=True
```

则可以进一步直接生成 Token IDs。

## 七、System Prompt

System Prompt 本质上是上下文中的高优先级指令，可以影响：

- 模型身份
- 回答风格
- 输出格式
- 任务规则
- 行为约束

例如：

```python
{"role": "system", "content": "你是一名计算机专业教师，请用简洁语言回答。"}
```

需要注意：

> System Prompt 不会修改模型参数，也不能保证模型百分之百遵守，它仍然只是当前 Context 的一部分。

## 八、多轮对话如何实现

多轮对话并不是模型永久记住了上一轮内容。

最基础实现方式是：

```text
历史消息
+
当前消息
↓
重新一起发送给模型
```

例如：

```python
messages = [
    {"role": "user", "content": "我正在学习Transformer。"},
    {"role": "assistant", "content": "好的。"},
    {"role": "user", "content": "我刚才说我在学什么？"}
]
```

模型能回答“Transformer”，是因为上一轮内容仍然存在于当前 Context 中。

因此基础聊天记忆可以理解为：

```text
Conversation History
+
Context Replay
```

## 九、为什么多轮对话会越来越长

每进行一轮：

```text
历史消息增加
↓
Token 数增加
↓
Context Length 增加
```

带来的结果包括：

- 推理时间增加
- 内存占用增加
- Token 成本增加
- 最终可能达到 Context Window 上限

真实系统中常见处理方式包括：

- 历史截断
- Sliding Window
- Conversation Summary
- Long-term Memory
- Vector Retrieval

这些内容后续 Agent 阶段还会继续学习。

## 十、Prompt 与 Context 的区别

Prompt 通常指当前给模型的任务或指令。

Context 范围更广，可以包含：

```text
System Prompt
+
Conversation History
+
Current User Message
+
RAG Retrieved Documents
+
Tool Results
```

因此可以理解为：

```text
Prompt ⊆ Context
```

## 十一、Chat Template 与 Tokenizer

Chat Template 通常由 Tokenizer 管理，因此调用的是：

```python
tokenizer.apply_chat_template(...)
```

完整流程：

```text
messages
↓
tokenizer.apply_chat_template()
↓
formatted text
↓
tokenizer()
↓
input_ids
↓
model
```

Chat Template 也可以直接 Tokenize：

```python
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)
```

学习阶段先使用 `tokenize=False`，主要是为了看清中间的聊天格式。

## 十二、Chat Template 与 SFT 的关系

以后做 SFT 时，训练数据经常也是：

```python
{
    "messages": [
        {"role": "user", "content": "什么是LoRA？"},
        {"role": "assistant", "content": "LoRA是一种..."}
    ]
}
```

训练前通常经历：

```text
messages
↓
Chat Template
↓
Tokenization
↓
input_ids / labels
↓
Causal LM Loss
```

所以 D3 学习 Chat Template 不只是为了聊天，它会直接影响：

- SFT
- Assistant Loss Masking
- DPO
- Preference Data
- 推理格式一致性

## 十三、D3 核心流程

```text
messages
│
├── system
├── user
├── assistant
└── user
      ↓
Chat Template
      ↓
Formatted Conversation
      ↓
Tokenizer
      ↓
input_ids
      ↓
Qwen
      ↓
Assistant Response
      ↓
追加回 messages
      ↓
下一轮
```

## 十四、D3 核心知识点速记

```text
messages ≠ input_ids
```

```text
Chat Template
=
结构化消息 → 模型专属聊天格式
```

```text
System Prompt
=
上下文中的高优先级指令
≠
修改模型参数
```

```text
多轮聊天记忆
=
历史消息重新进入 Context
```

```text
add_generation_prompt=True
=
告诉模型下一步应该生成 assistant
```

```text
Prompt ⊆ Context
```

```text
训练模板 ≈ 推理模板
```
