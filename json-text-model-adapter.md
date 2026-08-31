# JSON 文本模型 Adapter

Kernel 保存的是 `Message`、`ToolCall` 和 `ToolResult`，文本模型接收的是 `ProviderMessage`，返回的则是一段原始文本。两边的数据不能直接互换，因此需要 Adapter 做双向转换。

这部分实现由三个文件组成：

```text
HistoryItem
    ↓
json_text_messages.py
    ↓
ProviderMessage
    ↓
json_text_adapter.py
    ↓
ProviderResponse.text
    ↓
json_action_parser.py
    ↓
ToolCall 或 FinalAnswer
```

**目录**

- [一、文件关系](#一文件关系)
- [二、消息转换](#二消息转换)
- [三、Adapter 调度](#三adapter-调度)
- [四、行动解析](#四行动解析)

## 一、文件关系

三个文件各自只处理一件事：

| 文件 | 职责 |
|---|---|
| `json_text_messages.py` | 把 Kernel 历史转换成 Provider 消息 |
| `json_text_adapter.py` | 调用 Provider，统一返回值和异常 |
| `json_action_parser.py` | 检查原始文本，并转换成 Kernel Action |

`json_text_adapter.py` 位于中间。请求发出前，它调用 `build_provider_messages()`；响应回来后，它调用 `parse_json_action()`。

```python
provider_messages = build_provider_messages(history)
raw_response = self._provider.complete(...)
action = parse_json_action(raw_response.text)
```

所以它本身不负责每一种消息怎样编码，也不负责 JSON 字段怎样验证。这些细节分别留在另外两个文件中。

## 二、消息转换

`json_text_messages.py` 处理的是出站方向：

```text
Kernel History → ProviderMessage
```

Kernel 的历史类型是：

```python
HistoryItem = Message | ToolCall | ToolResult
```

`build_provider_messages()` 依次判断每个历史项的类型。

普通 `Message` 已经包含角色和内容，可以直接转换：

```python
ProviderMessage(
    role=item.role.value,
    content=item.content,
)
```

`ToolCall` 是模型此前发出的行动，所以使用 `assistant` 角色，并编码成 JSON：

```python
ProviderMessage(
    role="assistant",
    content=_encode_json({
        "type": "tool_call",
        "call_id": item.call_id,
        "tool_name": item.tool_name,
        "arguments": item.arguments,
    }),
)
```

`ToolResult` 是环境返回给模型的观察结果。当前使用的是纯文本消息格式，没有独立的 `tool` 角色，因此暂时使用 `user`：

```python
ProviderMessage(
    role="user",
    content=_encode_json({
        "type": "tool_result",
        "call_id": item.call_id,
        "content": item.content,
        "is_error": item.is_error,
    }),
)
```

`call_id` 必须原样保留。模型可能提出多个工具调用，如果结果没有对应 ID，就无法判断每个结果属于哪次调用。

## 三、Adapter 调度

`json_text_adapter.py` 把 Provider 接到 `ModelAdapter` interface 上。这里会出现三个容易混淆的类型：

| 类型 | 方向 | 内容 |
|---|---|---|
| `ProviderMessage` | Adapter 发给 Provider | 模型能读取的消息 |
| `ProviderResponse` | Provider 返回给 Adapter | 原始文本和 Token 用量 |
| `ModelResponse` | Adapter 返回给 Kernel | 合法 Action 和标准用量 |

`JsonTextModelAdapter.complete()` 的执行顺序如下：

```python
provider_messages = build_provider_messages(history)

raw_response = self._provider.complete(
    messages=provider_messages,
    tools=tools,
    max_output_tokens=budget.max_output_tokens,
)

usage = TokenUsage(
    input_tokens=raw_response.input_tokens,
    output_tokens=raw_response.output_tokens,
)

action = parse_json_action(raw_response.text)

return ModelResponse(
    action=action,
    usage=usage,
)
```

Provider 调用失败时，底层 `ProviderError` 会转换成统一的 `ModelProviderError`。Kernel 不需要认识具体 Provider 的异常类型。

如果 Provider 已经返回文本，但文本无法解析，Adapter 会抛出 `InvalidModelOutputError`。这时 Token 已经消耗，所以异常会携带刚才构造的 `TokenUsage`。

## 四、行动解析

`json_action_parser.py` 处理的是入站方向：

```text
不可信文本 → 合法 Action
```

解析从 `json.loads()` 开始，然后逐层检查：

```text
必须是合法 JSON
    ↓
最外层必须是字典
    ↓
type 必须是 tool_call 或 final_answer
    ↓
字段集合和字段类型必须匹配
```

工具调用必须包含这些字段：

```json
{
  "type": "tool_call",
  "call_id": "call_1",
  "tool_name": "read_file",
  "arguments": {
    "path": "README.md"
  }
}
```

解析器会检查 `arguments` 是不是字典，但不会继续检查 `path` 的业务类型。不同工具拥有不同参数规则，这部分由工具自己的参数验证器处理。

最终回答只允许两个字段：

```json
{
  "type": "final_answer",
  "content": "42"
}
```

只要缺少字段、增加额外字段或字段类型错误，解析器就会抛出 `InvalidModelOutputError`。这样 Kernel 收到的 `ModelResponse.action` 一定是已经通过结构检查的 `ToolCall` 或 `FinalAnswer`。

一句话概括这三个文件的关系：

```text
json_text_messages 负责把历史送出去
json_text_adapter 负责组织一次模型调用
json_action_parser 负责把模型文本安全地收回来
```
