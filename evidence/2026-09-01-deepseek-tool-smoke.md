# DeepSeek V4 Pro 原生工具调用烟测

> 日期：2026-09-01
>
> 模型：`deepseek-v4-pro`
>
> 接口：非流式 `POST /chat/completions`
>
> 模式：`thinking.type=disabled`
>
> 工具：纯计算 `calculator(a: int, b: int)`

## 验收目标

验证真实 Provider 能完成以下闭环，并保持 Kernel 的预算与权限语义：

```text
user → DeepSeek tool call → ToolController → calculator
     → tool result → DeepSeek final answer
```

密钥只从 `DEEPSEEK_API_KEY` 环境变量读取。烟测输出、异常和本文件均不包含密钥或 Authorization header。

## 配置

| 配置 | 值 |
| --- | ---: |
| 单次 `max_output_tokens` | 128 |
| 单次 `reserved_tokens` | 1024 |
| 整轮 `max_total_tokens` | 4096 |
| `max_model_calls` | 4 |
| `max_action_steps` | 2 |
| Provider / 非法输出最大重试 | 各 1 次 |
| 工具副作用 | 无 |

## 首次失败

第一次完整烟测受控失败：

```text
InvalidModelOutputError:
provider returned a tool call and final text
```

重试策略执行后仍遇到同一错误并终止。由于响应已返回 usage，此次失败可能已经产生消费，不能记为零；原烟测脚本在异常退出前没有输出累计 usage，因此总量标记为 `unknown`。

随后执行一次只输出结构元数据的诊断调用：

```json
{
  "content_blank": true,
  "content_is_none": false,
  "content_length": 0,
  "finish_reason": "tool_calls",
  "input_tokens": 316,
  "output_tokens": 58,
  "tool_call_count": 1
}
```

根因不是 DeepSeek 同时给出实质 final answer，而是 tool-call message 的 `content` 为 `""`。Fake Provider 使用了 `None`，没有覆盖真实 Provider 的空字符串形状。

修复：只有 tool call 同时带有非空文本时才判定为冲突；一个 tool call 加空字符串被视为没有 final text。无 tool call 的空 final answer 仍然非法。新增回归测试固定该语义。

## 修复后成功证据

第二次完整烟测成功输出：

```json
{
  "action_steps": 2,
  "call_id": "call_00_7DHycsub1sTMOINjIVzm9963",
  "elapsed_seconds": 2.953,
  "final_answer": "17 + 25 = **42**",
  "input_tokens": 747,
  "model": "deepseek-v4-pro",
  "model_calls": 2,
  "output_tokens": 66,
  "reserved_tokens": 0,
  "status": "ok",
  "tool_name": "calculator",
  "tool_result": "42"
}
```

已验证：

- Provider `tool_calls[i].id` 原样成为 Kernel `call_id`；
- `ToolController` 执行已允许的纯计算工具；
- 工具结果与原始 `call_id` 一起进入第二轮 history；
- 第二轮返回 `FinalAnswer`；
- 两次成功响应均以实际 usage 结算，最终 `reserved_tokens=0`；
- `model_calls=2`、`action_steps=2` 与真实调用链一致。

## 费用上界

成功运行没有保留 cache hit/miss 明细，因此按全部 input 都是 cache miss 估算。根据 2026-09-01 的 [DeepSeek 官方价格](https://api-docs.deepseek.com/quick_start/pricing/)：

- Off-peak 上界约 `$0.000624`；
- Peak 上界约 `$0.001247`。

这只计算成功运行的 747 input / 66 output Token，不包含首次失败和诊断调用。整次调试会话的准确总费用未知，不能用成功运行费用代替。

## 回归证据

```text
122 passed
Success: no issues found in 30 source files
git diff --check passed
```

## 当前局限

- 尚未实现 EventStore，因此这是结构化烟测记录，不是 Kernel JSONL trace；
- thinking mode 暂时关闭，尚未设计 `reasoning_content` 的回放和日志策略；
- Kernel v0 只接受一个 tool call，多调用仍受控拒绝；
- Provider 映射尚未保留 cache hit/miss Token，费用只能估算上界；
- 当前只验证安全纯计算工具，不能推出文件工具或代码执行已经安全。
