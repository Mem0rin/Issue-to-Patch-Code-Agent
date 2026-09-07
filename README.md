# Issue-to-Patch Code Agent

## 项目简介

Issue-to-Patch Code Agent v0 是一个可审计的同步单 Agent Kernel。它先把一次模型行动如何被接受、记账、授权和追踪做清楚，再扩展真实的代码修复能力。模型输出只是一项提议；只有通过运行时检查和权限判断后，工具才会接触环境。

一次模型请求从预算预留开始。响应返回后，Kernel 按实际 Token 用量结算，并把 `ToolCall` 与 `FinalAnswer` 都记作行动。已确认没有消费才释放预留，消费情况不明则保留。Provider 故障和模型格式错误各用一套重试额度，每次重试重新申请预算。这样即使运行失败，账本也不会把已经发生或可能发生的消耗抹掉。

工具调用由 Registry、Runtime、Policy 和一次性 Approval 共同处理。Registry 确认工具存在，Runtime 校验参数，Policy 决定允许、询问或拒绝。用户批准后，系统还会重新准备工具并再次检查 Policy，避免等待批准期间工具或权限发生变化。模型提出动作、系统授予权限、环境实际执行，是三件分开的事。

每次运行都会写入严格的 JSONL 轨迹。`model_call_prepared` 只表示调用意图，`tool_call_requested` 也不等于工具已经执行；完成、失败和最终终态分别记录。EventStore 拒绝重复 JSON 键、缺少 LF 的尾行、非有限浮点数和不连续序号，并在落盘前递归脱敏。轨迹写入一旦失败，Loop 会停止后续动作，不会拿一份缺失关键证据的记录宣称运行成功。

ModelAdapter 隔离了 Kernel 与具体 Provider 协议，Fake Model 和 DeepSeek 使用同一套循环、预算和工具规则。运行时 `history` 保存下一轮模型真正需要的内容，EventStore 保存审计证据；两者职责独立，日志不会自动混入模型上下文。

当前 v0 已完成预算约束下的模型循环、工具调用、同步审批、重试、明确终止和过程追踪。它仍是单进程、单写者实现，不提供抢占式超时、崩溃恢复、旧运行续接和完整 Issue-to-Patch 流程。这些限制写在接口和证据里，避免把演示能力说成已经解决的问题。

核心流程：

```text
预算预留 → 模型与 Adapter → 用量结算 → Agent Loop
                                      ├─ FinalAnswer → 结束
                                      └─ ToolCall
                                           ↓
                               Registry / Runtime
                                           ↓
                              Policy / Approval
                                           ↓
                                    ToolResult
                                           ↓
                                  history → 下一轮
```

## 运行

在仓库根目录使用 Python 3.12 环境：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$testTemp = Join-Path $env:TEMP ("agent-loop-user-" + [guid]::NewGuid())
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp $testTemp
.\.venv\Scripts\python.exe -m mypy src tests scripts
.\.venv\Scripts\python.exe scripts/agent_loop_smoke.py --output-dir evidence/my-loop-offline
```

烟测默认不联网，生成 success.jsonl、failure.jsonl 和 summary.json。输出目录必须不存在，以免覆盖已有证据。

真实模式读取现有 DEEPSEEK_API_KEY，调用 DeepSeek 的 /chat/completions。请求内容是固定的 17 + 25 任务、calculator 工具定义、服务生成的工具调用及计算结果 42；工具不访问文件或网络。明确允许真实调用后运行：

```powershell
.\.venv\Scripts\python.exe scripts/agent_loop_smoke.py --live --output-dir evidence/my-loop-live
```

## 入口契约

```python
result = run_agent(
    request,
    adapter=adapter,
    controller=controller,
    tools=registry.specs,
    event_store=store,
    approval_handler=approval_handler,
    clock=clock,
)
```

- request 为 RunRequest，初始 history 只接受 SYSTEM / USER 消息，至少包含一个 USER；预算与重试计数在入口完整验证。
- tools 为名称唯一的 ToolSpec 元组，允许空集合；schema 必须是有效 JSON 对象。请求、工具描述和各次模型参数使用独立快照。
- store 的 run_id 必须匹配 request.run_id，trace 必须为空。每个运行独占 controller 的审批状态及 trace，不支持并发写者或恢复旧运行。
- approval_handler 必须返回匹配请求 ID 的 ApprovalResponse；批准后仍重新检查工具和策略。硬性 DENY 优先，工具执行错误成为配对的 ToolResult。
- clock 默认 monotonic。时间经过检查必须有限且不倒退；同步模型、审批或工具不会被抢占中断，返回后及启动后续操作前检查时间。
- 正常终态为 completed、budget_stopped、model_failed。意外异常保留操作上下文；事件写入失败立即中止，不能返回 completed，也不会向已失效 Store 重试写错误事件。

重试分 Provider 与非法输出两类，额度分别计数；每次尝试重新预留，错误按消费状态结算。未知消费保留 reserved_tokens。安排的 ModelFeedback 会进入后续模型上下文及最终 history。

## 轨迹

run_started → 每次 model_call_prepared / completed 或 failed → 可选 model_retry_scheduled → 工具请求、实际策略、审批、工具结果 → run_finished。意外错误在 Store 仍可用时记录 run_aborted。

意图事件不能证明外部操作已经完成。文件写入与模型 / 工具操作没有事务，损坏或缺少终止事件的 trace 不支持安全自动重放。写盘前递归字段脱敏不保证检测任意自由文本中的秘密。

## 验收

2026-09-03：849 项测试通过；strict 检查 59 个文件通过。标准库 trace 行覆盖率：agent_loop、model_call、tool_controller 均为 100%。离线成功 / 失败轨迹已核对事件序号、尝试配对及用量。

用户明确允许使用现有密钥后，真实 DeepSeek Loop 已通过：2 次模型调用、1 次 calculator 执行，最终回答 42；719 input / 70 output Token，reserved_tokens=0，耗时约 2.734 秒。成功轨迹的 9 条事件已重读核对，模拟失败轨迹另有 7 条事件。见 [验收记录](evidence/2026-09-03-agent-loop-validation.md) 和 [真实运行摘要](evidence/2026-09-03-agent-loop-live/summary.json)。

2026-09-05：EventStore 缺 LF、浮点溢出和扩展字段兼容证据已完成。全量 878 项测试、strict 62 个文件通过；新增扩展字段测试行覆盖率 100%。09-03 真实运行和源码哈希保留为历史证据。显式资源上限继续延后，完整 Issue-to-Patch 修复流程另行推进。
