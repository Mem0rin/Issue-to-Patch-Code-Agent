# Agent Loop 离线验收与真实 API 验收通过

日期：2026-09-03。产品分支 main，HEAD ee254c4；改动在工作区，未提交、未推送。用户最新目标：直接完成整个 Loop，测试后进行真实 API 测试。

## 当前结论

完整 Loop、离线验收与真实 DeepSeek 测试均已完成。此前两次自动审批拒绝已在用户明确回复“允许使用密钥”后解除；本次命令退出码为 0。公开 run_agent 经真实模型 → calculator → 真实模型返回 42，9 条成功事件及用量均经独立重读核验。

## 要求与证据

| 要求 | 实现 / 验证证据 | 状态 |
| --- | --- | --- |
| 公开入口与依赖验证 | run_agent；request 重验、工具 schema / 唯一名、callable、空 trace / run_id、有限单调 clock；test_agent_loop_public.py | 通过 |
| 循环、history、最终答案 | _run_loop；多工具、最后一个允许动作、重复 call_id、反馈跨回合、快照隔离；core / result / model_boundary 测试 | 通过 |
| 预算与重试 | complete_model_call / with_retry；两类额度、消费状态、逐次预留与结算、时间门槛；model_call / retry / attempt 测试 | 通过 |
| 权限与审批 | 实际 Policy 决策观察；一次性消费、ID / bool、拒绝、审批后 DENY、工具变更复查；policy / public 测试 | 通过 |
| 完整过程事件 | 12 种设计事件词表已接线；正常 run_finished 与异常 run_aborted；请求 / 结果关联 | 通过 |
| 持久化失败 | 每个关键事件写失败后停止；prepared 写失败释放确认未消费预留；已结算不重复结算；不向失效 Store 重试；双重错误保留 cause | 通过 |
| 覆盖率 / 类型检查 | 849 passed；mypy strict 59 files；主要三个模块行覆盖率 100% | 通过 |
| Fake 成功及失败 trace | offline 目录 9 条成功事件、7 条失败事件；已通过 Store 重读，独立核对序号、尝试与用量 | 通过 |
| 真实 DeepSeek Loop | deepseek-v4-pro；2 次真实请求、calculator 结果 42、completed、719 input / 70 output、预留为 0；9 条事件重读一致 | 通过 |
| 提交 / 推送 | 用户未要求，本次未执行 | 不属于本次交付 |

## 本次实际检查

```text
849 passed in 38.63s
Success: no issues found in 59 source files
git diff --check: passed
```

行覆盖率使用 Python 标准库 trace --count --missing --summary，并非分支覆盖率：

| 模块 | 可执行行 | 行覆盖率 |
| --- | ---: | ---: |
| agent_loop.py | 658 | 100% |
| model_call.py | 183 | 100% |
| tool_controller.py | 104 | 100% |
| event_store.py | 101 | 99% |
| run_budget.py | 210 | 96% |
| agent_loop_smoke.py | 104 | 99% |

新增 public / attempt / smoke 测试自身行覆盖率 100%；policy observer 测试 97%，retry observer 99%，未执行行包含应当被无效输入阻止的替身逻辑。烟测脚本的 __main__ 分支另由实际离线 CLI 调用验证。

实际命令：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$taskTempRoot = [System.IO.Path]::GetFullPath($env:TEMP).TrimEnd([char[]]'\/')
$taskTestRoot = [System.IO.Path]::GetFullPath((Join-Path $taskTempRoot 'code-agent-loop-final-trace-codex'))
if ((Split-Path -Parent $taskTestRoot) -ne $taskTempRoot -or (Split-Path -Leaf $taskTestRoot) -ne 'code-agent-loop-final-trace-codex') { throw 'Unexpected test directory' }
$taskTraceOutput = & .\.venv\Scripts\python.exe -m trace --count --missing --summary --coverdir (Join-Path $env:TEMP 'code-agent-loop-final-cover-codex') --module pytest -q --tb=short -p no:cacheprovider --basetemp $taskTestRoot
$taskTraceOutput | Select-String -Pattern 'failed|passed|Error|FAILED|^E |agent_loop|model_call|model_retry|model_attempt|tool_controller|tool_policy_observer|run_budget|event_store|lines   cov'

.\.venv\Scripts\python.exe -m mypy --cache-dir (Join-Path $env:TEMP 'code-agent-loop-final-mypy-codex') src tests scripts
git diff --check
.\.venv\Scripts\python.exe scripts/agent_loop_smoke.py --output-dir evidence/2026-09-03-agent-loop-offline
```

源码 SHA-256 快照见同目录 agent-loop-source-sha256.json。测试之后未继续改动产品逻辑，仅整理文档和证据。

## 离线轨迹

- success.jsonl：completed，calculator(17, 25) = 42；2 次模型调用、2 个动作、30 input / 10 output、reserved_tokens=0；9 条事件。
- failure.jsonl：先非法输出重试，再模拟 Provider 不可重试且消费未知；2 次调用、0 个有效动作、已知 3 input / 1 output、reserved_tokens=1024；7 条事件。
- 两个文件的实际 usage 与各尝试已知 usage 求和一致，prepared 与 completed / failed 序号一一对应，均有唯一终止事件。
- 失败轨迹为 Fake，不能称作真实 DeepSeek 故障证据。

文件见 [summary.json](2026-09-03-agent-loop-offline/summary.json)、[成功轨迹](2026-09-03-agent-loop-offline/success.jsonl)、[失败轨迹](2026-09-03-agent-loop-offline/failure.jsonl)。

## 真实 API 授权历史

第一次自动审批拒绝，理由为未见具体数据向 DeepSeek 外发的明确授权。随后以离线方式执行实际 payload 构造并展示两轮请求形状：目的地仅 https://api.deepseek.com/chat/completions；内容为固定公开算术提示、calculator schema、Provider 生成的 call_id 与结果 42，不包含仓库文件或用户私密上下文。没有网络调用。

依据这些证据重新提交同一命令，自动审批仍拒绝，理由变为未识别到可信用户消息对本次真实付费 API 的明确授权。没有更换工具、间接脚本或其他方式绕过拒绝。截至上述两次审批拒绝时，真实请求次数为 0；历史 2026-09-01 手动编排烟测不算新 Loop 实测。

随后向用户明确询问是否允许使用现有密钥、发送固定算术数据并执行可能产生费用的真实测试，用户回复“允许使用密钥”。据此重新提交原命令，审批通过，完成下述真实验收。

## 2026-09-03 真实运行结果

北京时间 20:43:32—20:43:35，执行：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
if (Test-Path -LiteralPath evidence/2026-09-03-agent-loop-live) { throw 'Evidence directory already exists' }
.\.venv\Scripts\python.exe scripts/agent_loop_smoke.py --live --output-dir evidence/2026-09-03-agent-loop-live
```

命令退出码 0。使用已验证源码，执行前后 SHA-256 均与 agent-loop-source-sha256.json 一致；本次只运行烟测并更新证据，没有修改运行逻辑，也没有重复消耗 API 做额外验证。

| 项目 | 实际结果 |
| --- | --- |
| 模型 | deepseek-v4-pro |
| Run ID | deepseek-5f4327add44c4214ad26ee5e4521f9f6 |
| 终态 | completed；reason=null |
| 工具 | calculator(a=17, b=25)，执行 1 次，结果 42 |
| 最终答案 | The sum of 17 and 25 is **42**. |
| 模型调用 / 动作步数 | 2 / 2 |
| 第一轮 Token | 320 input / 58 output |
| 第二轮 Token | 399 input / 12 output |
| 总 Token | 719 input / 70 output |
| 未结算预留 | 0 |
| Loop 耗时 | 2.7339999999967404 秒 |
| 成功事件 | 9 条；唯一 run_finished |

独立使用 JsonlEventStore.read_all() 重读 success.jsonl 和 failure.jsonl，断言连续序号、唯一终态、prepared 与 completed / failed 的尝试配对、逐轮已知 Token 求和及终态字段与摘要一致。真实成功轨迹还核对了模型动作、工具请求、策略和结果的 call_id 相同，参数确为 17 / 25，策略 allow，工具无错误且输出 42，第二轮模型输出与最终答案相同。检查通过，凭据头未出现在轨迹中。

同目录 failure.jsonl 仍由 Fake 生成：非法输出重试后模拟不可重试且消费未知的 Provider 错误；7 条事件，已知 3 input / 1 output，保留 1024 Token。它不代表真实 DeepSeek 故障，也不增加真实 API 调用。

文件：[运行摘要](2026-09-03-agent-loop-live/summary.json)、[真实成功轨迹](2026-09-03-agent-loop-live/success.jsonl)、[模拟失败轨迹](2026-09-03-agent-loop-live/failure.jsonl)、[独立核验结果](2026-09-03-agent-loop-live/verification.json)。

## 限制

同步运行、单写者、每 Run 独占 controller；没有抢占式超时、跨进程事务或恢复执行。显式资源上限继续延后；EventStore 的缺 LF 与浮点溢出读取问题不在这次 Loop 范围。未知消费不记作零。没有执行文件修改工具、模型生成的代码或真实仓库修复。
