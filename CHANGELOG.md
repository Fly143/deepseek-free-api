## [v2.4.2] — 2026-09-18

覆盖 PR #32/#33（@khs0927 + maintainer review）与 PR #34（Issue #31）。

### 修复
- **账号禁言不再空 completion** — DeepSeek 上游 HTTP 200 + JSON mute/ban（`biz_code=5` 等）时，chat 返回 **403 `account_muted`**（含 `mute_until`）；覆盖 nonstream 缓冲体、裸 JSON 行、SSE `data:` JSON 包装
- **注册 `RISK_DEVICE_DETECTED` 可诊断**（Issue #31 / PR #34）— 解析 `data.biz_code`；`11`/`6` 明确报错，不再误报「注册响应中无 token」
- **发码与注册指纹统一** — 验证码接口默认与 register 同一 iOS 指纹族，消除 web→ios 平台跳变
- **模型 fallback 不再优先 search** — 未知模型落到 `deepseek-default`
- **Python 3.10 兼容** — 去掉嵌套同引号 f-string（PEP 701 为 3.12+）

### 变更
- **上游聊天 pacing（全局）** — `app/lock_guard.upstream_slot()`：默认 `DEEPSEEK_MIN_INTERVAL_SEC=2.5`、`DEEPSEEK_MAX_CONCURRENT=1`、`DEEPSEEK_ERROR_COOLDOWN_SEC=60`；多账号共用闸门，优先降低 mute 风险
- **默认 search 关闭** — 仅当 model id 含 `search` 时开启；官方名 normalize 到 non-search bridge id
- **上下文预算** — `_DEFAULT_MAX_INPUT` 64k→1M；chat 路径传入模型发现的 `max_in`
- **`get_next_account` 热路径不再每次写盘** — 仅 mute/cooldown 状态变化时 save

### 新增
- **账号池 mute/cooldown** — `mark_account_muted` / `mark_account_cooldown` / `mark_account_ok`；跳过不可用账号，健康号 LRU
- **注册 device_id 拉黑重试** — 风控拒后 burn 该 ID、换指纹、重发码、重试一次；日志含 `proxy=on|OFF`
- **`DEEPSEEK_DEVICE_IDS_FILE`** — 私有 device_id 池（一行一个）
- **`docs/OPS-RUNBOOK.md`** — pacing / 注册风控运维说明
- **离线测试** — `tests/test_pr32_mute_guard.py`、`tests/test_register_risk_device.py`

### 致谢
- PR #32 by [@khs0927](https://github.com/khs0927)
- PR #30 作者 [@Anai-Guo](https://github.com/Anai-Guo)（已在 v2.4.0 摘要提及）

## [v2.4.1] — 2026-09-12

### 变更
- **移除代理层 `max_tokens=4096` 兜底** — Anthropic 转换层未指定时不再写入，纯透传（对齐 MiMo2API）
- **temperature / top_p / max_tokens 纯透传** — 客户端显式传入才下传 DeepSeek；不猜默认值，由被代理端处理
- **思考+输出取消默认 HTTP 时限** — `DS_CLIENT_TIMEOUT` 默认 `0`（不限）；整条流（thinking + content）纯透传，不再被 600s 掐断。需要保护时显式设秒数
- **上下文压缩默认仍为 `compress`** — 管理面板可切 `truncation`；不改则走 LLM 摘要

## [v2.4.0] — 2026-09-12

从 MiMo2API v2.5.1→v2.6.6 移植可共用能力，并修复若干工具/会话缺陷。

### 修复
- **session token 记录改为峰值** — `prompt_tokens` 是完整上下文而非增量，累加会过早触发续期（对齐 MiMo session_store 峰值策略）
- **Anthropic 流式异常收尾丢失 tool_use** — 无 `finish_reason` 时先下发已攒好的 `tool_use` blocks，再定 `stop_reason`
- **Anthropic message_delta 补 input_tokens** — 客户端可显示输入/输出 token
- **带 tools 流式 content+tool_calls 一致性** — 正文缓冲；命中工具调用则丢弃正文，否则收尾补发（与非流式一致）
- **StreamSieve 标记前缀大小写不敏感** — 小写 DSML 标记跨 chunk 切断时不再被当正文吐出
- **CORS `allow_credentials=false`** — 与 `*` origin 组合更安全
- **batch 缺失 import 修复**（PR #30）— `json` / `uuid` 补齐，Batch API 可创建/落盘

### 新增
- **上下文压缩** — `compress`（LLM 摘要）/ `truncation`（滑动窗口）+ 80% 阈值；管理面板可切换；默认接入 `context_manager.enforce_context_limit`
- **Anthropic 模型别名增强** — Claude 4.7 / 日期后缀 / `-latest` / 未知 claude-* 启发式
- **凭证 Fernet 加密落盘** — token / password / cookie / headers / admin_password / mailcx_api_key，密钥 `.secret_key`；旧明文自动迁移
- **指数退避重试** — 聊天 5xx/网络失败自动重试（`DS_RETRY_*`）
- **HOST 环境变量** — `PROXY_HOST`（默认 `0.0.0.0`），端口兼容 `PORT`/`PROXY_PORT`

## [v2.3.7] — 2026-05-30

### 新增
- **管理员密码认证** — 管理面板支持 HTTP Basic Auth 密码认证，保护管理接口安全
  - 新增 `app/auth.py` 认证模块，使用 `secrets.compare_digest` 防时序攻击
  - `config.json` 新增 `admin_password` 字段，默认密码 `admin`
  - 所有管理接口（`/`、`/admin`、`/api/*`）添加认证保护，`/v1/*` API 接口不受影响
  - 管理面板添加密码输入框，支持中英文提示
  - 密码保存在 `sessionStorage`，关闭浏览器后需重新输入

## [v2.2.9] — 2026-05-12

### 修复
- **语言切换按钮位置** — 从标签栏移至右上角，添加 `position:relative` 修复布局
- **README 中英互链** — 中英文 README 互相引用

## [v2.2.8] — 2026-05-12

### 新增
- **多语言支持** — 管理面板支持中英双语切换（🌐 EN/中 按钮），涵盖所有 UI 文本、Toast 消息、表格表头
- **CORS 中间件** — 添加 CORS 支持，允许跨域访问 API（解决浏览器客户端 CORS 错误）
- **英文 README** — 新增 `README_EN.md` 英文版文档

## [v2.2.7] — 2026-05-12

### 修复
- **SSE 注释行导致解析失败** — DeepSeek 原始 SSE 中的 `:` 注释行（标准 SSE keepalive）未被过滤，被 `non_json_line_count` 误判为非法内容，3 行后触发错误退出。Hermes 等客户端收到错误后报 `too many non-JSON lines`

## [v2.2.6] — 2026-05-11

### 修复
- **换行符保留** — `clean_tool_text` 不再 strip 末尾空白，避免独立 `
` 分块被吃导致 Markdown 格式挤在一起

# 更新日志（Changelog）

本文件记录 deepseek-free-api 的所有重要变更。

---

## [v2.2.5] — 2026-05-11

### 修复
- **工具标签泄漏补全** — `clean_tool_text` 覆盖所有文本输出路径（tools 流式 / 无 tools 流式），DSML 和 DeepSeek 原生工具标签全部兜底清理

## [v2.2.4] — 2026-05-11

### 修复
- **思考链泄漏** — tools 流式路径改为缓冲 text content，确认有工具调用后不发预调用思考文本；无工具调用时一次性发送缓冲内容
- reasoning / thinking 流式不受影响，无 tools 路径流式不变

## [v2.1.0] — 2026-05-07

### Added
- **Anthropic 模型名映射** — Claude Code CLI 等工具可使用 Anthropic 风格模型名（如 `claude-sonnet-4-6`），内部自动映射为对应 DeepSeek 模型
  - `claude-opus-4-6` → `deepseek-expert-reasoner`（最强）
  - `claude-sonnet-4-6` → `deepseek-reasoner`（均衡）
  - `claude-haiku-4-5` → `deepseek-default`（快速）
  - 支持 search / nothinking 变体及 Claude 3.x 历史模型名
- DeepSeek 原生名（`deepseek-*`）继续直接可用，`/v1/models` 返回不变，不影响其他 OpenAI 兼容客户端

## [v2.0.0] — 2026-05-06

### Added
- **Anthropic Messages API 全兼容** — 新增 9 个 Anthropic 端点：`/v1/messages`（流式/非流式）、count_tokens、message CRUD、batch 全流程
- **多账号管理** — Web 面板增删账号、轮询负载均衡、401 自动重登
- **用量统计** — tiktoken 精确计数 + Web UI（表格/时间筛选）
- **会话管理** — 900K token 阈值自动续期，多账号独立追踪

### Changed
- 路由从 `proxy.py` 拆分为 `app/anthropic_routes.py`（APIRouter 模式）
- `app/anthropic.py` + `app/batch.py` 模块化，两分支共用 batch、分支差异在 anthropic.py

### Fixed
- 401 重登失败后账号未标记无效（死循环 bug）
- `relogin()` key 名不一致导致 token 写不回账号池
- Anthropic 路由：多账号未接入、ref_file_ids 硬编码为空
- Anthropic 路由：`tool_result` block 遗漏 + 工具定义未注入 prompt
- 路由顺序：`{message_id}` 在 batch 路由后，避免参数化匹配冲突

## [v1.1.0] — 2026-05-04

### Added
- **OpenAI Responses API 兼容层**（#2）— 新增 `/v1/responses` 端点家族，基于现有 DeepSeek 聊天流实现本地适配
  - `POST /v1/responses` — 创建 Response（流式/非流式）
  - `GET /v1/responses/{id}` — 查询 Response（支持 stream replay）
  - `DELETE /v1/responses/{id}` — 删除 Response
  - `GET /v1/responses/{id}/input_items` — 分页查询输入项
  - `POST /v1/responses/{id}/cancel` — 取消进行中的 Response
  - `POST /v1/responses/compact`、`POST /v1/responses/{id}/compact` — 多轮对话压缩
  - `POST /v1/responses/input_tokens` — 计算输入 Token
- **SSE 生命周期事件** — response.created → response.in_progress → response.completed，含 output_text.delta 逐 Token 流式输出
- **Structured Output** — 支持 `json_object` / `json_schema` 格式的 schema 验证与自动归一化
- **本地持久化** — `response_store.py` 本地 JSON 文件存储 Responses 记录，线程安全
- **function tool 兼容** — Responses API 函数调用与 chat completions 共享工具定义

### Changed
- **双分支同步更新** — main 和 no-tools 分支均已添加 Responses API 支持
- **no-tools 分支深度清理** — 完全移除 `tool_call.py` / `tool_dsml.py` / `tool_sieve.py` 引用，代码零工具调用残留

### Thank you
- **[@Acidmoon](https://github.com/Acidmoon)** — 提交 PR #2，实现完整的 Responses API 兼容层

## [v1.0.0] — 2026-05-04

### Added
- **工具调用（main 分支）** — DSML 格式 XML 工具提取 + 流式筛分（参考 ds2api 架构重构）
- **流式筛分** — 实时分离响应中的正文与工具调用内容
- **会话管理** — Token 阈值（90 万字符）自动检测并续接会话，超限自动新建
- **按模型上下文大小** — 从 DeepSeek API 的 `input_character_limit` 字段动态推算（大部分模型映射为 1M）
- **文本文件上传** — 使用 `ref_file_ids` 方式，与网页端行为一致（上传 → `wait_for_file_parsing` → 引用原始 file_id）
- **TikToken 用量统计** — Token 计数 + Web 面板可视化，固定表头/合计行的 440px 滚动表格
- **Expert 模型路由** — 通过 SSE ready 事件的 `model_type` 字段判断；路由失效时自动降级到 default，并提供诊断方案
- **Web 管理面板** — 用量统计 Tab，支持今日/本周/全部时间筛选和清空

### Changed
- **双分支架构** — `main`（工具调用）和 `no-tools`（纯对话）独立维护
- **SSE 解析器** — 修复 fragments metadata 和旧格式下第一个内容事件被丢弃的问题
- **deploy.sh** — 从硬编码包列表改为 `-r requirements.txt`，补充缺失的 `tiktoken` 依赖
- **代理逻辑重构** — 区分普通请求和视觉请求的路径

### Fixed
- **Token 过期静默降级** — expert 模型在 Token 过期后无声降级到 default，已提供检测方法（对比 expert/default 响应差异）和修复方案（重新登录）
- **model_type 字段不可靠** — ready 事件的 `model_type` 即使路由正确也可能显示"default"，已说明非确定性信号
- **TikToken 依赖缺失** — 补充到 deploy.sh 安装命令
- **视觉请求模型判断** — 从 `ref_file_ids` 非空启发式改为按模型名判断
- **UI 布局错位** — 用量面板移入 `.c` 容器，修复右偏显示问题
- **首个 SSE 内容丢失** — fragments metadata 解析现在正确捕获初始内容事件

### 已知问题
- Token 过期后静默降级（不返回错误，expert→default 退化）
- SSE ready 事件 `model_type` 字段不可靠，不能作为路由验证依据
- 不支持 Embeddings 端点
- 非原生 function calling（通过文本提示模拟）

---

## [0.x] — 初期开发阶段

项目初期的基本 DeepSeek API 代理功能（网页直接上传文件，无 git 历史记录）：
- OpenAI 兼容 `/v1/chat/completions`、`/v1/models` 端点
- 多账号轮询负载均衡
- Cookie / 凭证导入
- Think 块分离（`<think>`/`</think>`）
- Termux/Android 部署脚本

---

## 分支说明

| 分支 | 功能 |
|------|------|
| `main` | DSML 工具调用、流式筛分、会话管理、文件上传、用量统计 |
| `no-tools` | 纯对话代理 — 无 prompt 注入，输出更干净 |

纯对话、写作、翻译、代码生成等场景推荐使用 no-tools 分支。
