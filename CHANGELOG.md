# 更新日志（Changelog）

本文件记录 deepseek-free-api 的所有重要变更。

---

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
