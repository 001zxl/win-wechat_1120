# 微信 AI 自动回复 · 客服控制台

一个可运行的微信 AI 客服 MVP：会话同步、聊天记录读取、人设/知识库/工具/护栏、草稿与人工确认发送，以及项目内模拟接待与多 IM 通道扩展。

> 支持 Windows 微信 PC（3.x / 4.x）与 macOS（默认只读，不操作微信）。

## 快速开始（Windows）

```bash
pip install -r requirements.txt
copy config.example.yaml config.yaml        # 填 api_key 与 whitelist
python run.py doctor                        # 环境自检：微信窗口、DPI、OCR、模型
python run.py capture --out shot.png        # 抓一张窗口图，校准布局
python server.py                            # 打开浏览器聊天控制台（默认 dry-run）
python server.py --live                     # 确认无误后再开真实发送
python run.py run --once                    # 无界面跑一轮（调试用）
```

控制台打开后你会看到：
左侧会话列表、中间聊天记录 + 可编辑草稿 + 发送/重新生成/转人工、右侧工作进度、底部状态栏。
浏览器地址默认 `http://127.0.0.1:8765/`。

macOS（仅开发/验证逻辑，不连真微信）：

```bash
python3 run.py mock            # 合成窗口跑全链路（dry-run）
python3 run.py mock --live     # 模拟真实发送（在 mock 世界里）
python3 -m unittest discover -s tests -v
```

## 聊天控制台（浏览器）

`python server.py` 会启动本地控制台并自动打开浏览器（默认 `http://127.0.0.1:8765/`）：

- **左栏**：检测到的会话与状态（草稿待发 / 已发送 / 已跳过 / 转人工）
- **中栏**：聊天记录（对方/我方气泡 + 草稿气泡）、可编辑草稿框、发送/重新生成/转人工按钮
- **右栏**：工作进度（扫描未读、草稿就绪、发送结果、跳过原因）
- **底栏**：模型/模式/人设/知识库/白名单状态
- 顶部：暂停自动回复、允许真实发送开关

界面结构：会话列表 + 聊天区 + 工作进度。

## 测试模式：默认虚拟平台（不碰真实微信）

```bash
python server.py --demo        # 虚拟微信：头像/会话/未读/自动处理，全部本地
python server.py --real        # 连接真实微信（默认只读，不点击不发送）
```

虚拟平台（mock）用程序生成的窗口和头像，所有点击/发送都在内存里完成，**不会碰你的真实微信账号**，适合演示、联调和回归测试。真实微信只在 `--real` 时使用。

## macOS 真实微信（已跑通读取+草稿）

macOS 没有 Windows 的 `PrintWindow/PostMessage`，项目改用另一套后端：

- 抓图：`screencapture -l <窗口ID>`（窗口被遮挡也能抓）
- 文字：macOS 自带 **Vision OCR**（无需 RapidOCR）
- 输入：先 `open -a WeChat` 唤醒可能被隐藏/最小化的微信窗口，再用 HID 事件点击/按键，完成后光标归位
- 权限：**屏幕录制** + **辅助功能**（系统设置 → 隐私与安全性）

安装与启动：

```bash
# 一键：建 venv、装 pyobjc、检查权限、启动（暂停自动扫描）
./setup-mac.command
# 或手动
data/venv/bin/python server.py --paused
```

打开控制台后：

- 在微信里**手动打开一个联系人的聊天**（当前会话）
- 左栏白名单加入该联系人（或点会话行「+ 白名单」）
- 点顶栏 **「处理当前会话」**：读取真实消息 → 生成草稿（dry-run 不发送）
- 填 API Key 后草稿就是真实模型输出
- 确认无误后勾「允许真实发送」，再点「发送草稿」才会真正发出

自动扫描默认 **暂停**：安全模式下只处理你手动打开的会话；开启自动扫描时，程序会先用 OCR 认会话名，只点白名单会话，认不出名字的会跳过（避免误点、误标已读）。

**会话列表同步**：即使暂停，也每 2 秒只读地 OCR 微信会话列表，把联系人、**真实头像**、最后一条消息、时间、未读数量同步到控制台（不点击任何会话）。头像从窗口截图裁剪，通过 `/api/avatar` 接口提供；也可以点顶栏「同步会话」立即刷新。白名单只控制"是否自动回复"，不影响列表显示。

## Windows 首次启动（3 步）

1. 双击 **setup.bat**：安装依赖 + RapidOCR + 自动跑 doctor 自检
2. 双击 **start.bat**：打开控制台
3. 点顶栏 **「启动检查」**，按 6 项清单逐个「去处理」：
   - **RapidOCR**：setup.bat 已装；缺失时按提示 `pip install rapidocr_onnxruntime`
   - **布局校准**：点「校准布局」→ 选「微信 4.x / 3.x 预设」→ 微调滑块 → 保存
   - **白名单**：左侧会话行点「+ 白名单」，或底部输入框添加
   - **模型 Key**：点「设置」填 Key

全部变绿后，先 dry-run 看草稿，再勾「允许真实发送」。

## 设置面板（不用手改配置文件）

控制台右上角两个按钮：

- **设置**：每个模型单独填 API Key（切换缺 Key 的模型会自动弹出），配置视觉模型、Embedding 模型；保存后立即生效。
- **校准布局**：抓当前窗口截图，拖动 4 个滑块（图标栏/会话列表/标题栏/输入区），彩色框实时贴合，保存后立即生效。
- **白名单**：左栏"白名单"里直接增删联系人，回车即保存，立即生效。

所有改动写入 `data/settings.json`（不改 `config.yaml`，保留注释）：

```json
{ "api_keys": {"qwen": "sk-..."}, "whitelist": ["张三"], "layout": {"nav_ratio": 0.06}, "vision_model": "qwen-vl-plus" }
```

Windows 一键安装：[setup.bat](setup.bat)（装依赖 + RapidOCR + doctor 自检）；日常启动双击 [start.bat](start.bat)。

## 打不开 http://127.0.0.1:8765/ 怎么办

这个地址是**本地服务**，不是常驻网站——必须先启动进程，关掉窗口/终端就没了。

1. 确认在跑：浏览器访问 `http://127.0.0.1:8765/api/health`，返回 `{"ok":true}` 才正常
2. 没跑就启动：`python server.py`（Windows 可双击 `start.bat`，macOS 双击 `start.command`）
3. 端口被占：换端口 `python server.py --port 8766`，然后访问新地址
4. 关掉终端就没了：想常驻可以用 `nohup python server.py &`，或 Windows 建个计划任务/快捷方式

## v0.3 新增能力

| 能力 | 实现 | 状态 |
| --- | --- | --- |
| 模型切换 | 顶栏下拉框，`ai.models` 配置多模型，运行时热切换 | ✅ 已验证 |
| 图片理解 | 消息模型支持 `image`；Windows 检测图片块裁图 → 视觉模型描述；未配置视觉模型时标注"[图片]" | ✅ mock 验证 / Windows 待校准 |
| 语音/文件 | 识别为 `voice` / `file` 消息类型，标注"[语音消息]（未转写）"；语音转写需后续接 ASR | ✅ 类型标注 |
| 群聊回复 | `allow_group_chat` + 群白名单 + `mention/keyword/all` 三种放行模式 | ✅ 单测覆盖 |
| 长程记忆 | SQLite 存最近消息/摘要/客户档案，每 12 条自动压缩摘要 | ✅ 已验证 |
| 工具调用 | `data/price_list.json` 报价/库存；支持 function calling，失败自动降级为"预取注入" | ✅ dry-run 已出报价 |
| 向量 RAG | `embedding_model` 配置后 hybrid（向量 + BM25 RRF），不可用自动回退 BM25 | ✅ 代码就绪 |
| 发送重试 | 未确认时先读回确认（幂等），再最多重试 `send_retries` 次 | ✅ 单测覆盖 |

配置对应项：`ai.models` / `ai.vision_model` / `ai.embedding_model` / `runtime.allow_group_chat` / `runtime.group_*` / `memory.*` / `rag.*` / `tools.*`。

## 工作原理（一轮循环）

```
抓窗口(PrintWindow) → 找未读红点 → 点开会话(PostMessage)
  → 读消息（结构化/OCR + 连发聚合）→ 白名单&群聊过滤
  → 人设 + 知识库(RAG) + 模型生成
  → 护栏检查（敏感词/越界/长度/转人工）
  → 发送安全链（header校验→抢前台校验→粘贴回车→还原→差分验收）
  → 审计落库
```

## 安全设计（必须理解）

1. **默认 dry-run**：配置文件 `runtime.dry_run: true`，只生成草稿写审计，不发消息。
2. **白名单物理闸门**：不在 `safety.whitelist` 的会话一律跳过；群聊/公众号按名称特征拦截。
3. **发送前三重校验**：header OCR 必须等于目标会话 → 抢前台后 `GetForegroundWindow` 必须是我们 → 发送后差分验收。
4. **不确定就放弃**：验收 `UNCERTAIN` 重试 2 次仍失败则跳过并记录，绝不盲发。
5. **限频**：每会话冷却 + 全局滑动窗口，防触发平台风控。

## 目录

```
run.py                  CLI：doctor / capture / mock / run
server.py               浏览器聊天控制台（HTTP + 轮询状态）
web/index.html          控制台页面（会话 / 聊天 / 草稿 / 工作进度 / 模型切换）
data/price_list.json    报价与库存（工具数据）
wxreply/brain/memory.py 长程记忆（SQLite）
wxreply/brain/tools.py  报价/库存/线索工具
wxreply/brain/vision.py 图片理解（视觉模型）
wxreply/perception/media.py 图片块检测（Windows）
wxreply/config.py       配置加载
wxreply/platform/       平台层：base(协议) / win32 / mock
wxreply/perception/     layout / unread / ocr / chat
wxreply/brain/          persona / llm / rag
wxreply/executor/       sender（安全链）/ keyboard
wxreply/policy.py       白名单、护栏、限频
wxreply/agent.py        主循环
wxreply/audit.py        SQLite 审计 + 证据截图
config.example.yaml     配置模板
personas/               人设
kb/                     知识库（放 .md/.txt）
tests/                  离线单测（macOS 可跑）
scripts/make_mock_frames.py  生成 mock 截图
```

## 上线前 5 步

1. 微信登录，`run.py capture` 抓图核对布局比例（不同版本/DPI 可能要微调 `layout`）
2. 白名单填真实联系人名称（与微信显示名一致，支持 OCR 容错变体）
3. `run.py doctor` 全绿（RapidOCR 能装就装，装不上只能跑结构化/mock）
4. 先 `dry_run: true` 跑 2~3 天，人工看审计里的草稿质量
5. 人设评测通过后开 `--live`，从 1 个白名单好友开始，观察 24h

## 风险与边界

- 个人微信自动化违反平台使用条款，**有封号风险**；建议小号、白名单、低频、自用。
- 本工具不读微信数据库、不注入进程、不改微信文件（与开源参考一致），但仍属非官方自动化。
- 截图与聊天属敏感数据，默认本地保存；调用外部模型前会经过护栏过滤。
