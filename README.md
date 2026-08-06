# kira-ai-plugin-arxiv

arXiv 学术助手插件 v2.1.0 —— 由 `kira-ai-plugin-arxiv-search` 与 `pdf_translator`（论文翻译中文 PDF 引擎）二合一合并而来。

## 功能

- **arXiv 论文查询**：`/arxiv search <关键词>` 搜索论文，支持高级语法（`au:` / `ti:` / `cat:` / `abs:` 等）
- **论文详情**：`/arxiv get <ID>` 获取单篇论文完整信息
- **摘要翻译**：`/arxiv tr <ID>` 将标题与摘要翻译成中文（默认快速模型）
- **PDF 下载**：`/arxiv dl <ID>` 下载 PDF 到 `data/files/arxiv_pdf/`
- **LaTeX 源码下载**：`/arxiv src <ID>` 下载源码包到 `data/files/arxiv_src/`
- **论文翻译中文 PDF**（pdf_translate）：
  - 源码优先翻译（推荐）：`arxiv_id` / `tex_path` → 下载/读取源码 → 解压 → 只翻译正文 → xelatex 编译中文 PDF
  - PDF 直接翻译：`pdf_path` → 提取 → 分块 → 翻译（断点续传）→ 重组 Markdown → xelatex 编译
  - 长 PDF 自动转后台任务（返回任务 ID、进度可查、完成推送）

## LLM 工具

| 工具 | 说明 |
|------|------|
| `arxiv_search` | 按关键词搜索 arXiv 论文 |
| `arxiv_get` | 获取单篇论文详情 |
| `arxiv_translate` | 标题与摘要翻译成中文 |
| `arxiv_download` | 下载论文 PDF（支持批量） |
| `arxiv_src` | 下载 LaTeX 源码包 |
| `parse_arxiv_command` | 代为执行 `/arxiv` 斜杠命令 |
| `pdf_translate` | 论文翻译成中文 PDF（源码优先 + PDF 直接翻译） |
| `query_pdf_translate_task` | 查询后台翻译任务状态 |

## 斜杠命令

```
/arxiv search <关键词> [-t]   搜索论文（-t 附带标题译文）
/arxiv get <ID> [-t]         获取详情（-t 附带标题+摘要译文）
/arxiv tr <ID>               翻译标题与摘要
/arxiv dl <ID> [多个ID]      下载 PDF
/arxiv src <ID>              下载 LaTeX 源码
/arxiv task [任务ID]          查后台翻译任务进度（不传 ID 列出全部）
/arxiv help                  帮助
```

群聊使用需先 @ 机器人；私聊无需。

## 配置

- **arXiv 设置**：默认搜索条数、请求超时、User-Agent、排序、下载目录
- **翻译设置**：摘要翻译开关与目标语言
- **PDF 翻译**：翻译模型（model_select 下拉，默认快速模型 `3937f0fdf6b7:deepseek-v4-flash-0731`，留空回退旧字段/默认 LLM）、API Base URL/Key 覆盖（留空读对应 provider 配置）、分块大小、后台任务阈值、输出目录、Mineru 后端开关

## 依赖

```bash
pip install -r requirements.txt
```

**系统依赖：必须安装 TeX Live（完整版，含 `xelatex`、`ctex` 宏包与 `bibtex`）**，用于将翻译结果编译为中文 PDF。缺少任一组件会导致 PDF 编译失败：

- `xelatex`：主编译器（PDF 翻译与源码优先翻译均依赖）
- `ctex` 宏包：中文排版支持（缺少会报 `ctex.sty not found`）
- `bibtex`：参考文献编译（源码优先翻译保留参考文献时使用）

各平台安装参考：

```bash
# Ubuntu / Debian
apt install texlive-xetex texlive-lang-chinese texlive-bibtex-extra

# macOS（推荐完整版 MacTeX）
brew install --cask mactex

# Windows
# 安装 TeX Live 完整版：https://tug.org/texlive/
```

验证安装：

```bash
xelatex --version && kpsewhich ctex.sty
```

若 `kpsewhich ctex.sty` 无输出，说明缺少 ctex 宏包，请安装对应语言包。

## 许可

KiraAI Community
