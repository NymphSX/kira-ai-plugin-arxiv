# kira-ai-plugin-arxiv

arXiv 学术助手插件 v2.0.0 —— 由 `kira-ai-plugin-arxiv-search`、`kira-ai-plugin-translate`、`kira-ai-plugin-pdf-gen` 三合一合并而来。

## 功能

- **arXiv 论文查询**：`/arxiv search <关键词>` 搜索论文，支持高级语法（`au:` / `ti:` / `cat:` / `abs:` 等）
- **论文详情**：`/arxiv get <ID>` 获取单篇论文完整信息
- **摘要翻译**：`/arxiv tr <ID>` 将标题与摘要翻译成中文（默认快速模型，可切换翻译引擎后端）
- **PDF 下载**：`/arxiv dl <ID>` 下载 PDF 到 `data/files/arxiv_pdf/`
- **LaTeX 源码下载**：`/arxiv src <ID>` 下载源码包到 `data/files/arxiv_src/`
- **多后端文本翻译**：百度 / DeepL / Google / 阿里云 / 本地模型，自动语言检测与后端回退
- **PDF 生成**：将文本内容排版生成 PDF 文件

## LLM 工具

| 工具 | 说明 |
|------|------|
| `arxiv_search` | 按关键词搜索 arXiv 论文 |
| `arxiv_get` | 获取单篇论文详情 |
| `arxiv_translate` | 标题与摘要翻译成中文 |
| `arxiv_download` | 下载论文 PDF（支持批量） |
| `arxiv_src` | 下载 LaTeX 源码包 |
| `parse_arxiv_command` | 代为执行 `/arxiv` 斜杠命令 |
| `translate` | 多后端文本翻译 |
| `generate_pdf` | 生成 PDF 文件 |

## 斜杠命令

```
/arxiv search <关键词> [-t]   搜索论文（-t 附带标题译文）
/arxiv get <ID> [-t]         获取详情（-t 附带标题+摘要译文）
/arxiv tr <ID>               翻译标题与摘要
/arxiv dl <ID> [多个ID]      下载 PDF
/arxiv src <ID>              下载 LaTeX 源码
/arxiv help                  帮助
```

群聊使用需先 @ 机器人；私聊无需。

## 配置

- **arXiv 设置**：默认搜索条数、请求超时、排序、下载目录
- **翻译设置**：`translate_backend` 选择摘要翻译后端（`fast`=快速模型 / `auto`=翻译引擎回退链 / 指定后端）
- **翻译引擎**：百度/DeepL/Google/阿里云/本地模型的密钥与额度限流
- **PDF 生成**：开关与输出目录

## 安装依赖

```bash
pip install -r requirements.txt
```

阿里云后端需要额外安装 `aliyun-python-sdk-core` 与 `aliyun-python-sdk-alimt`。

## 许可

KiraAI Community
