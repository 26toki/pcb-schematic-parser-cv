# 外部资产与运行边界

本公开候选仅包含项目编写的源码，不提供下列文件，也不授权其再分发：

| 原运行目录中的路径 | 用途 |
|---|---|
| assets/yolo.pt | YOLO元件检测权重 |
| assets/BoxICLibrary.json | 官方IC元件与引脚参考库 |
| assets/size_priors.json | 元件几何尺寸先验 |
| assets/symbol_pin_priors.json | 符号引脚位置先验 |
| bundle_manifest.json | 原提交包文件完整性与权重校验清单 |

其他未发布内容：官方训练图片及标注、公开GT、队友收集数据、平台结果包、基础镜像与离线依赖wheel。

入口agent_system保留原容器路径/submission/code/submission_entry.py，源码主要按项目根目录寻找assets。复现需要自行获得合法资产并配置相应环境。不能用任意类别定义的YOLO权重直接替换原权重并期待同样输出；类别映射、先验格式与代码必须兼容。

requirements-extra.txt是原赛事基础环境上的附加依赖，不含完整PyTorch环境，且部分版本可能受包源可用性影响。当前公开候选尚未完成脱离赛事基础镜像的独立安装验证。

请勿将API密钥写入源码或提交仓库。CV默认关闭LLM，可选客户端仅从运行配置取得参数。
