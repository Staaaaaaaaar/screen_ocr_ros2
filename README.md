# PaddleOCR 管线仪屏幕识别

基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 与 OpenCV 的**管线探测仪（管线仪）屏幕信息识别**方案，针对固定型号设备的 LCD 界面，从实拍截图中解析仪表读数。

## 适用设备（占位）

本仓库标定配置仅适用于以下型号，部署到其他设备前请重新标定 `config/`：

| 项目 | 说明 |
|------|------|
| 设备名称 | `<DEVICE_NAME>` |
| 设备型号 | `<MODEL_NAME>` |
| 生产厂商 | `<MANUFACTURER>` |
| 屏幕规格 | `<SCREEN_WIDTH>` × `<SCREEN_HEIGHT>` |
| 固件版本 | `<FIRMWARE_VERSION>` |
| 备注 | `<NOTES>` |

## 识别字段

从管线仪屏幕截图中读取：

| 字段 | 说明 | 示例 |
|------|------|------|
| `signal_strength` | 信号强度 | `0.0%` |
| `pipeline_current` | 管线电流 | `350 mA` |
| `burial_depth` | 埋设深度 | `140 m` |
| `compass_angle` | 罗盘角度 | `61°` |
| `arrow_direction` | 箭头方向 | `none` / `left` / `right` / `both` |

识别流程：

- **数字字段**：七段 LCD 模板匹配（针对该型号屏幕标定）
- **罗盘指针**：OpenCV 边缘检测 + 霍夫直线
- **箭头图标**：OpenCV 轮廓检测
- 实拍图自动水平翻转（镜像校正）

技术栈：`paddlepaddle` + `paddleocr`（环境依赖）/ `opencv-python`（图像处理）

## 仓库结构

```
.
├── config/                 # 该型号设备的标定配置
│   ├── rois.json
│   ├── compass.json
│   └── digit_slots.json
├── assets/
│   └── digit_templates/    # 数字模板（首次运行自动生成）
├── examples/
│   └── images/             # 该型号示例截图
├── scripts/
│   ├── recognize.py        # CLI：读图输出 JSON
│   └── api_server.py       # HTTP API 服务
├── src/
│   └── meter_ocr/          # 识别核心库
├── output/                 # 调试输出（--debug 时生成）
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

## 环境安装

### conda（推荐）

```bash
conda env create -f environment.yml
conda activate ocr
```

### pip

```bash
conda create -n ocr python=3.10 -y
conda activate ocr
pip install -r requirements.txt
```

## PaddleOCR 模型

将随项目提供的 `.paddleocr.zip` 解压到仓库根目录：

```bash
unzip -o .paddleocr.zip -d .
```

## 使用

### 命令行识别

在仓库根目录执行：

```bash
python scripts/recognize.py
python scripts/recognize.py examples/images/image0000000.bmp
python scripts/recognize.py examples/images/image0000000.bmp --debug
```

### HTTP API

```bash
python scripts/api_server.py
```

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/v1/recognize \
  -F "image=@examples/images/image0000000.bmp"
```

## 配置与换型

| 文件 | 说明 |
|------|------|
| `config/rois.json` | 屏幕各字段区域 `[x1, y1, x2, y2]` |
| `config/compass.json` | 罗盘圆心与半径 |
| `config/digit_slots.json` | 七段数码管切片比例 |
| `assets/digit_templates/` | 数字匹配模板 |

更换管线仪型号时：

1. 更新上方「适用设备」信息
2. 重新标定 `config/` 下配置文件
3. 删除 `assets/digit_templates/` 后运行 `scripts/recognize.py` 重新生成模板

## 输出示例

```json
{
  "signal_strength": "0.0%",
  "pipeline_current": "350 mA",
  "burial_depth": "140 m",
  "arrow_direction": "none",
  "compass_angle": "61°",
  "compass_angle_deg": 61.0
}
```

## 核心依赖

| 包 | 版本 |
|----|------|
| Python | 3.10 |
| paddlepaddle | 2.6.2 |
| paddleocr | 2.7.0 |
| opencv-python | 4.6.0.66 |
| numpy | 1.23.5 |
| pillow | 10.0.0 |

完整列表见 `requirements.txt`。

## License

MIT
