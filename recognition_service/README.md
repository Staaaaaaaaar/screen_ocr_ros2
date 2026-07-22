# 管线仪屏幕识别

基于 OpenCV 的**管线探测仪（管线仪）屏幕信息识别**方案，针对固定型号设备的 LCD 界面，从实拍截图中解析仪表读数。

本目录带有 `COLCON_IGNORE`，**不会**被上级 colcon 工作区编译；识别服务需在 conda 环境 `ocr` 中独立运行。

## 适用设备（占位）

本仓库标定配置仅适用于以下型号，部署到其他设备前请重新标定 `config/`（详见 `config/DEVICE.md`）：

| 项目 | 说明 |
|------|------|
| 设备名称 | `<DEVICE_NAME>` |
| 设备型号 | `<MODEL_NAME>` |
| 生产厂商 | `<MANUFACTURER>` |
| 屏幕规格 | `<SCREEN_WIDTH>` × `<SCREEN_HEIGHT>` |
| 固件版本 | `<FIRMWARE_VERSION>` |
| 备注 | `<NOTES>` |

## 识别字段

`POST /v1/recognize` 直接返回以下 **6 个字段**（数值型，`null` 表示该字段未识别）：

| 字段 | 说明 | 示例 |
|------|------|------|
| `signal_strength_percent` | 信号强度（%） | `5.7` |
| `current_milliamps` | 管线电流（mA） | `220` |
| `depth_meters` | 埋设深度（m） | `140` |
| `pipeline_heading_degrees` | 罗盘/管线方向（°） | `0.0` |
| `left_arrow` | 左箭头是否亮起 | `false` |
| `right_arrow` | 右箭头是否亮起 | `false` |

识别流程：

- **数字字段**：七段 LCD 模板匹配（启动时预加载 `assets/digit_templates/`）
- **罗盘指针**：圆心定位 + 径向边缘采样
- **箭头图标**：OpenCV 轮廓检测（`arrow_left` / `arrow_right` ROI）

技术栈：`opencv-python`（图像处理与模板匹配）

## 仓库结构

```
recognition_service/
├── config/                 # 该型号设备的标定配置
│   ├── rois.json
│   ├── compass.json
│   ├── digit_slots.json
│   └── DEVICE.md
├── assets/
│   └── digit_templates/    # 已标定数字模板
├── examples/
│   └── images/             # 示例截图 image0000001.png ~ image0000047.png
├── scripts/
│   ├── recognize.py        # CLI：读图输出 JSON
│   └── api_server.py       # HTTP API 服务
├── src/
│   └── screen_ocr/         # 识别核心库
├── output/                 # 调试输出（--debug 时生成单张 ROI 标注图）
├── requirements.txt
├── environment.yml
├── pyproject.toml
└── start_api_server.sh     # 启动 HTTP 服务（推荐）
```

## 环境安装

以下命令均在 `recognition_service/` 目录下执行。

### conda（推荐）

```bash
cd recognition_service
conda env create -f environment.yml   # 环境名: ocr
conda activate ocr
```

### pip

```bash
conda create -n ocr python=3.10 -y
conda activate ocr
pip install -r requirements.txt
```

## 使用

### 命令行识别

```bash
cd recognition_service
conda activate ocr

python scripts/recognize.py
python scripts/recognize.py examples/images/image0000001.png
python scripts/recognize.py examples/images/image0000001.png --debug
```

`--debug` 会在 `output/` 追加保存一张当前帧的 ROI 标注图（含识别结果文字）。

### 启动 HTTP 服务

```bash
cd recognition_service
conda activate ocr
./start_api_server.sh
# 或: python scripts/api_server.py
```

环境变量（可选）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `SCREEN_OCR_API_HOST` | `127.0.0.1` | 监听地址 |
| `SCREEN_OCR_API_PORT` | `8000` | 监听端口 |

### HTTP API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/v1/config` | 读取标定配置 |
| PUT | `/v1/config` | 更新标定配置 |
| POST | `/v1/recognize` | 上传图片识别（`image` 文件 + 可选 `debug`、`frame_id`） |

`POST /v1/recognize` 成功时返回上述 6 个扁平字段；失败时返回 HTTP 错误（如 `400` 空图/解码失败，`503` 未标定），body 为 FastAPI 标准 `{"detail": "..."}` 格式。

示例：

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/v1/recognize \
  -F "image=@examples/images/image0000001.png"
```

`/health` 响应示例：

```json
{
  "status": "ok",
  "model_loaded": true,
  "runtime": {"ready": true, "templates_loaded": true},
  "calibrated": true
}
```

## 配置与换型

| 文件 | 说明 |
|------|------|
| `config/rois.json` | 屏幕各字段区域 `[x1, y1, x2, y2]` |
| `config/compass.json` | 罗盘圆心与半径 |
| `config/digit_slots.json` | 七段数码管切片比例 |
| `assets/digit_templates/` | 数字匹配模板 |

`assets/digit_templates/` 中，信号强度模板命名为 `0_i_20.png`、`1_i_30.png` 这类形式（数字 + 可选 `_i` + 变体编号）；电流与深度模板命名为 `0_1.png`、`1_2.png` 或 `0.png`、`1.png` 这类形式；小数点模板命名为 `dot.png`。

更换管线仪型号时：

1. 更新 `config/DEVICE.md` 与上方「适用设备」信息
2. 重新标定 `config/` 下配置文件
3. 重新制作并替换 `assets/digit_templates/` 下的数字模板
4. 重启 HTTP 服务

## 输出示例

以 `examples/images/image0000001.png` 为例：

```json
{
  "signal_strength_percent": 5.7,
  "depth_meters": null,
  "current_milliamps": 220,
  "pipeline_heading_degrees": 0.0,
  "left_arrow": false,
  "right_arrow": false
}
```

## 核心依赖

| 包 | 版本 |
|----|------|
| Python | 3.10 |
| opencv-python | 4.6.0.66 |
| numpy | 1.23.5 |
| fastapi | 0.115.6 |
| uvicorn | 0.32.1 |

完整列表见 `requirements.txt` / `pyproject.toml`。

## License

MIT
