const express = require('express');
const multer = require('multer');
const { DdddOcr } = require('ddddocr-node');
const path = require('path');
const os = require('os');
const pkg = require('./package.json');

process.on('uncaughtException', (err) => {
    console.error('[SYSTEM] 未捕获异常:', err);
});

process.on('unhandledRejection', (err) => {
    console.error('[SYSTEM] Promise异常:', err);
});

const getEnvNumber = (val, def) => {
    const n = Number(val);
    return Number.isFinite(n) ? n : def;
};

const PORT = getEnvNumber(process.env.PORT, 7788);
const OCR_MODE = getEnvNumber(process.env.OCR_MODE, 0); // 0-1
const OCR_RANGE = getEnvNumber(process.env.OCR_RANGE, 6); // 0-7
const OCR_CHARSET = OCR_RANGE === 7 ? (process.env.OCR_CHARSET || '0123456789+-x/=') : undefined; // 字符集

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

let ocrInstance = null;

const isPkg = (process?.pkg?.entrypoint ?? '').includes('snapshot');
console.log(`[INFO] 运行环境: ${os.platform()}${isPkg ? '打包环境' : '开发环境'}`);

const ocrCharsetMap = {
    0: '0123456789',
    1: 'abcdefghijklmnopqrstuvwxyz',
    2: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    3: 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
    4: 'abcdefghijklmnopqrstuvwxyz0123456789',
    5: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    6: 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
};
const ocrCharset = (OCR_RANGE === 7 ? OCR_CHARSET : ocrCharsetMap[OCR_RANGE]) ?? 'unknown';

const isHttpUrl = (str) => /^https?:\/\//i.test(str);
const isImageMime = (str) => /^image\//i.test(str);

const normalizeInput = async (data, file) => {
  // 文件上传
  if (file) {
    if (!isImageMime(file.mimetype)) {
      throw new Error('上传文件不是图片');
    }

    return `data:${file.mimetype};base64,${file.buffer.toString('base64')}`;
  }

  // URL
  if (isHttpUrl(data)) {
    const res = await fetch(data, { signal: AbortSignal.timeout(10000) });
    if (!res.ok) {
      throw new Error('无法访问图片链接');
    }

    const contentType = res.headers.get('content-type') || '';
    if (!isImageMime(contentType)) {
      throw new Error('链接资源不是图片');
    }

    const buffer = await res.arrayBuffer();
    return `data:${contentType};base64,${Buffer.from(buffer).toString('base64')}`;
  }

  // base64（无前缀补全）
  if (!data.includes('base64,')) {
    return `data:image/png;base64,${data}`;
  }

  return data;
}

const initOcr = async () => {
    const ocrOnnxPath = path.join(__dirname, 'node_modules/ddddocr-node/onnx/');
    console.log(`[OCR] 配置 - 模型: ${OCR_MODE}, 范围: ${OCR_RANGE} (${ocrCharset}), 模型路径: ${ocrOnnxPath}`);

    const ocr = new DdddOcr();
    ocr.setPath(ocrOnnxPath); // ONNX模型根路径
    ocr.setOcrMode(OCR_MODE); // 模型 beta
    ocr.setRanges(OCR_RANGE === 7 ? OCR_CHARSET : OCR_RANGE); // 范围 0-6 或 自定义字符集

    return ocr;
}

const bootstrap = async () => {
    try {
        ocrInstance = await initOcr();
        if (!ocrInstance) {
            throw new Error('实例初始化失败');
        }

        app.use(express.json({ limit: '10mb' })); // max 10MB

        app.post('/ocr', upload.single('data'), async (req, res) => {
            try {
                let { data } = req.body || {};
                const file = req.file;

                if (!data && !file) {
                    return res.status(400).json({ status: -1, msg: '缺少 data 或文件' });
                }

                const input = await normalizeInput(data, file);
                const result = await ocrInstance.classification(input);
                console.debug('[OCR] 识别结果:', result);
                
                res.send({ status: 0, data: { code: result }, msg: 'success' });
            } catch (err) {
                console.error('[OCR] 识别错误:', err);
                res.status(500).send({ status: -1, msg: err.message || '识别失败' });
            };
        });

        app.get('/health', (_req, res) => {
            res.send({
                status: 0,
                data: {
                    version: pkg.version,
                    charset: ocrCharset,
                    timestamp: Date.now()
                },
                msg: 'ok'
            });
        });

        app.listen(PORT, '0.0.0.0', () => {
            console.log(`[OCR] 请求地址: http://127.0.0.1:${PORT}/ocr`);
            console.log(`[OCR] 使用方式: POST`);
            console.log(`[OCR] 请求方式一:\n      请求头: {"Content-Type": "application/json"}\n      请求体: {"data": "图片链接/图片base64字符串"}`);
            console.log(`[OCR] 请求方式二:\n      请求头: {"Content-Type": "multipart/form-data"}\n      请求体: {"data": "图片文件"}`);
        });
    } catch (err) {
        console.error('[SYSTEM] 启动失败:', err);
        process.exit(1);
    }
}

bootstrap();
