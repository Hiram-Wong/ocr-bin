const { DdddOcr } = require("ddddocr-node");
const path = require("path");

class OcrCaptchaService {
  OCR_CHARSET_MAP = {
    0: "0123456789",
    1: "abcdefghijklmnopqrstuvwxyz",
    2: "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    3: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    4: "abcdefghijklmnopqrstuvwxyz0123456789",
    5: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    6: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    7: "0123456789+-x/",
  };

  static instance = null;
  ocrInstance = null;

  mode = 0;
  range = 6;
  charset = undefined;

  constructor() {}

  static getInstance() {
    if (!OcrCaptchaService.instance) {
      OcrCaptchaService.instance = new OcrCaptchaService();
    }
    return OcrCaptchaService.instance;
  }

  init(mode = 0, range = 6, charset = "") {
    this.mode = mode;
    this.range = range;
    this.charset = range === 7 ? charset : undefined;

    const ocrOnnxPath = path.join(
      __dirname,
      "../node_modules/ddddocr-node/onnx/",
    );
    // console.log(
    //   `[OCR] 配置 - 模型: ${mode}, 范围: ${range}, 模型路径: ${ocrOnnxPath}`,
    // );

    const ocr = new DdddOcr();

    ocr.setPath(ocrOnnxPath); // ONNX模型根路径
    ocr.setOcrMode(mode); // 模型 beta
    ocr.setRanges(
      range === 7 ? (charset ? charset : this.OCR_CHARSET_MAP[range]) : range,
    ); // 范围 0-6 或 自定义字符集

    this.ocrInstance = ocr;
  }

  switchModel(mode = 0, range = 6, charset = "") {
    this.mode = mode;
    this.range = range;
    this.charset = range === 7 ? charset : undefined;

    this.ocrInstance.setOcrMode(mode);
    this.ocrInstance.setRanges(
      range === 7 ? (charset ? charset : this.OCR_CHARSET_MAP[range]) : range,
    );
  }

  getRangeCharset() {
    return this.range === 7 && this.charset
      ? this.charset
      : this.OCR_CHARSET_MAP[this.range];
  }
}

const ocrCaptchaService = OcrCaptchaService.getInstance();

module.exports = {
  ocrCaptchaService,
  OcrCaptchaService,
};
