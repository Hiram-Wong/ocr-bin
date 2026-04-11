const express = require("express");
const multer = require("multer");
const os = require("os");

const { toImageBase64, toNumber } = require("./utils/format");

const { rotateCaptchaService } = require("./captcha/rotate");
const { ocrCaptchaService } = require("./captcha/ocr");

const pkg = require("./package.json");

process.on("uncaughtException", (err) => {
  console.error("[SYSTEM] 未捕获异常:", err);
});

process.on("unhandledRejection", (err) => {
  console.error("[SYSTEM] Promise异常:", err);
});

const PORT = toNumber(process.env.PORT, 7788);
const AUTH = process.env.AUTH || "";
const OCR_MODE = toNumber(process.env.OCR_MODE, 0); // 0-1
const OCR_RANGE = toNumber(process.env.OCR_RANGE, 6); // 0-7
const OCR_CHARSET =
  OCR_RANGE === 7 ? process.env.OCR_CHARSET || "0123456789+-x/=" : undefined; // 字符集

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

const isPkg = (process?.pkg?.entrypoint ?? "").includes("snapshot");

const bootstrap = async () => {
  try {
    await ocrCaptchaService.init(OCR_MODE, OCR_RANGE, OCR_CHARSET);
    if (!ocrCaptchaService.ocrInstance) {
      throw new Error("OCR实例初始化失败");
    }

    app.use(express.json({ limit: "10mb" })); // max 10MB

    const authMiddleware = (req, res, next) => {
      if (!AUTH) return next();

      const authHeader = req.headers.authorization;
      const token = authHeader?.split(" ")?.[1];

      if (token && token === AUTH) return next();

      console.warn(`[AUTH] 拦截到非法请求: ${req.ip}`);
      res.status(401).send({ status: -1, msg: "认证失败" });
    };

    app.post(
      "/ocr",
      authMiddleware,
      upload.single("data"),
      async (req, res) => {
        try {
          const data = req.file || req.body?.data;

          if (!data) {
            return res.status(400).send({ status: -1, msg: "缺少data字段" });
          }

          const imageBase64 = await toImageBase64(data);
          const result =
            await ocrCaptchaService.ocrInstance.classification(imageBase64);
          console.debug(`[OCR] 识别结果: ${result}`);

          res.send({ status: 0, data: { code: result }, msg: "success" });
        } catch (err) {
          console.error("[OCR] 识别错误:", err);
          res.status(500).send({ status: -1, msg: err.message || "识别失败" });
        }
      },
    );

    app.post(
      "/rotate",
      authMiddleware,
      upload.fields([
        { name: "bg", maxCount: 1 },
        { name: "thumb", maxCount: 1 },
      ]),
      async (req, res) => {
        try {
          const bgData = req.files?.["bg"]?.[0] || req.body?.bg;
          const thumbData = req.files?.["thumb"]?.[0] || req.body?.thumb;

          if (!bgData || !thumbData) {
            return res
              .status(400)
              .send({ status: -1, msg: "缺少bg或thumb字段" });
          }

          const [bgImgB64, thumbImgB64] = await Promise.all([
            toImageBase64(bgData),
            toImageBase64(thumbData),
          ]);

          const result = await rotateCaptchaService.main(bgImgB64, thumbImgB64);
          const cw = 360 - result;
          const ccw = result;
          console.debug(`[ROTATE] 识别结果: 顺时针-${cw}, 逆时针-${ccw}`);

          res.send({ status: 0, data: { angle: { cw, ccw } }, msg: "success" });
        } catch (err) {
          console.error("[ROTATE] 识别错误:", err);
          res.status(500).send({ status: -1, msg: err.message || "识别失败" });
        }
      },
    );

    app.get("/health", (_req, res) => {
      res.send({
        status: 0,
        data: {
          version: pkg.version,
          timestamp: Date.now(),
        },
        msg: "ok",
      });
    });

    app.use((_req, res) => {
      res.status(404).send({ status: -1, msg: "路径不存在" });
    });

    app.use((err, _req, res, _next) => {
      console.error(`[GLOBAL_ERROR] 捕获到未处理错误:`, err.stack || err);

      res
        .status(500)
        .send({ status: -1, msg: err.message || "Internal Server Error" });
    });

    app.listen(PORT, "0.0.0.0", () => {
      console.log("\n" + "=".repeat(60));
      console.log(`🚀 验证码识别服务启动成功!`);
      console.log(
        `📦 版本: ${pkg.version} | 系统: ${os.platform()} | 环境: ${isPkg ? "发行版" : "测试版"}`,
      );
      console.log(`🌐 地址: http://127.0.0.1:${PORT}`);
      console.log(`🔒 认证: ${AUTH ? `已启用(Bearer ${AUTH})` : "未启用"}`);
      console.log("=".repeat(60));

      console.group("\n📝 接口文档简述:");
      console.table([
        { 路径: "/ocr", 方法: "POST", 说明: "通用验证码识别 (data)" },
        { 路径: "/rotate", 方法: "POST", 说明: "旋转验证码识别 (bg, thumb)" },
        { 路径: "/health", 方法: "GET", 说明: "健康检查" },
      ]);
      console.groupEnd();

      console.log("\n💡 调用说明:");
      console.log(
        "  - JSON: Content-Type: application/json (传 Base64 或 URL)",
      );
      console.log("  - Form: Content-Type: multipart/form-data (传 图片文件)");
      console.log("=".repeat(60) + "\n");
    });
  } catch (err) {
    console.error("[SYSTEM] 启动失败:", err);
    process.exit(1);
  }
};

bootstrap();
