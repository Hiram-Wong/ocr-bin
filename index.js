const express = require("express");
const multer = require("multer");
const os = require("os");

const { toImageBase64, toNumber } = require("./utils/format");
const { authMW } = require("./utils/auth");

const { ocrCaptchaService, OcrCaptchaService } = require("./captcha/ocr");
const { rotateCaptchaService } = require("./captcha/rotate");
const { slideCaptchaService } = require("./captcha/slide");

const pkg = require("./package.json");

process.on("uncaughtException", (err) => {
  console.error("[SYSTEM] 未捕获异常:", err);
});

process.on("unhandledRejection", (err) => {
  console.error("[SYSTEM] Promise异常:", err);
});

const PORT = toNumber(process.env.PORT, 7788);
const OCR_MODE = toNumber(process.env.OCR_MODE, 0); // 0-1
const OCR_RANGE = toNumber(process.env.OCR_RANGE, 7); // 0-7
const OCR_CHARSET =
  OCR_RANGE === 7 ? process.env.OCR_CHARSET || "0123456789+-x/" : undefined; // 字符集

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

const isPkg = (process?.pkg?.entrypoint ?? "").includes("snapshot");

const bootstrap = async () => {
  try {
    await ocrCaptchaService.init(OCR_MODE, OCR_RANGE, OCR_CHARSET);
    if (!ocrCaptchaService.ocrInstance) {
      throw new Error("OCR实例初始化失败");
    }

    app.set("trust proxy", true);
    app.use(express.json({ limit: "10mb" })); // max 10MB

    app.post(
      "/ocr",
      authMW.middleware(),
      upload.single("data"),
      async (req, res) => {
        try {
          const data = req.file || req.body?.data;
          const { range, mode, charset } = req.body;

          if (!data) {
            return res.status(400).send({ status: -1, msg: "缺少data字段" });
          }

          let ins = ocrCaptchaService.ocrInstance;
          if (range !== undefined || mode !== undefined) {
            OcrCaptchaService.getInstance().init(
              mode ?? OCR_MODE,
              range ?? OCR_RANGE,
              range === 7 ? charset || OCR_CHARSET : undefined,
            );
            ins = OcrCaptchaService.getInstance().ocrInstance;
          }

          const imageBase64 = await toImageBase64(data);
          const result = await ins.classification(imageBase64);
          console.debug(`[OCR] 识别结果: ${result}`);

          res.send({
            status: 0,
            data: { code: result },
            msg: "success",
          });
        } catch (err) {
          console.error("[OCR] 识别错误:", err);
          res.status(500).send({
            status: -1,
            msg: err.message || "识别失败",
          });
        }
      },
    );

    app.post(
      "/rotate",
      authMW.middleware(),
      upload.fields([
        { name: "thumb", maxCount: 1 },
        { name: "bg", maxCount: 1 },
      ]),
      async (req, res) => {
        try {
          const bgData = req.files?.["bg"]?.[0] || req.body?.bg;
          const thumbData = req.files?.["thumb"]?.[0] || req.body?.thumb;

          if (!bgData || !thumbData) {
            return res.status(400).send({
              status: -1,
              msg: "缺少bg或thumb字段",
            });
          }

          const [bgImgB64, thumbImgB64] = await Promise.all([
            toImageBase64(bgData),
            toImageBase64(thumbData),
          ]);

          const result = await rotateCaptchaService.main(thumbImgB64, bgImgB64);
          const cw = 360 - result;
          const ccw = result;
          console.debug(`[ROTATE] 识别结果: 顺时针-${cw}, 逆时针-${ccw}`);

          res.send({
            status: 0,
            data: { cw, ccw },
            msg: "success",
          });
        } catch (err) {
          console.error("[ROTATE] 识别错误:", err);
          res.status(500).send({
            status: -1,
            msg: err.message || "识别失败",
          });
        }
      },
    );

    app.post(
      "/slide",
      authMW.middleware(),
      upload.fields([
        { name: "thumb", maxCount: 1 },
        { name: "bg", maxCount: 1 },
      ]),
      async (req, res) => {
        try {
          let { simple = false, type = "match", thumb, bg } = req.body || {};
          if (req.files?.["thumb"]?.[0]) thumb = req.files["thumb"][0];
          if (req.files?.["bg"]?.[0]) bg = req.files["bg"][0];

          if (!thumb || !bg) {
            return res.status(400).send({
              status: -1,
              msg: "缺少thumb或bg字段",
            });
          }

          const [bgImgB64, thumbImgB64] = await Promise.all([
            toImageBase64(bg),
            toImageBase64(thumb),
          ]);

          let result;
          if (type === "comparison") {
            result = await slideCaptchaService.simpleComparison(
              thumbImgB64,
              bgImgB64,
            );
          } else if (type === "match") {
            result = await slideCaptchaService.simpleMatch(
              thumbImgB64,
              bgImgB64,
              simple,
            );
          }
          console.debug(`[SLIDE] 识别结果: x-${result.x}, y-${result.y}`);

          res.send({
            status: 0,
            data: { x: result.x, y: result.y },
            msg: "success",
          });
        } catch (err) {
          console.error("[SLIDE] 识别错误:", err);
          res.status(500).send({
            status: -1,
            msg: err.message || "识别失败",
          });
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
      res.status(404).send({
        status: -1,
        msg: "路径不存在",
      });
    });

    app.use((err, _req, res, _next) => {
      console.error(`[GLOBAL_ERROR] 捕获到未处理错误:`, err.stack || err);

      res.status(500).send({
        status: -1,
        msg: err.message || "Internal Server Error",
      });
    });

    app.listen(PORT, "0.0.0.0", () => {
      console.log("=".repeat(60));
      console.log(`验证码识别服务启动成功!`);
      console.log(`项目: ${pkg.homepage}`);
      console.log(
        `版本: ${pkg.version} | 系统: ${os.platform()} | 环境: ${isPkg ? "发行版" : "测试版"}`,
      );
      console.log(
        `地址: http://127.0.0.1:${PORT} | 认证: ${authMW.auth ? "已启用" : "未启用"}`,
      );
      console.log("=".repeat(60) + "\n");

      console.group("接口简述:");
      console.table([
        {
          说明: "通用验证码",
          路径: "/ocr",
          方法: "POST",
          参数: "data(必传), mode: 0-1, range: 0-7, charset",
        },
        {
          说明: "旋转验证码",
          路径: "/rotate",
          方法: "POST",
          参数: "thumb(必传), bg(必传)",
        },
        {
          说明: "滑动验证码",
          路径: "/slide",
          方法: "POST",
          参数: "thumb(必传), bg(必传), type: match/comparison",
        },
        { 说明: "健康检查", 路径: "/health", 方法: "GET", 参数: "无" },
      ]);
      console.groupEnd();

      console.group("调用说明:");
      console.table([
        {
          路径: "JSON",
          方法: "Content-Type: application/json (传 Base64 或 URL), bg)",
        },
        {
          路径: "Form",
          方法: "Content-Type: multipart/form-data (传 图片文件)",
        },
      ]);
      console.groupEnd();
      console.log("\n" + "=".repeat(60) + "\n");

      console.log("███████████████████████████████");
      console.log("█ ▄▄▄▄▄ █▀ █▀▀ █▀ ▀▄▄ █ ▄▄▄▄▄ █");
      console.log("█ █   █ █▀ ▄ █▄▄▀▀▀▄ ▄█ █   █ █");
      console.log("█ █▄▄▄█ █▀█ █▄▀██▀  ▄▀█ █▄▄▄█ █");
      console.log("█▄▄▄▄▄▄▄█▄█▄█ ▀ ▀▄█▄█ █▄▄▄▄▄▄▄█");
      console.log("█  ▄ ▄▀▄   ▄█▄▀▄ ▄ █ ▀ ▀ ▀▄█▄▀█");
      console.log("█▀▄▄▀▄▀▄█  ▀ ▄▄▀▀▄█ ▀ ▀▄▄ ▀█▀██");
      console.log("███▀▄▄█▄▄▀▄▀▄▀▀▀▄▀█▄ ▀▀▀▀▀▄▄█▀█");
      console.log("█▀ █ ██▄▄ ▀▄█▀▄▀▄▄█ ▀▄▄▄▀█▄▄▀██");
      console.log("█▀▀ █▄ ▄ ▀ ▄█▄▄ ▀▄▄ ▀▀█▀█▀▄ █▀█");
      console.log("█ █▀█  ▄██▀  ▄▄▀▄▄▀ ▀▀ ██▀█▄▀██");
      console.log("█▄████▄▄█  █▄ ▀ █▀▀▄▄ ▄▄▄ ▀   █");
      console.log("█ ▄▄▄▄▄ █▄▄██ ▀▀ █ █▄ █▄█ ▄▄███");
      console.log("█ █   █ █ ▀▀██▀▀▄██ ▀▄▄▄ ▄▀ ▄▄█");
      console.log("█ █▄▄▄█ █  ▄█ ▄▀▄▄▀ ▀  ▄   ▄ ██");
      console.log("█▄▄▄▄▄▄▄█▄▄▄████▄█▄█▄████▄▄▄███");
      console.log("\n" + "赞助: 支付宝扫描如上二维码请作者喝杯咖啡" + "\n");
    });
  } catch (err) {
    console.error("[SYSTEM] 启动失败:", err);
    process.exit(1);
  }
};

bootstrap();
