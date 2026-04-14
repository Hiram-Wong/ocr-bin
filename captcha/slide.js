/**
 * 滑块验证码识别
 *
 * @see https://github.com/sml2h3/ddddocr/blob/master/ddddocr/core/slide_engine.py
 */

const cv = require("@techstark/opencv-js");
const { Jimp } = require("jimp");

class SlideCaptchaService {
  static instance = null;

  constructor() {}

  static getInstance() {
    if (!SlideCaptchaService.instance) {
      SlideCaptchaService.instance = new SlideCaptchaService();
    }
    return SlideCaptchaService.instance;
  }

  async base64ToMat(base64) {
    const image = await Jimp.read(
      Buffer.from(base64.replace(/^data:image\/\w+;base64,/, ""), "base64"),
    );

    const { width, height, data } = image.bitmap;

    const matRGBA = new cv.Mat(height, width, cv.CV_8UC4);
    matRGBA.data.set(data);

    const matRGB = new cv.Mat();
    cv.cvtColor(matRGBA, matRGB, cv.COLOR_RGBA2RGB);

    matRGBA.delete();

    return matRGB;
  }

  toGray(mat) {
    const gray = new cv.Mat();
    cv.cvtColor(mat, gray, cv.COLOR_RGB2GRAY);
    return gray;
  }

  _simpleTemplateMatch(target_gray, background_gray) {
    const result = new cv.Mat();
    try {
      cv.matchTemplate(
        background_gray,
        target_gray,
        result,
        cv.TM_CCOEFF_NORMED,
      );

      const { maxLoc } = cv.minMaxLoc(result);

      return {
        x: maxLoc.x,
        y: maxLoc.y,
      };
    } finally {
      result.delete();
    }
  }

  _edgeBasedMatch(target_gray, background_gray) {
    const matsToDelete = [];

    try {
      // 高斯模糊
      const blurTarget = new cv.Mat();
      matsToDelete.push(blurTarget);
      const blurBg = new cv.Mat();
      matsToDelete.push(blurBg);

      cv.GaussianBlur(target_gray, blurTarget, new cv.Size(3, 3), 0);
      cv.GaussianBlur(background_gray, blurBg, new cv.Size(3, 3), 0);

      // Canny 边缘
      const edgeTarget = new cv.Mat();
      matsToDelete.push(edgeTarget);
      const edgeBg = new cv.Mat();
      matsToDelete.push(edgeBg);

      cv.Canny(blurTarget, edgeTarget, 100, 200);
      cv.Canny(blurBg, edgeBg, 100, 200);

      // 形态学增强
      const kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        new cv.Size(5, 5),
      );
      matsToDelete.push(kernel);

      cv.dilate(edgeTarget, edgeTarget, kernel, new cv.Point(-1, -1), 2);
      cv.dilate(edgeBg, edgeBg, kernel, new cv.Point(-1, -1), 2);

      const result = new cv.Mat();
      matsToDelete.push(result);

      cv.matchTemplate(edgeBg, edgeTarget, result, cv.TM_CCOEFF_NORMED);

      const { maxLoc } = cv.minMaxLoc(result);

      return {
        x: maxLoc.x,
        y: maxLoc.y,
      };
    } finally {
      for (const m of matsToDelete) {
        try {
          if (m && !m.isDeleted()) m.delete();
        } catch {}
      }
    }
  }

  async simpleMatch(thumbBase64, bgBase64, simple = false) {
    const matsToDelete = [];

    try {
      const thumb = await this.base64ToMat(thumbBase64);
      matsToDelete.push(thumb);

      const bg = await this.base64ToMat(bgBase64);
      matsToDelete.push(bg);

      if (!thumb || !bg) {
        throw new Error("图像加载失败");
      }

      // console.debug(
      //   `[SLIDE] 输入图像尺寸: thumb-${thumb.cols}x${thumb.rows}, bg-${bg.cols}x${bg.rows}`,
      // );

      const grayThumb = this.toGray(thumb);
      matsToDelete.push(grayThumb);

      const grayBg = this.toGray(bg);
      matsToDelete.push(grayBg);

      if (simple) {
        return this._simpleTemplateMatch(grayThumb, grayBg);
      } else {
        return this._edgeBasedMatch(grayThumb, grayBg);
      }
    } finally {
      for (const m of matsToDelete) {
        try {
          if (m && !m.isDeleted()) m.delete();
        } catch {}
      }
    }
  }

  async simpleComparison(thumbBase64, bgBase64) {
    const matsToDelete = [];

    try {
      const thumb = await this.base64ToMat(thumbBase64);
      matsToDelete.push(thumb);

      const bg = await this.base64ToMat(bgBase64);
      matsToDelete.push(bg);

      if (!thumb || !bg) {
        throw new Error("图像加载失败");
      }

      // console.debug(
      //   `[SLIDE] 输入图像尺寸: thumb-${thumb.cols}x${thumb.rows}, bg-${bg.cols}x${bg.rows}`,
      // );

      const grayThumb = this.toGray(thumb);
      matsToDelete.push(grayThumb);

      const grayBg = this.toGray(bg);
      matsToDelete.push(grayBg);

      // 差异
      const diff = new cv.Mat();
      matsToDelete.push(diff);
      cv.absdiff(grayThumb, grayBg, diff);

      // 二值化
      const thresh = new cv.Mat();
      matsToDelete.push(thresh);
      cv.threshold(diff, thresh, 50, 255, cv.THRESH_BINARY);

      // 形态学（增强轮廓）
      const kernel = cv.Mat.ones(3, 3, cv.CV_8U);
      matsToDelete.push(kernel);

      const morph = new cv.Mat();
      matsToDelete.push(morph);
      cv.morphologyEx(thresh, morph, cv.MORPH_CLOSE, kernel);

      // 找轮廓
      const contours = new cv.MatVector();
      matsToDelete.push(contours);

      const hierarchy = new cv.Mat();
      matsToDelete.push(hierarchy);

      cv.findContours(
        morph,
        contours,
        hierarchy,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
      );

      let maxArea = 0;
      let best = { x: 0, y: 0 };

      for (let i = 0; i < contours.size(); i++) {
        const cnt = contours.get(i);
        try {
          const rect = cv.boundingRect(cnt);
          const area = rect.width * rect.height;

          if (area > maxArea) {
            maxArea = area;
            best = rect;
          }
        } finally {
          cnt.delete();
        }
      }

      return {
        x: best.x,
        y: best.y,
      };
    } finally {
      for (const m of matsToDelete) {
        try {
          if (m && !m.isDeleted()) m.delete();
        } catch {}
      }
    }
  }
}

const slideCaptchaService = SlideCaptchaService.getInstance();

module.exports = {
  slideCaptchaService,
};
