const { isFile, isHttp, isImageMime } = require("./validate");

const toImageBase64 = async (data) => {
  // File
  if (isFile(data)) {
    if (!isImageMime(data.mimetype)) {
      throw new Error("上传文件不是图片");
    }

    return `data:${data.mimetype};base64,${data.buffer.toString("base64")}`;
  }

  // URL
  if (isHttp(data)) {
    const res = await fetch(data, { signal: AbortSignal.timeout(10000) });
    if (!res.ok) {
      throw new Error("无法访问图片链接");
    }

    const contentType = res.headers.get("content-type") || "";
    if (!isImageMime(contentType)) {
      throw new Error("链接资源不是图片");
    }

    const buffer = await res.arrayBuffer();
    return `data:${contentType};base64,${Buffer.from(buffer).toString("base64")}`;
  }

  // base64（无前缀补全）
  if (!data.includes("base64,")) {
    return `data:image/png;base64,${data}`;
  }

  return data;
};

const toNumber = (val, def) => {
  const n = Number(val);
  return Number.isFinite(n) ? n : def;
};

module.exports = {
  toImageBase64,
  toNumber,
};
