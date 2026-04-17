const crypto = require("crypto");

const { toNumber } = require("../utils/format");

const AUTH_CHARSET = process.env.AUTH_CHARSET || "";
const AUTH_TYPE = toNumber(process.env.AUTH_TYPE, 0);

class AuthMW {
  static instance = null;

  auth = !!AUTH_CHARSET;
  key = AUTH_CHARSET;
  type = AUTH_TYPE;

  static getInstance() {
    if (!AuthMW.instance) {
      AuthMW.instance = new AuthMW();
    }
    return AuthMW.instance;
  }

  makeToken() {
    const ts = Math.floor(Date.now() / 1000).toString();
    const nonce = crypto.randomBytes(16).toString("hex"); // 32位随机串
    const data = `${ts}:${nonce}:${this.key}`;
    const sig = crypto.createHash("md5").update(data).digest("hex");

    return `${ts}:${nonce}:${sig}`;
  }

  verifyToken(token, deadline = 3) {
    const parts = token.split(":");
    if (parts.length !== 3) return false;

    const [tsStr, nonce, sig] = parts;
    const ts = parseInt(tsStr, 10);
    if (
      isNaN(ts) ||
      Math.abs(Math.floor(Date.now() / 1000) - ts) > deadline * 60
    )
      return false;

    const data = `${tsStr}:${nonce}:${this.key}`;

    const expectedSig = crypto.createHash("md5").update(data).digest("hex");

    try {
      const isValid = crypto.timingSafeEqual(
        Buffer.from(sig, "hex"),
        Buffer.from(expectedSig, "hex"),
      );

      return isValid;
    } catch {
      return false;
    }
  }

  middleware() {
    return (req, res, next) => {
      if (!this.auth) return next();

      const authHeader = req.headers.authorization;
      if (!authHeader || !authHeader.startsWith("Bearer ")) {
        return res.status(401).json({ status: -1, msg: "认证失败" });
      }

      const token = authHeader.slice(7);

      try {
        const isValid =
          this.type === 0 ? this.key === token : this.verifyToken(token);
        if (isValid) return next();

        return res.status(401).send({
          status: -1,
          msg: "认证失败",
        });
      } catch (err) {
        return res.status(401).send({
          status: -1,
          msg: "认证失败",
        });
      }
    };
  }
}

const authMW = AuthMW.getInstance();

module.exports = {
  authMW,
};
