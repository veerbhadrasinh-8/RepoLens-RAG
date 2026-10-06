// backend/middleware.js
const jwt = require("jsonwebtoken");
const { JWT_SECRET } = require("./auth");

/**
 * Express middleware for verifying JWT tokens on protected routes.
 * The JWT token is verified before protected routes are accessed.
 */
function verifyAuthToken(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ error: "Access denied. No token provided." });
  }

  const token = authHeader.split(" ")[1];
  try {
    // Verify token signature and expiration
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    return res.status(403).json({ error: "Invalid or expired token." });
  }
}

module.exports = { verifyAuthToken };
