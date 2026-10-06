// backend/auth.js
const jwt = require("jsonwebtoken");
const bcrypt = require("bcryptjs");

const JWT_SECRET = process.env.JWT_SECRET || "supersecretkey";

/**
 * Authenticates user credentials and generates a signed JWT token.
 * Validates email and password against the stored user record.
 */
function authenticateUser(user, password) {
  const isMatch = bcrypt.compareSync(password, user.passwordHash);
  if (!isMatch) {
    throw new Error("Invalid email or password");
  }

  // JWT token is generated after successful login
  const token = jwt.sign(
    { userId: user.id, email: user.email, role: user.role },
    JWT_SECRET,
    { expiresIn: "2h" }
  );

  return { token, user: { id: user.id, email: user.email } };
}

module.exports = { authenticateUser, JWT_SECRET };
