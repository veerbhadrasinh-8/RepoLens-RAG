// backend/server.js
const express = require("express");
const { authenticateUser } = require("./auth");
const { verifyAuthToken } = require("./middleware");
const db = require("./database");

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 4000;

app.get("/health", (req, res) => {
  res.json({ status: "healthy", timestamp: new Date().toISOString() });
});

app.post("/api/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    const result = await db.query("SELECT * FROM users WHERE email = $1", [email]);
    if (result.rows.length === 0) {
      return res.status(401).json({ error: "Invalid credentials" });
    }
    const authResult = authenticateUser(result.rows[0], password);
    res.json(authResult);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/api/tasks", verifyAuthToken, async (req, res) => {
  try {
    const result = await db.query("SELECT * FROM tasks WHERE user_id = $1", [req.user.userId]);
    res.json(result.rows);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
