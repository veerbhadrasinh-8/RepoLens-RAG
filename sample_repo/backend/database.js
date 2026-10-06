// backend/database.js
const { Pool } = require("pg");

/**
 * Initializes and manages the PostgreSQL database connection pool.
 */
const pool = new Pool({
  host: process.env.DB_HOST || "localhost",
  port: parseInt(process.env.DB_PORT || "5432", 10),
  database: process.env.DB_NAME || "taskflow_db",
  user: process.env.DB_USER || "postgres",
  password: process.env.DB_PASSWORD || "postgres",
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

pool.on("connect", () => {
  console.log("Database connection established successfully.");
});

pool.on("error", (err) => {
  console.error("Unexpected error on idle database client", err);
});

module.exports = {
  query: (text, params) => pool.query(text, params),
  pool,
};
