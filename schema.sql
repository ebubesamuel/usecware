-- Initialize the database.
-- Drop any existing data and create empty tables.

DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS record;
-- DROP TABLE IF EXISTS post;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL
);

CREATE TABLE record (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  image1 TEXT NOT NULL,
  image2 TEXT NOT NULL ,
  result BOOLEAN NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
