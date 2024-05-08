import unittest
from flask import Flask, session, g
from camera_flask_app import app, get_db, init_db, close_db, IsFaceMatching
from unittest.mock import patch
import os, tempfile


class AuthActions(object):
    def __init__(self, client):
        self._client = client

    def register(self, username="test", password="test", follow_redirects=True):
        return self._client.post(
            "/register",
            data={"username": username, "password": password},
            follow_redirects=follow_redirects,
        )

    def login(self, username="test", password="test", follow_redirects=True):
        return self._client.post(
            "/login",
            data={"username": username, "password": password},
            follow_redirects=follow_redirects,
        )

    def logout(self, follow_redirects=True):
        return self._client.get("/logout", follow_redirects=follow_redirects)


class UserAuthTestCase(unittest.TestCase):
    def setUp(self):
        """Set up a test client before each test."""
        self.db_fd, self.db_path = tempfile.mkstemp()

        self.app = app
        self.app.config["TESTING"] = True
        self.app.config["DATABASE"] = self.db_path

        self.client = self.app.test_client()
        self.auth = AuthActions(self.client)

        with self.app.app_context():
            init_db()

    def tearDown(self):
        """Clean up after each test."""
        os.close(self.db_fd)
        # os.unlink(self.db_path)

    def test_register(self):
        response = self.auth.register(username="ebubesam", password="bube123")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please login to your account", response.data)

        with self.app.app_context():
            db = get_db()
            user = db.execute(
                "SELECT username, password" " FROM user WHERE username='ebubesam'"
            ).fetchone()

            self.assertIsNotNone(user)
            self.assertEqual(user["username"], "ebubesam")
            self.assertTrue(user["password"].startswith("scrypt"))

    def test_login(self):
        self.auth.register(username="ebubesam", password="bube123")
        response = self.auth.login(username="ebubesam", password="bube123")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Capture", response.data)

        with self.client:
            self.client.get("/")
            self.assertEqual(session["user_id"], 1)
            self.assertEqual(g.user["username"], "ebubesam")

    def test_login_required(self):
        response = self.client.get("/records")
        self.assertEqual(response.location, "/login")

    def test_logout(self):
        self.auth.register(username="ebubesam", password="bube123")
        self.auth.login(username="ebubesam", password="bube123")

        with self.client:
            response = self.auth.logout()
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Capture", response.data)
            self.assertNotIn("user_id", session)

class FaceMatchTestCase(unittest.TestCase):
    def test_face_match(self):
        img1 = "tests/test_images/Me_pic_1.JPG"
        img2 = "tests/test_images/me_pic_2.JPG"
        self.assertTrue(IsFaceMatching(img1, img2))

    def test_face_mismatch(self):
        img1 = "tests/test_images/me_pic_3.JPG"
        img2 = "tests/test_images/Ngo_pic_1.jpg"
        self.assertFalse(IsFaceMatching(img1, img2))
        
if __name__ == "__main__":
    unittest.main()
    