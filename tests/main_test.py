import unittest
from flask import Flask, session, g
from camera_flask_app import app, get_db, init_db
from unittest.mock import patch
import os, tempfile

class UserAuthTestCase(unittest.TestCase):

    def setUp(self):
        """Set up a test client before each test."""
        # with open(os.path.join(os.path.dirname(__file__), 'data.sql'), 'rb') as f:
        #     _data_sql = f.read().decode('utf8')
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['DATABASE'] = self.db_path
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        with self.app.app_context():
            init_db()
        #     get_db().executescript(_data_sql)
                
            
        # self.app = app
        # self.app.config['TESTING'] = True
        # self.app.config['DATABASE'] = 'sqlite:///:memory:'
        

        # Set up the database here, if necessary, you might need to import additional helpers to initialize the DB
        # with self.app.app_context():
        #     init_db()
        #     db = get_db()
            # Setup database schema (Assuming you have a function to initialize it)
            # db.execute(SQL_SCHEMA)

    def tearDown(self):
        """Clean up after each test."""
        self.app_context.pop()
        os.close(self.db_fd)
        os.unlink(self.db_path)   
        
        
    def test_register(self):
        response = self.client.post('/register', data={
            'username': 'ebubesam',
            'password': 'bube123'
            
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Please login to your account', response.data)
        
        with self.app.app_context():
            db = get_db()
            user= db.execute(
            "SELECT username, password"
            " FROM user WHERE username='ebubesam'"
            ).fetchone()

            #user = User.query.filter_by(username='testuser').first()
            self.assertIsNotNone(user)
            self.assertEqual(user.username, 'ebubesam')
            print(user.password)
            self.assertTrue(user.password.startswith('$2'))
        
if __name__ == '__main__':
    unittest.main()
