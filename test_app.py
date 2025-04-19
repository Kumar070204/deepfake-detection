from flask import Flask

def test_app_runs():
    app = Flask(__name__)
    assert app is not None