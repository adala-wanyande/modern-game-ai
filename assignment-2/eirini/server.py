import logging
import os
import typing

from flask import Flask
from flask import request

from logger import BattlesnakeDirectLogger

LOGGER = BattlesnakeDirectLogger(out_dir="logs")

# Added port parameter with a default of None
def run_server(handlers: typing.Dict, port: int = None):
    app = Flask("Battlesnake")

    @app.get("/")
    def on_info():
        return handlers["info"]()

    @app.post("/start")
    def on_start():
        game_state = request.get_json()
        LOGGER.start_game(game_state)
        LOGGER.log_turn(game_state)
        handlers["start"](game_state)
        return "ok"

    @app.post("/move")
    def on_move():
        game_state = request.get_json()
        LOGGER.log_turn(game_state)
        return handlers["move"](game_state)

    @app.post("/end")
    def on_end():
        game_state = request.get_json()
        LOGGER.end_game(game_state)
        handlers["end"](game_state)
        return "ok"

    @app.after_request
    def identify_server(response):
        response.headers.set("server", "battlesnake/github/starter-snake-python")
        return response

    host = "0.0.0.0"
    
    # Priority: 1. Argument passed to function, 2. ENV variable, 3. Default 8000
    if port is None:
        port = int(os.environ.get("PORT", "8000"))

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print(f"\nRunning {handlers['info']().get('name', 'Snake')} at http://{host}:{port}")
    app.run(host=host, port=port)