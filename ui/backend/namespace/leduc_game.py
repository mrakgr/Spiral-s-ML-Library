from typing import Any, Callable, Never, TypedDict, Literal

from flask import request
from flask_socketio import Namespace, emit # type: ignore

from game.leduc.play import main
funs = main()

class Leduc_Namespace(Namespace):
    user_state : dict[str, Any] = {}

    def sid(self) -> str: return request.sid # type: ignore
    def emit_update(self, data: Any): emit('update', data)

    def on_connect(self):
        print(f'Client connected to Leduc game: {self.sid()}')
        state = funs.init()
        Leduc_Namespace.user_state[self.sid()] = state
        self.emit_update(state["game"]["public"])

    def on_disconnect(self):
        Leduc_Namespace.user_state.pop(self.sid())
        print(f'Client disconnected: {self.sid()}')

    def on_update(self, msg : Any):
        state = Leduc_Namespace.user_state[self.sid()]
        state = funs.event_loop_gpu(msg,state)
        Leduc_Namespace.user_state[self.sid()] = state
        self.emit_update(state["game"]["public"])
