import datetime
import errno
import logging
import queue
import select
import socket
import struct
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

import matplotlib
import matplotlib.ticker as plt_ticker
import numpy as np
import psutil
from matplotlib import pyplot as plt

from denylist import DENY_LIST

from ppo import PPO, PolicyNetwork, ValueNetwork

matplotlib.use("Qt5agg")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remote_connection_closed(sock: socket.socket) -> bool:
    """
    Returns True if the remote side did close the connection

    """
    try:
        buf = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
        if buf == b'':
            return True
    except BlockingIOError as exc:
        if exc.errno != errno.EAGAIN:
            # Raise on unknown exception
            raise
    except OSError:
        return True
    
    return False


class PlaybackState (Enum):
    PLAYING = 0
    PAUSED = 1
    FAST_FORWARD = 2


class RecvMessageType (Enum):
    ACTIVE_FILES = 0
    STATE_INACTIVE = 1
    PLAYBACK_RATE = 2
    STATE_PLAYING = 3
    STATE_PAUSED = 4
    STATE_FAST_FORWARD = 5
    CURRENT_TICK = 6
    DEBUG_TICK = 7
    MESSAGE = 8
    PROCESSED_SCRIPT = 10
    ENTITY_INFO = 100
    GAME_LOCATION = 255


class EntityInfo:
    state: bool

    x: float
    y: float
    z: float

    pitch: float
    yaw: float
    roll: float

    vx: float
    vy: float
    vz: float

    def dist_sq(self, x=None, y=None, z=None, yaw=None, pitch=None, roll=None, vx=None, vy=None, vz=None) -> float:
        """
        Compute the distance squared to the given coords.
        Be careful not to compare values for which you have given
        different input counts/types!
        """
        dist = 0

        if x is not None:
            dist += (self.x - x)**2
        if y is not None:
            dist += (self.y - y)**2
        if z is not None:
            dist += (self.z - z)**2
        if pitch is not None:
            dist += (self.pitch - pitch)**2
        if yaw is not None:
            dist += (self.yaw - yaw)**2
        if roll is not None:
            dist += (self.roll - roll)**2
        if vx is not None:
            dist += (self.vx - vx)**2
        if vy is not None:
            dist += (self.vy - vy)**2
        if vz is not None:
            dist += (self.vz - vz)**2

        return dist

    def __str__(self) -> str:
        res = "EntityInfo: \n"
        if self.state:
            res += f"\tpos: [{self.x}, {self.y}, {self.z}]\n"
            res += f"\tang: [{self.pitch}, {self.yaw}, {self.roll}]\n"
            res += f"\tvel: [{self.vx}, {self.vy}, {self.vz}]\n"
        else:
            res += "\tEntity not found\n"
        return res


class GameInstanceBase:
    """
    Generic building block for the two possible communication scenarios
    SAR server, this client or
    SAR client, this server
    """

    def __init__(self):
        # The socket field should be initialized by the parent class
        self.sock: socket.socket

        # Server state
        self.active_scripts: List[str] = []
        self.state: PlaybackState = PlaybackState.PLAYING
        self.playback_rate: float = 1.0
        self.current_tick: int = 0
        self.debug_tick: int = 0
        self.game_location = ""
        
        self.messages: List[str] = []

    def set_sock(self, sock: socket.socket):
        """
        Sets this server to communicate with the given socket
        The purpose of this 
        """
        self.sock = sock

    # =============================================================
    #                             Send
    # =============================================================

    def start_file_playback(self, script_path1: str, script_path2: str = ""):
        """
        Request a playback of the given file(s)
        """
        packet = b''
        packet += struct.pack("!B", 0)
        packet += struct.pack("!I", len(script_path1))
        packet += script_path1.encode()
        packet += struct.pack("!I", len(script_path2))
        packet += script_path2.encode()

        self.sock.send(packet)
        self.current_tick = 0

    def start_content_playback(self, script1: str, script2: str = ""):
        """
        Request a playback of the given scripts
        """
        script1_name = "script1"
        script2_name = "script2" if script2 != "" else ""
        packet = b''
        packet += struct.pack("!B", 10)
        for string in [script1_name, script1, script2_name, script2]:
            packet += struct.pack("!I", len(string))
            packet += string.encode()

        self.sock.send(packet)

    def stop_playback(self):
        """
        Request for the playback to stop
        """
        self.sock.send(struct.pack("!B", 1))

    def change_playback_speed(self, speed: float):
        """
        Request a change to the playback speed
        """
        packet = b''
        packet += struct.pack("!B", 2)
        packet += struct.pack("!f", speed)

        self.sock.send(packet)

    def resume_playback(self):
        """
        Request for the playback to resume
        """
        self.sock.send(struct.pack("!B", 3))

    def pause_playback(self):
        """
        Request for the playback to pause
        """
        self.sock.send(struct.pack("!B", 4))

    def fast_forward(self, to_tick=0, pause_after=True):
        """
        Request fast-forward
        """
        packet = b''
        packet += struct.pack("!B", 5)
        packet += struct.pack("!I", to_tick)
        packet += struct.pack("!?", pause_after)

        self.sock.send(packet)

    def pause_at(self, tick=0):
        """
        Request for the playback to pause at the given tick
        """
        packet = b''
        packet += struct.pack("!B", 6)
        packet += struct.pack("!I", tick)

        self.sock.send(packet)

    def advance_playback(self):
        """
        Request for the playback to advance a single tick
        """
        self.sock.send(struct.pack("!B", 7))

    def entity_info(self, entity_selector="player"):
        """
        Request information on an entity, player is default
        """
        packet = b''
        packet += struct.pack("!B", 100)
        packet += struct.pack("!I", len(entity_selector))
        packet += entity_selector.encode()

        self.sock.send(packet)

    def entity_info_continuous(self, entity_selector="player"):
        """
        Request continuous information on an entity, player is default
        """
        packet = b''
        packet += struct.pack("!B", 101)
        packet += struct.pack("!I", len(entity_selector))
        packet += entity_selector.encode()

        self.sock.send(packet)

    def entity_info_continuous_stop(self):
        """
        Stop recieving contiuous entity info
        """
        self.entity_info_continuous("")

    # =============================================================
    #                            Receive
    # =============================================================

    def readable(self) -> bool:
        readable, w, e = select.select([self.sock], [], [], 0)
        return len(readable) > 0

    def receive(self) -> List[EntityInfo]:
        """
        receive all pending data from the server. Non blocking.
        """
        entity_info_list = []

        while self.readable():
            # read data
            _, ent_info = self.__recv_blocking()
            if ent_info is not None:
                entity_info_list.append(ent_info)

        return entity_info_list

    def receive_until(self, message_type=RecvMessageType.CURRENT_TICK) -> List[EntityInfo]:
        """
        receive all pending data and return once a specific packet type is received. Data may still be left unread.
        """
        entity_info_list = []

        while True:
            # read data
            msg_type, ent_info = self.__recv_blocking()

            if ent_info is not None:
                entity_info_list.append(ent_info)

            if msg_type == message_type:
                return entity_info_list

    def __recv_blocking(self):
        ent_info = None
        msg_type = self.__recv(1)
        msg_type = RecvMessageType(struct.unpack("!B", msg_type)[0])

        logging.debug(msg_type)

        if msg_type == RecvMessageType.ACTIVE_FILES:
            len1 = struct.unpack("!I", self.__recv(4))[0]
            if len1 > 0:
                self.active_scripts.append(self.__recv(len1).decode())
            len2 = struct.unpack("!I", self.__recv(4))[0]
            if len2 > 0:
                self.active_scripts.append(self.__recv(len2).decode())
        elif msg_type == RecvMessageType.STATE_INACTIVE:
            self.active_scripts.clear()
        elif msg_type == RecvMessageType.PLAYBACK_RATE:
            self.playback_rate = struct.unpack("!f", self.__recv(4))[0]
        elif msg_type == RecvMessageType.STATE_PLAYING:
            self.state = PlaybackState.PLAYING
        elif msg_type == RecvMessageType.STATE_PAUSED:
            self.state = PlaybackState.PAUSED
        elif msg_type == RecvMessageType.STATE_FAST_FORWARD:
            self.state = PlaybackState.FAST_FORWARD
        elif msg_type == RecvMessageType.CURRENT_TICK:
            self.current_tick = struct.unpack("!I", self.__recv(4))[0]
        elif msg_type == RecvMessageType.DEBUG_TICK:
            self.debug_tick = struct.unpack("!i", self.__recv(4))[0]
        elif msg_type == RecvMessageType.MESSAGE:
            message_len = struct.unpack("!i", self.__recv(4))[0]
            message = self.__recv(message_len).decode()
            self.messages.append(message)
        elif msg_type == RecvMessageType.PROCESSED_SCRIPT:
            self.active_scripts.clear()
            slot = struct.unpack("!B", self.__recv(1))[0]
            raw_script_len = struct.unpack("!I", self.__recv(4))[0]
            raw_script = self.__recv(raw_script_len)
        elif msg_type == RecvMessageType.ENTITY_INFO:
            ent_info = EntityInfo()
            info_state = struct.unpack("!B", self.__recv(1))[0]
            ent_info.state = info_state != 0

            if ent_info.state:  # The rest of the data is present
                ent_data = struct.unpack("!fffffffff", self.__recv(4*9))
                ent_info.x = ent_data[0]
                ent_info.y = ent_data[1]
                ent_info.z = ent_data[2]
                ent_info.pitch = ent_data[3]
                ent_info.yaw = ent_data[4]
                ent_info.roll = ent_data[5]
                ent_info.vx = ent_data[6]
                ent_info.vy = ent_data[7]
                ent_info.vz = ent_data[8]

        elif msg_type == RecvMessageType.GAME_LOCATION:
            str_len = struct.unpack("!I", self.__recv(4))[0]
            if str_len > 0:
                self.game_location = self.__recv(str_len).decode()

        return (msg_type, ent_info)

    def __recv(self, size: int, timeout: float = 30) -> bytes:
        """
        receive size bytes from the socket and return them.
        """
        before = time.time()
        after = time.time()

        buf = b''
        while len(buf) < size:
            if self.readable():
                buf += self.sock.recv(size - len(buf))
            else:
                if (after - before) > 5:
                    logging.warn(f"({self.sock.getpeername()}) client is being slow!")
                time.sleep(.5)
            after = time.time()

            if (after - before) > timeout:
                raise socket.timeout

        return buf

    def __str__(self) -> str:
        res = "TasServer: \n"
        res += f"\tactive_scripts: {str(self.active_scripts)}\n"
        res += f"\tstate: {str(self.state)}\n"
        res += f"\tplayback_rate: {str(self.playback_rate)}\n"
        res += f"\tcurrent_tick: {str(self.current_tick)}\n"
        res += f"\tdebug_tick: {str(self.debug_tick)}\n"
        res += f"\tgame_location: {str(self.game_location)}\n"
        return res


class TasServer(GameInstanceBase):
    def __init__(self, ip="127.0.0.1", port=6555):
        super().__init__()

        self.ip = ip
        self.port = port

    def connect(self):
        """
        Initiate a connection to the server
        """
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.connect((self.ip, self.port))


class TasClient(GameInstanceBase):
    def __init__(self, sock):
        super().__init__()
        self.sock = sock


MeasureFn = Callable[[GameInstanceBase, List[float]], float]
NewBestAction = Callable[[str, List[float], float], None]

class ClientPool:
    def __init__(self, script_template: str, init_script: str, measure_fn: Callable, on_policy_update: Union[Callable, None] = None) -> None:
        self.script_template = script_template
        self.init_script = init_script
        self.measure_fn = measure_fn
        self.on_policy_update = on_policy_update

        # Queue for tasks and results
        self.task_queue = queue.Queue()
        self.results_queue = queue.Queue()

        # List of client threads
        self.clients: Dict[socket._RetAddress, ClientHandlerThread] = {}

        # Various statistics
        self.rewards_hist: List[float] = []
        self.iter_end_times: List[float] = []

    def listen(self, port: int):
        self.sock = socket.socket(socket.AF_INET)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", port))
        self.sock.listen(10)

        # Start a thread to accept client connections
        threading.Thread(target=self.handle_connections, daemon=True).start()

    def handle_connections(self):
        while True:
            try:
                conn, addr = self.sock.accept()
                if addr[0] in DENY_LIST:
                    conn.close()
                    continue
                client_thread = ClientHandlerThread(
                    script_template=self.script_template,
                    init_script=self.init_script,
                    target=self.measure_fn,
                    sock=conn,
                    task_queue=self.task_queue,
                    results_queue=self.results_queue
                )
                client_thread.run()
                self.clients[addr] = client_thread
                conn.settimeout(60)
                logging.info(f"Client connected from {addr}. Total clients: {len(self.clients)}")
            except Exception as e:
                logging.exception("The connection accept thread crashed. Restarting.")
                break

    def distribute_tasks(self, tasks: List[Tuple[int, List[float]]]):
        for task in tasks:
            self.task_queue.put(task)

    def gather_results(self) -> List[Tuple[int, float]]:
        results = []
        while not self.results_queue.empty():
            result = self.results_queue.get()
            results.append(result)
        return results

    def run_optimization(self, tasks: List[Tuple[int, List[float]]]):
        self.distribute_tasks(tasks)
        while self.task_queue.unfinished_tasks > 0:
            time.sleep(1)
        return self.gather_results()

    def shutdown(self):
        for client in self.clients.values():
            client.close()
        self.sock.close()
        logging.info("ClientPool shutdown.")

class ClientDataTracker:
    def __init__(self) -> None:
        self.timestamps: List[float] = []
        self.client_counts: Dict[str, List[int]] = {} # ip->count
        self.client_names: Dict[str, str] = {} # ip->name

    def set_client_count(self, new_counts: Dict[str, int], timestamp: float):
        self.timestamps.append(timestamp)
        self.timestamps.append(timestamp)
        for client in self.client_counts:
            self.client_counts[client].append(self.client_counts[client][-1])
            self.client_counts[client].append(new_counts.get(client, 0))
        
        for client in new_counts:
            if not self.client_counts.get(client):
                self.client_counts[client] = [0] * (len(self.timestamps)-1)
                self.client_counts[client].append(new_counts[client])
    
    def set_client_name(self, ip: str, name: str):
        if name != "Unknown":
            self.client_names[ip] = name
    
    def get_client_name(self, ip: str) -> str:
        return self.client_names.get(ip, "Unknown")

    def get_total_clients(self) -> int:
        sum = 0
        for counts in self.client_counts.values():
            sum += counts[-1]
        return sum

    def get_client_counts(self) -> Dict[str, int]:
        res = {}
        for ip, count in self.client_counts.items():
            res[self.get_client_name(ip)] = count
        return res

    def get_timestamps(self) -> List[float]:
        return self.timestamps

class ClientHandlerThread(threading.Thread):
    def __init__(self, script_template: str, init_script: Union[str, None], target: Callable, sock: socket.socket, task_queue: queue.Queue, results_queue: queue.Queue, init_every: int = 200) -> None:
        super().__init__(daemon=True)
        self.init_script = init_script
        self.script_template = script_template
        self._target = target
        self.task_queue = task_queue
        self.results_queue = results_queue
        self.game = TasClient(sock)  # Assuming TasClient is defined elsewhere
        self.init_every = init_every
        self.contributor = "Unknown"

    def run(self):
        total_runs = 0
        while True:
            if self.init_every != 0 and total_runs % self.init_every == 0:
                try:
                    self.run_init()
                except Exception as e:
                    logging.exception(f"{self.contributor} client failed to run init script! Dropping.")
                    break
            try:
                
                # Get the next task
                task = self.task_queue.get()
                parameters = task['parameters']


                # Fill in the script with the parameters
                filled_script = self.script_template.format(*parameters)

                # Run the filled script and get the result
                score = self._target(self.game, filled_script)
                self.results_queue.put((task, score))
                total_runs += 1
            except Exception as e:
                logging.exception(f"{self.contributor} failed to run a task")
                self.task_queue.put(task)  # Put the task back in the queue in case of failure
                self.task_queue.task_done()
                break

            self.task_queue.task_done()

        logging.info(f"{self.contributor} exited.")

    def run_init(self):
        if self.init_script is not None:
            self.game.receive()
            self.game.stop_playback()
            self.game.fast_forward()  # reset fast forward
            time.sleep(0.1)
            self.game.receive()
            self.game.start_content_playback(self.init_script)
            self.game.receive_until(RecvMessageType.PROCESSED_SCRIPT)

        for message in self.game.messages:
            if message.startswith("name: "):
                self.contributor = message[6:30].strip()

        self.game.messages.clear()

    def close(self):
        self.game.sock.close()

class ClientDataTracker:
    def __init__(self) -> None:
        self.timestamps: List[float] = []
        self.client_counts: Dict[str, List[int]] = {} # ip->count
        self.client_names: Dict[str, str] = {} # ip->name

    def set_client_count(self, new_counts: Dict[str, int], timestamp: float):
        self.timestamps.append(timestamp)
        self.timestamps.append(timestamp)
        for client in self.client_counts:
            self.client_counts[client].append(self.client_counts[client][-1])
            self.client_counts[client].append(new_counts.get(client, 0))
        
        for client in new_counts:
            if not self.client_counts.get(client):
                self.client_counts[client] = [0] * (len(self.timestamps)-1)
                self.client_counts[client].append(new_counts[client])
    
    def set_client_name(self, ip: str, name: str):
        if name != "Unknown":
            self.client_names[ip] = name
    
    def get_client_name(self, ip: str) -> str:
        return self.client_names.get(ip, "Unknown")

    def get_total_clients(self) -> int:
        sum = 0
        for counts in self.client_counts.values():
            sum += counts[-1]
        return sum

    def get_client_counts(self) -> Dict[str, int]:
        res = {}
        for ip, count in self.client_counts.items():
            res[self.get_client_name(ip)] = count
        return res

    def get_timestamps(self) -> List[float]:
        return self.timestamps