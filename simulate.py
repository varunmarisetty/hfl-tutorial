import subprocess
import yaml
import os
import shutil
import platform
import time
import socket
import requests
from config import TOPOLOGY_FILE
import config
import random

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EXP_ID = f"experiment_{random.randint(1000, 9999)}"


def get_abs_path(filename):
    """Get the absolute path of a file in the same directory."""
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️ Warning: {filename} not found at {path}")
    return path


def get_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def spawn_processes():
    topo_file = get_abs_path(f"topologies/{TOPOLOGY_FILE}")

    if not os.path.exists(topo_file):
        print(f"❌ Error: topo.yml not found at {topo_file}")
        return

    with open(topo_file, "r") as file:
        topology = yaml.safe_load(file)

    current_os = platform.system()

    # calculate the min number of clients per edge
    edge_client_counts = {}
    total_edges = 0

    # Initialize counts for all defined edges
    for name, cfg in topology.items():
        if cfg.get("kind") == "edge":
            edge_client_counts[name] = 0
            total_edges += 1

    # Count clients for each edge by matching ports
    for name, cfg in topology.items():
        if cfg.get("kind") == "client":
            target_port = cfg.get("port")
            # Find the edge listening on this port
            for edge_name, edge_cfg in topology.items():
                if edge_cfg.get("kind") == "edge":
                    if edge_cfg["client"]["port"] == target_port:
                        edge_client_counts[edge_name] += 1

    print(f"Topology Analysis:")
    print(f"  - Central Server expects: {total_edges} edges")
    for edge, count in edge_client_counts.items():
        print(f"  - {edge} expects: {count} clients")

    # Resolve missing ports and host references
    # 1. Assign default port to coordinator/server if not specified
    for name, cfg in topology.items():
        if cfg.get("kind") == "server":
            if not cfg.get("port"):
                auto_port = get_free_port()
                print(f"Assigning free port {auto_port} to server {name}")
                cfg["port"] = auto_port

    # 2. Resolve edge configurations
    for name, cfg in topology.items():
        if cfg.get("kind") == "edge":
            # Server side
            svr = cfg.get("server", {})
            ref = svr.get("host")
            if ref in topology:
                target = topology[ref]
                svr_host = target.get("host")
                svr_port = target.get("port")
            else:
                svr_host = ref
                svr_port = svr.get("port") or get_free_port()
            cfg["server"]["host"] = svr_host
            cfg["server"]["port"] = svr_port
            print(f"Edge {name} server -> {svr_host}:{svr_port}")

            # Client side
            cli = cfg.get("client", {})
            cli_host = cli.get("host")
            cli_port = cli.get("port") or get_free_port()
            cfg["client"]["host"] = cli_host
            cfg["client"]["port"] = cli_port
            print(f"Edge {name} client -> {cli_host}:{cli_port}")

    # 3. Resolve client configurations
    for name, cfg in topology.items():
        if cfg.get("kind") == "client":
            ref = cfg.get("host")
            if ref in topology and topology[ref].get("kind") == "edge":
                edge_cli = topology[ref]["client"]
                cfg["host"] = edge_cli.get("host")
                cfg["port"] = edge_cli.get("port")
            elif ref in topology and topology[ref].get("kind") == "server":
                server = topology[ref]
                cfg["host"] = server.get("host")
                cfg["port"] = server.get("port")
            else:
                # direct host, ensure port exists
                if not cfg.get("port"):
                    raise ValueError(f"Port not specified for client {name}")
            print(f"Client {name} -> {cfg['host']}:{cfg['port']}")

    # Sort by kind order
    order = {"server": 0, "edge": 1, "client": 2}
    sorted_topo = dict(
        sorted(topology.items(), key=lambda item: order.get(item[1].get("kind"), 99))
    )

    # Spawn processes per OS
    if current_os == "Windows":
        commands = []
        for name, cfg in sorted_topo.items():
            kind = cfg.get("kind")
            if kind == "server":
                cmd = f'py "{get_abs_path("central_server.py")}" {cfg["host"]}:{cfg["port"]} --exp_id {EXP_ID}'
            elif kind == "edge":
                required_clients = edge_client_counts.get(name, 1)
                # Ensure at least 1 to avoid logical errors if topology is empty
                required_clients = max(1, required_clients)
                cmd = (
                    f'py "{get_abs_path("edge_server.py")}" --server '
                    f'{cfg["server"]["host"]}:{cfg["server"]["port"]} --client '
                    f'{cfg["client"]["host"]}:{cfg["client"]["port"]} --name {name} --exp_id {EXP_ID} --min_clients {required_clients}'
                )
            elif kind == "client":
                cmd = (
                    f'py "{get_abs_path("client.py")}" '
                    f'{cfg["host"]}:{cfg["port"]} --partition_id {cfg["partition_id"]} '
                    f"--name {name} --exp_id {EXP_ID}"
                )
            else:
                continue

            commands.append(
                f'new-tab --title "{name}" -p "Command Prompt" cmd /k {cmd}'
            )

        if not shutil.which("wt"):
            print("❌ Error: Windows Terminal (wt) is not installed or not in PATH.")
            return

        full_command = f'wt {" ; ".join(commands)}'
        subprocess.run(full_command, shell=True)

    elif current_os == "Linux":
        procs = []
        for name, cfg in sorted_topo.items():
            kind = cfg.get("kind")
            if kind == "server":
                cmd = f'python3 "{get_abs_path("central_server.py")}" {cfg["host"]}:{cfg["port"]} --exp_id {EXP_ID} --min_edges {total_edges}'
            elif kind == "edge":
                required_clients = edge_client_counts.get(name, 1)
                # Ensure at least 1 to avoid logical errors if topology is empty
                required_clients = max(1, required_clients)
                cmd = (
                    f'python3 "{get_abs_path("edge_server.py")}" --server '
                    f'{cfg["server"]["host"]}:{cfg["server"]["port"]} '
                    f'--client {cfg["client"]["host"]}:{cfg["client"]["port"]} --name {name} --exp_id {EXP_ID} --min_clients {required_clients}'
                )
            elif kind == "client":
                cmd = (
                    f'python3 "{get_abs_path("client.py")}" '
                    f'{cfg["host"]}:{cfg["port"]} --partition_id {cfg["partition_id"]}'
                    f" --name {name} --exp_id {EXP_ID}"
                )
            else:
                continue

            proc = subprocess.Popen(cmd, shell=True)
            procs.append((name, proc))
            print(f"Starting process {name} with command: {cmd}")
            if kind == "server":
                # give server time to initialize
                time.sleep(30)

        while procs:
            for name, p in procs[:]:
                if p.poll() is not None:
                    print(f"❌ Process {name} has ended")
                    procs.remove((name, p))

            if len(procs) == 0:
                break
            time.sleep(5)

    else:
        print(f"❌ Unsupported OS: {current_os}")


if __name__ == "__main__":
    spawn_processes()
