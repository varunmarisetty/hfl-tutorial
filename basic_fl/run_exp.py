import subprocess
import time
import sys
import os
import shutil
import platform
import config

def get_python_cmd():
    """Returns the python executable for the current environment."""
    return sys.executable

def get_abs_path(filename):
    """Returns absolute path to ensure Windows Terminal finds the file."""
    return os.path.abspath(filename)

def run_experiment():
    current_os = platform.system()
    print(f"Detected OS: {current_os}")
    print(f"--- Starting Experiment with {config.NUM_CLIENTS} Clients ---")

    if current_os == "Windows":
        commands = []
        
        # 1. Server Command
        server_script = get_abs_path("server.py")
        server_cmd = f'"{get_python_cmd()}" "{server_script}"'
        
        # Add server tab (cmd /k keeps the window open if it crashes)
        commands.append(
            f'new-tab --title "FL Server" -p "Command Prompt" cmd /k {server_cmd}'
        )

        # 2. Client Commands
        client_script = get_abs_path("client.py")
        for cid in range(config.NUM_CLIENTS):
            # We pass the CID as an argument
            client_cmd = f'"{get_python_cmd()}" "{client_script}" {cid}'
            commands.append(
                f'new-tab --title "Client {cid}" -p "Command Prompt" cmd /k {client_cmd}'
            )

        # 3. Check for Windows Terminal
        if not shutil.which("wt"):
            print("Error: Windows Terminal (wt) is not installed or not in PATH.")
            print("Fallback: Please install 'Windows Terminal' from the Microsoft Store.")
            return

        # 4. Execute all in one go
        # Join with semicolons for WT arguments
        full_command = f'wt {" ; ".join(commands)}'
        print("Launching Windows Terminal...")
        subprocess.run(full_command, shell=True)

    elif current_os == "Linux" or current_os == "Darwin": # Darwin is macOS
        procs = []
        
        # 1. Start Server
        print("Launching Server...")
        server_cmd = [get_python_cmd(), "server.py"]
        # stdout=None allows it to print to the main terminal
        server_proc = subprocess.Popen(server_cmd)
        procs.append(("Server", server_proc))
        
        # Give server time to initialize
        time.sleep(5)

        # 2. Start Clients
        for cid in range(config.NUM_CLIENTS):
            print(f"Launching Client {cid}...")
            client_cmd = [get_python_cmd(), "client.py", str(cid)]
            client_proc = subprocess.Popen(client_cmd)
            procs.append((f"Client {cid}", client_proc))
            
            # Stagger slightly
            time.sleep(1)

        # 3. Monitor Processes
        try:
            while procs:
                for name, p in procs[:]:
                    if p.poll() is not None: # Process finished
                        print(f"Process {name} has ended.")
                        procs.remove((name, p))
                
                if not procs:
                    break
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopping all processes...")
            for _, p in procs:
                p.terminate()

    else:
        print(f"Unsupported OS: {current_os}")

if __name__ == "__main__":
    run_experiment()