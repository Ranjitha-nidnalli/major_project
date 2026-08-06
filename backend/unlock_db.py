import os
import psutil

def clear_db_locks():
    current_pid = os.getpid()
    locked_deleted = 0
    
    print("Scanning for hidden/zombie python processes holding the Qdrant Lock...")
    
    # Iterate through all running processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if it's a python process
            if proc.info['name'] and proc.info['name'].lower() in ['python.exe', 'python', 'python3', 'pythonw.exe']:
                # Do not kill the script currently doing the killing!
                if proc.info['pid'] == current_pid:
                    continue
                
                cmdline = proc.info.get('cmdline', [])
                if cmdline:
                    cmd_str = " ".join(cmdline).lower()
                    
                    # Target only the our project's backend scripts
                    if "main.py" in cmd_str or "evaluate_accuracy.py" in cmd_str or "vector_db.py" in cmd_str:
                        print(f"Terminating locking process zombie PID: {proc.info['pid']} ({cmd_str})")
                        proc.kill()
                        locked_deleted += 1
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if locked_deleted > 0:
        print(f"\n[OK] Terminated {locked_deleted} stuck processes.")
        print("[OK] All locks cleared. You can now start main.py or your evaluation script.")
    else:
        print("\n[OK] No locking processes found.")
        print("[OK] All locks cleared. You can now start main.py or your evaluation script.")

if __name__ == "__main__":
    clear_db_locks()
