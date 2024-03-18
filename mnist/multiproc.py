import multiprocessing
import subprocess

def run_script(args):
    subprocess.run(["python", "your_script.py"] + args)

if __name__ == "__main__":
    # List of argument sets for your script
    args_list = [
        ["arg1", "arg2"],
        ["arg3", "arg4"],
        # Add more argument sets as needed
    ]

    # Create a pool of processes to run the script in parallel
    with multiprocessing.Pool() as pool:
        pool.map(run_script, args_list)
