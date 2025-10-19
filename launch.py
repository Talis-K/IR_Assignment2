from threading import Thread
from main_Ihtisham import Environment
from guitest import launch_gui

def main():
    env = Environment()
    def _run():
        env.run_mission()
        env.env.hold() 
    Thread(target=_run, daemon=True).start()
    launch_gui()
if __name__ == "__main__":
    main()
