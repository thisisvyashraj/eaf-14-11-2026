"""Starts the bot with settings from the .env file next to it (works on Windows, Linux, Mac). Restarts it if it ever dies."""
import os, sys, time, runpy
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE)


def load_env(path):
    n = 0
    if not os.path.exists(path):
        return n
    with open(path, encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v != "":
                os.environ.setdefault(k, v)                 # a real environment variable wins over the file
                n += 1
    return n


if __name__ == "__main__":
    n = load_env(os.path.join(HERE, ".env"))
    if n == 0:
        print("No settings found. Copy .env.example to .env and fill in your values, then start again.")
        sys.exit(1)
    print(f"Loaded {n} settings from .env")
    while True:
        try:
            runpy.run_path(os.path.join(HERE, "bot.py"), run_name="__main__")
        except KeyboardInterrupt:
            print("Stopped."); break
        except SystemExit:
            break
        except Exception as e:
            print("Bot crashed:", repr(e), "- restarting in 20 s"); time.sleep(20)
