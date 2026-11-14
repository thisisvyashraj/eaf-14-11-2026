# Optional: running the bot on your own PC / a VPS instead of Render (v4.0 already runs full strength on Render's 512 MB)

IMPORTANT: only ONE copy may run with the same SESSION_STRING. Before you start it anywhere else,
**suspend or delete the Render service** (otherwise Telegram logs the session out: "AuthKeyDuplicated").

## Option 1 - your own PC / laptop (simplest, truly free, no RAM problem)
1. Install Python 3.12 from python.org (tick "Add Python to PATH").
2. Unzip this folder. Copy `.env.example` to `.env` and fill in the values (copy them from Render -> Environment).
3. Windows: double-click `start_windows.bat`.   Linux/Mac: `./start_linux.sh`.
4. It must stay switched on: turn off sleep (Windows: Settings -> Power -> Sleep: Never). Control it with pause/resume/status in
   Moonknight's Saved Messages (or your OWNER_IDS), as before. An old spare laptop or Android phone with Termux also works.

## Option 2 - Oracle Cloud "Always Free" VM (free, always on, 12 GB RAM)  [checked Nov 2026]
* Oracle cut the free Arm allowance on 15 June 2026 from 4 CPU / 24 GB to 2 CPU / 12 GB - still far more than the bot needs.
* Needs a credit/debit card for sign-up. New VMs often fail with "Out of capacity": ask for a SMALL one (1 OCPU / 6 GB) or retry later/other availability domain.
* Oracle can reclaim Always Free VMs that stay very idle. Upgrading the account to "Pay As You Go" (you are still not charged inside the free limits)
  is the usual way people avoid that - check Oracle's current rules yourself.
Steps: create a VM.Standard.A1.Flex (Ubuntu 22.04), open SSH, upload this folder (WinSCP / scp), then on the VM:
    sudo apt update && sudo apt install -y python3-venv
    cd ~/trace-farmer-full && cp .env.example .env && nano .env     # fill the values
    ./start_linux.sh                                                   # test; stop with Ctrl+C
Run it forever (auto-start, auto-restart) - create /etc/systemd/system/tracebot.service:
    [Unit]
    Description=Trace bot
    After=network-online.target
    [Service]
    User=ubuntu
    WorkingDirectory=/home/ubuntu/trace-farmer-full
    ExecStart=/home/ubuntu/trace-farmer-full/venv/bin/python run.py
    Restart=always
    RestartSec=20
    [Install]
    WantedBy=multi-user.target
then:  sudo systemctl daemon-reload && sudo systemctl enable --now tracebot && journalctl -u tracebot -f

## Option 3 - an old Android phone as the server (free, no card, no PC)  [NOT tested by me - I have no phone to try it on]
Needs a phone with >= 3 GB RAM that stays plugged in on Wi-Fi (use a charge limit / avoid keeping a swollen battery).
1. Install **Termux from F-Droid** (not the old Play Store version), open it, then:
       pkg update && pkg install -y python python-numpy git
       termux-wake-lock
2. Copy this folder to the phone (USB, or Telegram "Saved Messages" -> download -> move into Termux's storage; run `termux-setup-storage` once).
3. In the folder:  `cp .env.example .env && nano .env`  (fill the values), then
       AIOHTTP_NO_EXTENSIONS=1 pip install telethon aiohttp
       python run.py
4. Android must not kill Termux: Settings -> Apps -> Termux -> Battery -> Unrestricted, and keep the "wake lock" notification.
If a package will not install, tell me the exact error line.

## Option 4 - Azure for Students (free $100 credit, NO credit card) - only if you are a full-time university student  [checked Nov 2026]
Sign up at azure.microsoft.com/free/students with your SCHOOL email. (A normal Azure free account DOES need a card.)
Create a Linux VM: Ubuntu 22.04, size **Standard_B1s** (1 GB RAM; the page lists 750 free hours/month of B1s for 12 months for new customers). Set a budget alert.
A 1 GB VM is tight for the full-size solver, so give it swap and use the default (trimmed) memory settings - delete the 4 lines EMB_MAXW/EMB_NT/EMB_TMAX/EMB_EXT from .env:
    sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
Then follow the same steps and systemd service as in Option 2. Delete the VM when the credit ends so you are never billed.

## Not options (checked Nov 2026)
* GitHub Actions / Codespaces: GitHub's terms forbid using them for serverless computing or anything unrelated to building/testing the repo's software, and
  misuse can get jobs killed or the repository disabled. A game bot is exactly that. Codespaces also stops when idle. Don't.
* Vercel / Netlify: only static sites and short serverless functions, not a program that stays running and holds a Telegram connection.

## Not options any more (checked Nov 2026)
Hugging Face Spaces (Docker Spaces now need a paid plan), Fly.io (no free tier for new accounts), Railway / Koyeb / Render free (512 MB).
I did NOT check Google Cloud's free e2-micro (1 GB RAM); full strength needs ~450 MB, so it would be tight.

Check first with:  python selfcheck.py   (tests the word files and the Cloudflare key)

## A warning about "free bot hosting" panels / Telegram-bot hosts you find online
Your SESSION_STRING is a full login to the Moonknight account. Anyone who runs your code on THEIR server can read it and take the account.
Only use a host you trust. I would not put it on referral-panel or unknown free bot hosts.
