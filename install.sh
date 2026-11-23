#!/bin/bash
# Install (or remove) the meteor scorer on the Pi as a user timer, run as
# ubuntu, no sudo. From the Mac:
#
#   scp -r meteors/allsky_meteors meteors/data meteors/install.sh ubuntu@indi-allsky.local:allsky-meteors-src/
#   ssh ubuntu@indi-allsky.local allsky-meteors-src/install.sh install
#
#   ~/allsky-meteors-src/install.sh install     timer every 10 min (night frames only)
#   ~/allsky-meteors-src/install.sh uninstall   (results in ~/allsky-meteors are kept)
#   ~/allsky-meteors-src/install.sh status
#
# Uses indi-allsky's own Python (numpy, opencv, scipy, sgp4, ephem are in its
# virtualenv); installs nothing. Results: ~/allsky-meteors/ (RUNBOOK §6,
# "Meteor scorer").

set -euo pipefail

SRC=$(cd "$(dirname "$0")" && pwd)
UNITS=~/.config/systemd/user
PY=~/indi-allsky/virtualenv/indi-allsky/bin/python

case "${1:-}" in
    install)
        mkdir -p "$UNITS"
        cat > "$UNITS/allsky-meteors.service" <<EOF
[Unit]
Description=Meteor scorer: find and score meteors in new night frames

[Service]
Type=oneshot
WorkingDirectory=$SRC
ExecStart=$PY -m allsky_meteors.scorer run
Nice=15
IOSchedulingClass=idle
# one run normally takes seconds; a backlog after downtime is capped by the
# frames' 3-day expiry
TimeoutStartSec=2h
EOF
        cat > "$UNITS/allsky-meteors.timer" <<'EOF'
[Unit]
Description=Run the meteor scorer every 10 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=10min

[Install]
WantedBy=timers.target
EOF
        systemctl --user daemon-reload
        systemctl --user enable --now allsky-meteors.timer
        systemctl --user start allsky-meteors.service
        journalctl --user -u allsky-meteors -n 3 --no-pager
        ;;
    uninstall)
        systemctl --user disable --now allsky-meteors.timer 2>/dev/null || true
        rm -f "$UNITS/allsky-meteors.service" "$UNITS/allsky-meteors.timer"
        systemctl --user daemon-reload
        ;;
    status)
        systemctl --user --no-pager status allsky-meteors.timer allsky-meteors.service | head -20 || true
        cd "$SRC" && $PY -m allsky_meteors.scorer report | head -20
        ;;
    *) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
