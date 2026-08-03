#!/usr/bin/env python3
"""Run a command inside a PTY and tee output to a log file.
tqdm sees a real terminal → progress bar with \\r works correctly.
The log file records raw bytes (including \\r), so `tail -f log` in a
terminal shows the progress bar updating in place.
"""
import os
import pty
import sys
import select
import struct
import fcntl
import termios


# Terminal size to advertise to the child (columns x rows)
_COLS = 200
_ROWS = 50


def main(logfile, cmd):
    master_fd, slave_fd = pty.openpty()

    # Set terminal window size so shutil.get_terminal_size() returns a real value
    winsize = struct.pack('HHHH', _ROWS, _COLS, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

    pid = os.fork()
    if pid == 0:
        # Child: connect stdin/stdout/stderr to the slave PTY
        os.close(master_fd)
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        if slave_fd > 2:
            os.close(slave_fd)
        os.execv(cmd[0], cmd)
        sys.exit(1)

    # Parent: read from master PTY, write to log file
    os.close(slave_fd)
    with open(logfile, 'wb') as f:
        while True:
            try:
                r, _, _ = select.select([master_fd], [], [], 1.0)
            except (OSError, ValueError):
                break
            if r:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                f.write(data)
                f.flush()
            else:
                # Check if child exited
                result = os.waitpid(pid, os.WNOHANG)
                if result[0] != 0:
                    break
    try:
        os.close(master_fd)
    except OSError:
        pass
    _, status = os.waitpid(pid, 0)
    sys.exit(os.WEXITSTATUS(status))

if __name__ == '__main__':
    # Usage: pty_run.py <logfile> <cmd> [args...]
    main(sys.argv[1], sys.argv[2:])
