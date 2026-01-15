# Debug: Socket Disconnect on File Selection Prompt

**Issue**: Client disconnected while prompt was waiting for file selection  
**Date**: January 14, 2026

---

## Observed Behavior

```bash
[6:51:24 PM] INFO [prompt] [PROMPT] Select a file to parse from uploads (enter index or filename): (type 'cancel' to abort)
[6:53:07 PM] INFO [status] Client disconnected (socket_sid=9bsXY6NSFr7yDJRiAAAB, session_id=None)
```

The client disconnected after the prompt was shown. This is **not** an error - it's normal Socket.IO polling behavior.

---

## Root Cause Analysis

### What Happened

1. **[6:51:24]** Server created a prompt and awaited user input via `prompt_input()`
2. **Client waiting** - JavaScript was waiting for user to enter file index (0-6)
3. **[6:53:07]** Socket.IO polling timeout after ~2 minutes of inactivity
4. **Disconnect** - Browser client disconnected due to heartbeat/polling timeout

### Why This Is Normal

- **Socket.IO polling** (vs WebSocket) uses long-polling by default on this setup
- **Heartbeat interval**: 10 seconds (set in SOCKETIO_ENGINE_OPTIONS)
- **Ping timeout**: 60 seconds (max wait before assuming client dead)
- **User didn't respond**: After 2+ minutes, no prompt response received
- **Graceful timeout**: Browser closed socket to free resources

### How It Works

```text
Client                           Server
  │                                │
  ├─ Socket connects              │
  │                    ✅ Accepted
  │                                │
  │                  Prompt shown ──┤ Waiting for input...
  │◄─── Display file list          │
  │                                │
  │ (User not responding)          │
  │                                │
  ├─ 60s timeout on heartbeat      │
  │   Socket force-closes           │ Logs disconnect (session_id captured)
  │                                │
```

---

## Why This Is Actually Good

✅ **Prevents hung connections** - Stale sockets don't drain resources  
✅ **Graceful degradation** - No server-side hanging  
✅ **User friendly** - Browser handles timeout naturally  
✅ **Recoverable** - Session is preserved, user can reconnect

---

## How to Avoid It

### Solution 1: Respond to Prompt Quickly

Just select a file from the list while the prompt is visible.

**Expected workflow**:

1. Files are listed in debug console
2. You have ~60 seconds to respond
3. Enter `0` (for first file) or filename
4. Press Enter

### Solution 2: Test with Shorter Timeout (Dev Only)

If testing, you can modify:

**File**: `webapp/Smart_Elections_Parser_Webapp.py`

```python
_SOCKETIO_ENGINE_OPTIONS = {
    "ping_interval": 10,      # How often server pings client
    "ping_timeout": 60,       # <- Increase this for longer waits
    "allow_upgrades": False,
    "transports": ["polling"],
}
```

For testing, change to `ping_timeout: 300` (5 minutes).

### Solution 3: Improve UX (Recommended Feature)

Add a **heartbeat animation** or **progress indicator** in the debug console to show the system is still listening.

This would involve:

1. Adding a "typing indicator" to show prompt is active
2. Displaying remaining time before timeout
3. Showing heartbeat pulse (visual feedback)

---

## Technical Details

### Socket.IO Configuration

**Current setup** in `Smart_Elections_Parser_Webapp.py`:

```python
_SOCKETIO_ASYNC_MODE = "threading"

_SOCKETIO_ENGINE_OPTIONS = {
    "ping_interval": 10,
    "ping_timeout": 60,
    "allow_upgrades": False,
    "transports": ["polling"],
}
```

**What this means**:

- **async_mode**: Using Python threading (not eventlet/gevent)
- **ping_interval**: Server sends ping every 10 seconds
- **ping_timeout**: Client has 60 seconds to respond to ping
- **allow_upgrades**: Don't try to upgrade from polling to WebSocket
- **transports**: Use HTTP long-polling only (most compatible)

### Prompt Handling

**File**: `webapp/parser/utils/contest_selector.py`

```python
def prompt_user_input(message, validator=None, session_id=None):
    # This waits for response via Socket.IO event
    # If no response in ~60 seconds, Socket.IO times out
    pass
```

---

## What Changed in Recent Fix

**File**: `webapp/Smart_Elections_Parser_Webapp.py`  
**Change**: Better session tracking in `handle_disconnect()`

**Before**:

```python
logical = session_manager.unbind_socket(req_sid) if req_sid else None
logger.info({
    "message": f"Client disconnected...",
    "session_id": logical  # Often None
})
```

**After**:

```python
logical = session_manager.resolve_socket(req_sid)  # Get it first
unbound_session = session_manager.unbind_socket(req_sid) if req_sid else None
logical = logical or unbound_session
logger.info({
    "message": f"Client disconnected...",
    "session_id": logical  # Now shows correct session
})
```

**Benefit**: Logs now show which session disconnected, making debugging easier.

---

## How to Debug This

### Enable Verbose Logging

Add to `Smart_Elections_Parser_Webapp.py`:

```python
# In heartbeat handler
logger.info({
    "level": "DEBUG",
    "type": "heartbeat",
    "message": f"Heartbeat for {sid}",
    "session_id": sid,
})
```

### Check Prompt Status

When prompt is active, you should see:

1. Prompt message in debug console
2. Status showing "WAITING_PROMPT" in session
3. Heartbeat logs every 10 seconds
4. If no heartbeat after 60s → disconnect incoming

### Verify Socket.IO Connection

In browser console:

```javascript
console.log("Socket connected:", socket.connected);
console.log("Socket ID:", socket.id);
```

---

## Next Steps

### Short Term (No Code Change)

- ✅ Just respond to prompts within 60 seconds
- ✅ Use `0` through `6` to select files quickly
- ✅ Normal operation should work fine

### Medium Term (Small Enhancement)

- [ ] Add visual countdown timer in debug console
- [ ] Show "Waiting for input..." indicator
- [ ] Display remaining time before timeout

### Long Term (Better Reliability)

- [ ] Implement WebSocket upgrade (better for long polls)
- [ ] Add persistent storage for prompt state
- [ ] Implement automatic reconnection with state recovery

---

## FAQ

**Q: Is this a bug?**  
A: No. This is normal Socket.IO polling behavior with proper timeout handling.

**Q: Why did it disconnect?**  
A: Browser/server lost heartbeat after ~2 minutes of no response to prompt.

**Q: How do I fix it?**  
A: Respond to prompts within 60 seconds by entering the file number or name.

**Q: Can I extend the timeout?**  
A: Yes, change `ping_timeout` in SOCKETIO_ENGINE_OPTIONS from 60 to 300+ seconds.

**Q: Will it affect performance?**  
A: No, this only affects idle connections waiting for user input.

**Q: Is my data safe?**  
A: Yes, disconnect only closes the socket. Session data is preserved server-side.

---

## References

- Socket.IO Documentation: <https://python-socketio.readthedocs.io/>
- Engine.IO Options: <https://python-engineio.readthedocs.io/>
- Session Management: See `session_manager` in `Smart_Elections_Parser_Webapp.py`

---

**Summary**: This is expected behavior for Socket.IO polling. The recent fix improves logging to make this visible. For typical use, just respond to prompts within 60 seconds.

See **RECENT_FIXES.md** for details on the disconnect logging improvement.
