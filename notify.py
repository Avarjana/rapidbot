"""Telegram notification service and command listener for rapid_bot.
Uses requests for outbound alerts and background command polling.
"""

import logging
import queue
import threading
import time
from typing import Callable, Dict, Optional
import requests

logger = logging.getLogger("rapid_bot.notify")


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, bot_name: str = "rapid_bot"):
        self.token = token
        self.chat_id = chat_id
        self.bot_name = bot_name
        self.enabled = bool(token and chat_id)
        self.base_url = f"https://api.telegram.org/bot{token}" if self.enabled else ""
        
        self.msg_queue: queue.Queue = queue.Queue()
        self.running = False
        self._sender_thread: Optional[threading.Thread] = None
        self._poller_thread: Optional[threading.Thread] = None
        self.command_handlers: Dict[str, Callable[[], str]] = {}
        self.last_update_id = 0

    def start(self):
        """Start sender and command poller threads."""
        if not self.enabled:
            logger.info("Telegram notifications disabled (missing token or chat_id).")
            return

        self.running = True
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True, name="TelegramSender")
        self._sender_thread.start()

        self._poller_thread = threading.Thread(target=self._poller_loop, daemon=True, name="TelegramPoller")
        self._poller_thread.start()
        logger.info("Telegram notifier and command poller started.")

    def stop(self):
        """Stop background threads."""
        self.running = False

    def register_command(self, cmd: str, handler: Callable[[], str]):
        """Register command handler, e.g. cmd='/status'."""
        self.command_handlers[cmd.lower().strip()] = handler

    def send_alert(self, message: str, level: str = "INFO"):
        """Queue outbound alert. Only WARNING, ERROR, CRITICAL or explicit alerts sent."""
        if not self.enabled:
            return
        formatted = f"[{self.bot_name}] [{level}]\n{message}"
        self.msg_queue.put(formatted)

    def _sender_loop(self):
        """Consume messages from queue and dispatch to Telegram."""
        while self.running:
            try:
                msg = self.msg_queue.get(timeout=1.0)
                self._send_http(msg)
                self.msg_queue.task_done()
                time.sleep(0.1)  # Avoid Telegram rate limits
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in Telegram sender loop: {e}")

    def _send_http(self, text: str):
        """Direct POST to sendMessage API."""
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code != 200:
                logger.warning(f"Telegram send failed ({r.status_code}): {r.text}")
        except Exception as e:
            logger.warning(f"Exception sending Telegram message: {e}")

    def _poller_loop(self):
        """Poll getUpdates for inbound bot commands (/status, /pause, /resume, /flatten)."""
        while self.running:
            try:
                url = f"{self.base_url}/getUpdates"
                params = {"offset": self.last_update_id + 1, "timeout": 20}
                r = requests.get(url, params=params, timeout=25)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("ok"):
                        for update in data.get("result", []):
                            self.last_update_id = update.get("update_id", self.last_update_id)
                            msg = update.get("message", {})
                            chat = msg.get("chat", {})
                            from_chat_id = str(chat.get("id", ""))

                            # Security check: only respond to authorized chat_id
                            if from_chat_id != str(self.chat_id):
                                continue

                            text = msg.get("text", "").strip()
                            if text.startswith("/"):
                                cmd = text.split()[0].lower()
                                if cmd in self.command_handlers:
                                    response_text = self.command_handlers[cmd]()
                                    self._send_http(response_text)
                                else:
                                    self._send_http(
                                        f"Unknown command: `{cmd}`\nAvailable: `/status`, `/pause`, `/resume`, `/flatten`"
                                    )
            except Exception as e:
                logger.debug(f"Poller exception (normal during network drop): {e}")
                time.sleep(5)
            time.sleep(1)
