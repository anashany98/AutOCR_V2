import imaplib
import email
import os
import time
import logging
import threading
import re
from email.header import decode_header
from typing import List, Optional

class EmailImporter:
    def __init__(self, config: dict, input_folder: str):
        self.config = config
        self.input_folder = input_folder
        self.logger = logging.getLogger("EmailImporter")
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return
        
        if not self.config.get("enabled", False):
            self.logger.info("Email Importer is disabled in configuration.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.logger.info("Email Importer started.")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.logger.info("Email Importer stopped.")

    def _run_loop(self):
        check_interval = self.config.get("check_interval", 60)
        while self.running:
            try:
                self._check_email()
            except Exception as e:
                self.logger.error(f"Error checking email: {e}")
            
            # Sleep in small chunks to allow faster stopping
            for _ in range(check_interval):
                if not self.running:
                    break
                time.sleep(1)

    def _check_email(self):
        host = self.config.get("host")
        port = self.config.get("port", 993)
        user = self.config.get("user")
        password = self.config.get("password")
        allowed_exts = {ext.lower() for ext in self.config.get("allowed_extensions", [])}

        if not host or not user or not password:
            self.logger.warning("Email configuration missing (host, user, or password).")
            return

        try:
            self.logger.debug(f"Connecting to IMAP {host}:{port}...")
            mail = imaplib.IMAP4_SSL(host, port)
            mail.login(user, password)
            
            folders = self.config.get("folders", ["INBOX"])
            for folder in folders:
                status, _ = mail.select(folder)
                if status != "OK":
                    self.logger.warning(f"Could not select folder {folder}")
                    continue

                # Search for all unseen emails
                status, messages = mail.search(None, 'UNSEEN')
                
                if status != "OK":
                    continue

                email_ids = messages[0].split()
                if not email_ids:
                    continue
                    
                self.logger.info(f"Found {len(email_ids)} new emails in {folder}")

                for e_id in email_ids:
                    res, msg_data = mail.fetch(e_id, '(RFC822)')
                    if res != 'OK':
                        continue
                        
                    raw_email = msg_data[0][1] # type: ignore
                    msg = email.message_from_bytes(raw_email)
                    subject = self._decode_header(msg.get("Subject", "(No Subject)"))
                    sender = self._decode_header(msg.get("From", "(Unknown)"))
                    
                    self.logger.info(f"Processing email from {sender}: {subject}")

                    has_attachments = False
                    for part in msg.walk():
                        if part.get_content_maintype() == 'multipart':
                            continue
                        if part.get('Content-Disposition') is None:
                            continue
                            
                        filename = part.get_filename()
                        if filename:
                            filename = self._decode_header(filename)
                            # Sanitize filename
                            filename = re.sub(r'[<>:"/\\|?*]', '', filename)
                            
                            ext = os.path.splitext(filename)[1].lower()
                            
                            if ext in allowed_exts:
                                timestamp = int(time.time())
                                safe_subject = re.sub(r'[<>:"/\\|?*]', '', subject)[:20]
                                new_filename = f"Email_{timestamp}_{safe_subject}_{filename}"
                                filepath = os.path.join(self.input_folder, new_filename)
                                
                                payload = part.get_payload(decode=True)
                                if payload:
                                    with open(filepath, "wb") as f:
                                        f.write(payload)
                                    self.logger.info(f"Saved attachment: {new_filename}")
                                    has_attachments = True
                    
                    if not has_attachments:
                        self.logger.info("No valid attachments found in email.")
            
            mail.close()
            mail.logout()

        except Exception as e:
            self.logger.error(f"IMAP connection failed: {e}")

    def _decode_header(self, text):
        if not text:
            return ""
        try:
            decoded_list = decode_header(text)
            decoded_text = ""
            for bytes_data, encoding in decoded_list:
                if isinstance(bytes_data, bytes):
                    if encoding:
                        try:
                            decoded_text += bytes_data.decode(encoding)
                        except LookupError:
                            decoded_text += bytes_data.decode('utf-8', errors='ignore')
                        except Exception:
                             decoded_text += bytes_data.decode('utf-8', errors='ignore')
                    else:
                        decoded_text += bytes_data.decode('utf-8', errors='ignore')
                else:
                    decoded_text += str(bytes_data)
            return decoded_text
        except Exception:
            return str(text)
