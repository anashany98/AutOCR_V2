import sys
import logging
# Disable logging to keep output clean
logging.disable(logging.CRITICAL)

from web_app.app import app

print("Searching for batch_action...")
found = False
for rule in app.url_map.iter_rules():
    if "batch_action" in rule.endpoint:
        print(f"FOUND: {rule} -> {rule.endpoint}")
        found = True

if not found:
    print("NOT FOUND: batch_action")

# check verifies too
for rule in app.url_map.iter_rules():
    if "verify" in rule.endpoint:
         print(f"VERIFY RULE: {rule} -> {rule.endpoint}")
