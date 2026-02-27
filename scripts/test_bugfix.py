import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.llm_client import LLMClient
from web_app.services import load_configuration

def test_fix():
    print("Testing Bug Fixes...")
    
    config = load_configuration()
    llm = LLMClient(config.get("llm", {}))
    
    # 1. Test stripping logic (mocking response if possible or just checking code)
    # Since we can't easily mock the response object from OpenAI client without heavy mocking, 
    # let's test a manual string stripping helper if we had one, but we put it in chat().
    # Let's try a real call if the server is up.
    
    print("\n--- Step 2: Testing Surgical Reasoning Leak Simulation ---")
    leak_example = "Vale, el usuario está pidiendo cosas. Analizando... Así transformo esto. ¡Claro! Hola, soy tu asistente."
    
    # We can't call the internal logic of chat() directly without a real LLM call,
    # but we can verify if a real call still leaks.
    # For now, let's just run Step 1 again with a more 'leak-prone' prompt.
    res = llm.chat("que puedes hacer por mi", system_prompt="Eres el administrador del sistema.")
    cleaned = res.get('analysis', "")
    print(f"Response: {cleaned[:100]}...")
    
    if any(m in cleaned.lower()[:50] for m in ["vale", "analizando", "el usuario", "the user"]):
        print("❌ FAILURE: Reasoning markers found at the start of the response.")
    else:
        print("✅ SUCCESS: Reasoning was successfully stripped (or didn't occur).")

if __name__ == "__main__":
    test_fix()
