import subprocess

def run_ollama_prompt(prompt, model="gemma:2b"):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )
        if result.returncode != 0:
            return f"❌ Ollama Error:\n{result.stderr.decode('utf-8')}"
        return result.stdout.decode("utf-8").strip()
    except subprocess.TimeoutExpired:
        return "⚠️ Ollama took too long to respond. Please try again."
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"
