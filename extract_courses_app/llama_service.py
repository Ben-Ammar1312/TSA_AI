import subprocess

def extract_courses(text: str) -> list:
    prompt = f"""
    Extract all course names from the text below.
    Respond ONLY with a JSON list (e.g. ["Math","Physics"]).

    Text:
    {text}
    """

    try:
        # Don't use --json for now, it hides the real output
        result = subprocess.run(
            ["ollama", "run", "llama3.1:8b-instruct-q4_K_M"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )


        output = result.stdout.decode("utf-8").strip()
        print("🧠 Raw Llama output:", output)

        # Try to parse JSON list manually
        import json, re
        match = re.search(r'\[.*\]', output, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            return []

    except Exception as e:
        print("Error:", e)
        return []
