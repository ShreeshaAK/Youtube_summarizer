from flask import Flask, request, jsonify, render_template
from youtube_transcript_api import YouTubeTranscriptApi
import anthropic
import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an expert YouTube video summarizer. Analyze English transcripts and produce rich, structured summaries.

Given a transcript, respond ONLY with valid JSON in this exact format:
{
  "tldr": "2-3 sentence summary of the entire video",
  "key_points": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "detailed_summary": "3-5 paragraph detailed summary of the full content",
  "takeaways": ["takeaway 1", "takeaway 2", "takeaway 3"],
  "topics": ["topic1", "topic2", "topic3"]
}

Rules:
- Write in clear, simple English
- Never start with filler like "In this video..."
- Be factual and informative
- topics should be 1-2 word tags only
- Return ONLY the JSON, no extra text or markdown
"""

def extract_video_id(url):
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:shorts/)([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url.strip()):
        return url.strip()
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "Please enter a YouTube URL."}), 400

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Could not extract video ID. Please check the URL."}), 400

    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        full_text = " ".join([t.text for t in fetched])
    except Exception as e:
        return jsonify({"error": f"Could not fetch transcript: {str(e)}"}), 400

    trimmed_text = full_text[:12000]

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Summarize this YouTube video transcript:\n\n{trimmed_text}"
            }]
        )
        raw = message.content[0].text.strip()
        result = json.loads(raw)
        return jsonify(result)
    except json.JSONDecodeError:
        return jsonify({"error": "AI returned unexpected format. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
