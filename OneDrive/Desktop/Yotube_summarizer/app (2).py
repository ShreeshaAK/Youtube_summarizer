from flask import Flask, request, jsonify, render_template
import anthropic
import os
import re
import json
import glob
import tempfile
import yt_dlp
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

def parse_vtt(content):
    """Convert VTT subtitle file to clean plain text."""
    lines = content.split('\n')
    text_lines = []
    for line in lines:
        line = line.strip()
        # Skip WEBVTT header, timestamps, empty lines, numeric cue IDs
        if not line:
            continue
        if line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
            continue
        if '-->' in line:
            continue
        if re.match(r'^\d+$', line):
            continue
        # Remove HTML/VTT tags like <00:00:01.000>, <c>, </c>, <b>, etc.
        line = re.sub(r'<[^>]+>', '', line)
        line = line.strip()
        if not line:
            continue
        # Skip duplicate consecutive lines (common in VTT rolling captions)
        if text_lines and text_lines[-1] == line:
            continue
        text_lines.append(line)
    return ' '.join(text_lines)

def get_transcript(video_id):
    """Fetch transcript using yt-dlp — works with auto-generated and manual subtitles."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"yt_sub_{video_id}")

    # Clean up any leftover files from previous attempts
    for f in glob.glob(f"{output_path}*"):
        try:
            os.remove(f)
        except:
            pass

    ydl_opts = {
        'writesubtitles': True,       # manual subtitles
        'writeautomaticsub': True,    # auto-generated subtitles
        'subtitleslangs': ['en', 'en-US', 'en-GB', 'en-AU'],
        'subtitlesformat': 'vtt',
        'skip_download': True,        # don't download the video
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        # Find any downloaded .vtt subtitle file
        vtt_files = glob.glob(f"{output_path}*.vtt")

        if not vtt_files:
            # Try fetching with any available language and translate
            available_subs = info.get('subtitles', {})
            available_auto = info.get('automatic_captions', {})
            all_langs = list(available_subs.keys()) + list(available_auto.keys())

            if not all_langs:
                return None, "This video has no subtitles or captions available."

            # Retry with first available language
            ydl_opts['subtitleslangs'] = [all_langs[0]]
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
            vtt_files = glob.glob(f"{output_path}*.vtt")

        if not vtt_files:
            return None, "Could not download subtitles. The video may have captions disabled."

        # Read and parse the subtitle file
        with open(vtt_files[0], 'r', encoding='utf-8') as f:
            content = f.read()

        # Clean up temp files
        for f in vtt_files:
            try:
                os.remove(f)
            except:
                pass

        transcript = parse_vtt(content)

        if not transcript.strip():
            return None, "Transcript was empty after parsing."

        return transcript, None

    except yt_dlp.utils.DownloadError as e:
        err = str(e)
        if "private" in err.lower():
            return None, "This video is private."
        if "unavailable" in err.lower():
            return None, "This video is unavailable."
        return None, f"Could not access video: {err}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


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
        return jsonify({"error": "Invalid YouTube URL. Please check and try again."}), 400

    transcript, error = get_transcript(video_id)
    if error:
        return jsonify({"error": error}), 400

    trimmed_text = transcript[:12000]

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
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "AI returned unexpected format. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)