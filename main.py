import streamlit as st
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
import requests
import os
import tempfile
import time
from moviepy.editor import VideoFileClip
import json

# Initialize Vertex AI with your project and location
PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Function to load the Gemini Pro Vision model
@st.cache_resource
def load_model():
    return GenerativeModel("gemini-1.5-flash-001")

# Function to download video, store temporarily, and extract metadata using moviepy
def download_and_extract_metadata(cdn_url):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            # Fetch the video file
            video_response = requests.get(cdn_url, stream=True)
            if video_response.status_code == 200:
                for chunk in video_response.iter_content(chunk_size=8192):
                    tmp_video.write(chunk)
            video_path = tmp_video.name

        # Extract metadata using moviepy
        with VideoFileClip(video_path) as video:
            video_duration_sec = video.duration
            video_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB

        video_name = os.path.basename(cdn_url)

        # Return metadata
        return video_name, video_size, video_duration_sec, video_path
    except Exception as e:
        st.error(f"Error fetching or processing video: {e}")
        return None, None, None, None

# Function to delete the temporary file after use
def delete_temp_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        st.error(f"Error deleting temporary file: {e}")

# Function to convert video content to base64
def convert_video_to_base64(video_path):
    start_time = time.time()  # Start timer for base64 encoding
    with open(video_path, 'rb') as video_file:
        video_data = video_file.read()
        video_base64 = base64.b64encode(video_data).decode('utf-8')
    base64_time = time.time() - start_time  # Calculate the time taken
    st.write(f"Time for Base64 conversion: {base64_time:.2f} seconds")
    return Part.from_data(mime_type="video/mp4", data=base64.b64decode(video_base64)), base64_time

# Function to generate and display content
# Function to generate and display content
def generate_and_display_content(model, prompt, video_part, key):
    if video_part and st.button("Generate", key=key):
        with st.spinner("Generating..."):
            try:
                start_time = time.time()  # Start timing for API call (inference time)
                responses = model.generate_content(
                    [video_part, prompt],
                    generation_config={
                        "max_output_tokens": 512,
                        "temperature": 1,
                        "top_p": 0.95,
                    },
                    safety_settings=[
                        SafetySetting(
                            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                        ),
                        SafetySetting(
                            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                        ),
                        SafetySetting(
                            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                        ),
                        SafetySetting(
                            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                        ),
                    ],
                    stream=False,
                )

                # Debugging: Check the type of the response
                st.write(f"Response type: {type(responses)}")

                # If the response is not iterable, handle it directly
                if hasattr(responses, "text"):
                    final_response = responses.text
                else:
                    st.error(f"Unexpected response format: {responses}")
                    final_response = ""

                # For tags, parse the JSON and display it properly
                if key == "video_tags":
                    try:
                        tags_data = json.loads(final_response)
                        st.json(tags_data)
                    except json.JSONDecodeError:
                        st.write("Error parsing tags JSON. Raw output:")
                        st.write(final_response)
                else:
                    st.write(final_response)

                inference_time_ms = (time.time() - start_time) * 1000  # Inference time in milliseconds
                return inference_time_ms, final_response

            except Exception as e:
                st.error(f"Error during API call: {e}")
                return 0, ""
    
    return 0, ""


# Function to calculate cost based on token usage and duration
def calculate_cost(video_duration_sec, input_chars, output_chars):
    # Pricing details from the reference
    video_input_cost_per_sec = 0.00002  # Per second of video input
    text_input_cost_per_1k_chars = 0.00001875  # Per 1,000 characters of text input
    text_output_cost_per_1k_chars = 0.000075  # Per 1,000 characters of text output

    # Calculate costs based on actual input values
    video_cost = video_duration_sec * video_input_cost_per_sec
    input_text_cost = (input_chars / 1000) * text_input_cost_per_1k_chars
    output_text_cost = (output_chars / 1000) * text_output_cost_per_1k_chars

    total_cost = video_cost + input_text_cost + output_text_cost
    return total_cost

# Main function
def main():
    st.header("Paigeon AI Video Analyzer")
    multimodal_model = load_model()

    # User input for Bunny CDN URL
    user_input_url = st.text_input("Enter the Bunny CDN URL for your video:", "")

    # Fetch video, extract metadata, and convert to base64 if URL is provided
    if user_input_url:
        video_name, video_size, video_duration_sec, video_path = download_and_extract_metadata(user_input_url)
        if video_path:
            st.write(f"Video Name: {video_name}")
            st.write(f"Video Size: {video_size:.2f} MB")
            st.write(f"Video Duration: {video_duration_sec:.2f} seconds")

            # Measure time for base64 conversion
            video_part, base64_time = convert_video_to_base64(video_path)

            # Delete the temporary video file after processing
            delete_temp_file(video_path)

            # Tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["Video Description", "Video Tags", "Video Highlights", "Video Shopping Objects"])

            # Video Description
            with tab1:
                st.subheader("Generate Video Description")
                prompt = "Describe what is happening in this video."
                inference_time_ms, final_response = generate_and_display_content(multimodal_model, prompt, video_part, key="video_description")
                st.write(f"Time for inference: {inference_time_ms:.2f} ms")

                # Calculate cost based on actual content size
                input_chars = len(prompt)
                output_chars = len(final_response)
                cost = calculate_cost(video_duration_sec, input_chars, output_chars)
                st.write(f"Estimated cost for video description generation: ${cost:.6f}")

            # Video Tags
            with tab2:
                st.subheader("Generate Video Tags")
                prompt = """Generate potential tags with probability scores in JSON format. Limit to top 20 tags. Return only the JSON, no additional text."""
                inference_time_ms, final_response = generate_and_display_content(multimodal_model, prompt, video_part, key="video_tags")
                st.write(f"Time for inference: {inference_time_ms:.2f} ms")

                # Calculate cost based on actual content size
                input_chars = len(prompt)
                output_chars = len(final_response)
                cost = calculate_cost(video_duration_sec, input_chars, output_chars)
                st.write(f"Estimated cost for generating tags: ${cost:.6f}")

            # Video Highlights
            with tab3:
                st.subheader("Generate Video Highlights")
                prompt = "Summarize the key highlights of this video."
                inference_time_ms, final_response = generate_and_display_content(multimodal_model, prompt, video_part, key="video_highlights")
                st.write(f"Time for inference: {inference_time_ms:.2f} ms")

                # Calculate cost based on actual content size
                input_chars = len(prompt)
                output_chars = len(final_response)
                cost = calculate_cost(video_duration_sec, input_chars, output_chars)
                st.write(f"Estimated cost for generating highlights: ${cost:.6f}")

            # Video Shopping Objects
            with tab4:
                st.subheader("Generate Video Shopping Objects")
                prompt = "Identify the objects present in this video which can be used for online shopping."
                inference_time_ms, final_response = generate_and_display_content(multimodal_model, prompt, video_part, key="video_shopping_objects")
                st.write(f"Time for inference: {inference_time_ms:.2f} ms")

                # Calculate cost based on actual content size
                input_chars = len(prompt)
                output_chars = len(final_response)
                cost = calculate_cost(video_duration_sec, input_chars, output_chars)
                st.write(f"Estimated cost for generating shopping objects: ${cost:.6f}")

if __name__ == "__main__":
    main()