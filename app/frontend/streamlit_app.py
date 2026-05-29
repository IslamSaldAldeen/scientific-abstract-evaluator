import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Scientific Abstract Evaluator",
    page_icon="🧠",
    layout="centered"
)

st.title("Scientific Abstract Evaluator")
st.write("Evaluate scientific abstracts using your fine-tuned Mistral LoRA model.")

st.divider()

reference = st.text_area(
    "Reference / Paper Content",
    height=250,
    placeholder="Paste the reference paper content here..."
)

submission = st.text_area(
    "Abstract Submission",
    height=180,
    placeholder="Paste the abstract you want to evaluate here..."
)

evaluate = st.button("Evaluate Abstract", type="primary")

if evaluate:
    if not reference.strip() or not submission.strip():
        st.warning("Please enter both the reference and the abstract submission.")
    else:
        with st.spinner("Evaluating abstract..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "reference": reference,
                        "submission": submission
                    },
                    timeout=300
                )

                if response.status_code == 200:
                    result = response.json()

                    score = result.get("score")
                    rationale = result.get("rationale")
                    raw_output = result.get("raw_output")

                    st.success("Evaluation completed!")

                    st.metric("Score", score)

                    st.subheader("Rationale")
                    st.write(rationale)

                    with st.expander("Raw Model Output"):
                        st.code(raw_output, language="json")

                else:
                    st.error(f"API Error: {response.status_code}")
                    st.text(response.text)

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI. Make sure the backend is running on port 8000.")

            except Exception as e:
                st.error("Something went wrong.")
                st.exception(e)
