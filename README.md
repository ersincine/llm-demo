# Demo

## Instructions

1. Open a terminal and navigate to the directory containing this README. (Use `cd` command for this.)
2. Run `pip install -r requirements.txt` to install dependencies. (You may need to use `pip3` instead of `pip` if it is associated with Python 2 on your system.)
3. Run `streamlit run app.py` to start the app.

## Notes

- Tone selection, assistant and user descriptions have not been implemented yet. (First I want to make sure the core functionality works as expected.)
- We use `OpenAI` (text-davinci-003), not `ChatOpenAI` (gpt-3.5-turbo or gpt-4). (This was for compatibility with the Replicate's API. I will change this.)
