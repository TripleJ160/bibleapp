
# Bible Study App

## Overview

The Bible Study App is a web-based application that allows users to explore and study the Bible using a language model powered by OpenAI. It provides users with the ability to ask questions related to their Christian faith and receive responses based solely on the Bible.

## Features

- **Select Bible Book:** Users can choose from a list of books in the Bible, including both Old Testament and New Testament books.

- **Read Bible Content:** Upon selecting a book, the app displays the content of that book for the user to read.

- **Ask Questions:** Users can ask questions related to the selected book or any biblical topic.

- **AI-Powered Responses:** The app uses an AI language model based on the Bible to provide insightful answers to user questions.

- **PDF Embedding:** The app displays PDF versions of Bible books, allowing users to read and study them directly within the app.

- **Personalized URL:** The app has a personalized URL, making it easily accessible for users.

## Demo

[Live Demo](https://bibleai.streamlit.app/)

## Installation

1. Clone the repository:

```
git clone https://github.com/TripleJ160/bibleapp/
cd bibleapp
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Run the app locally:

```
streamlit run app.py
```

## Built With

- [Streamlit](https://streamlit.io/) - The web framework used for building the app.
- [OpenAI](https://openai.com/) - AI language model for answering Bible-related questions.
- [PyPDF2](https://pythonhosted.org/PyPDF2/) - Python library for working with PDF files.
- [LangChain](https://python.langchain.com/) - Python library for language model chaining.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## Credits

Used resources from @PromptEnginner to learn the concepts and frameworks required to build this app
