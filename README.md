# deepface-facematch

This script uses deepface to rank a list of faces (doubles) based on how similar they are to another face (the subject).

The similarity between two images is represented by a float ranging from 0 to 1 (0 = no similarity, 1 = identical).

data/ contains a couple mock images that can be used for testing, the script uses these images as the default.

This only focuses on the facial vector embeddings, and throws out all other data provided by deepface (facial_area, eye position, etc.). See deepface docs if you need to make use of those parameters.

## Setup and usage

#### Windows
```sh
python -m venv .venv
source .venv\Scripts\activate
pip install -r requirements.txt
python -m main <args>
```

#### MacOS/Linux
```sh
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
python3 -m main <args>
```

Example usage with args:
```sh
python3 -m main --subjectPath "path/to/subject.jpg" --doublesPath "path/to/doubles" --model_name "Facenet" --no-verbose
```