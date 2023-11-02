@ECHO OFF
CALL conda create --name CleverCaption python=3.11 -y
CALL conda activate CleverCaption
CALL pip install -r requirements.txt
ECHO CleverCaption environment setup is complete.
PAUSE