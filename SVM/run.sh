python3 -m venv ./venv
source venv/bin/activate
python3 -m pip install -q -r requirements.txt
python3 -m src.main "$@"
deactivate
