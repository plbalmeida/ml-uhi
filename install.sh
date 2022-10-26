# creating virtual environment
sudo apt install python3-virtualenv;
virtualenv -p python3 venv;
. venv/bin/activate;

# installing pre-commit to update requirements.txt file automatically
pip3 install -r requirements.txt; 
pip3 install pre-commit;
pre-commit install;

echo "repos:
-   repo: local
    hooks:
      - id: requirements
        name: requirements
        entry: bash -c 'venv/bin/pip3 freeze > requirements.txt; git add requirements.txt'
        language: system
        pass_filenames: false
        stages: [commit]" > .pre-commit-config.yaml