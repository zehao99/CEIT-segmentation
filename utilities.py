from json import loads

def get_config():
    with open('./config.json', 'r', encoding='utf-8') as f:
        config = loads(f.read())
    return config