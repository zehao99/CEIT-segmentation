from json import loads


def get_config():
    """
    Get the config in config.json

    Returns:
        config: config info for training

    """
    with open('./config.json', 'r', encoding='utf-8') as f:
        config = loads(f.read())
    return config
