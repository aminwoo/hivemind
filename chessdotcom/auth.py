import requests
from bs4 import BeautifulSoup

def get_session_key(username: str, password: str) -> str:
    """Log into chess.com and retrieve php session key 

    Args:
        username (str): chess.com username 
        password (str): chess.com password

    Returns:
        str: PHPSESSID
    """
    s = requests.Session()
    login_url = 'https://www.chess.com/login_and_go?returnUrl=https://www.chess.com/'

    response = s.get(login_url, allow_redirects=True)  
    soup = BeautifulSoup(response.content, 'html.parser')  
    token_input = soup.find('input', {'name': '_token'})  
    token = token_input.get('value')

    login_data = {'_username': username,
                    '_password': password,
                    'login': '',
                    '_target_path': 'https://www.chess.com/game/live',
                    '_token': token
                    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36',
        'Referer': login_url
    }
    r = s.post('https://www.chess.com/login_check', data=login_data, headers=headers, allow_redirects=False, verify=True)
    return r.headers['Set-Cookie'].split('=')[1].split(';')[0]

if __name__ == '__main__':
    print(get_session_key("pumpkinspicedream", "Qwerty12"))
    print(get_session_key("pumpkinspicefever", "Qwerty12"))