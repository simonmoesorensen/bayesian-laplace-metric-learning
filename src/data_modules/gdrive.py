import requests
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    """
    https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
    """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        total = int(response.headers.get('content-length', 0))

        with open(destination, "wb") as f, tqdm(total=total, unit='iB', unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    size = f.write(chunk)
                    pbar.update(size)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id, 'confirm': 't'}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)
