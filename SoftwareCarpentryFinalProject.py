import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ["Sd33f260b7da74a118592ffc7f01837c0"],
                                                           client_secret=os.environ["b5e33b148eeb4d74a3d8c5e224c8d8a8"]))