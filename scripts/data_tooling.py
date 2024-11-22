import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.utils import (
    get_song_id_by_name_artist,
    get_songs_by_artist
)


def main():
    # name, artist = "Higher Ground", "Red Hot Chili Peppers"
    # _id = get_song_id_by_name_artist(name, artist)
    # print(f"ID for {name} - {artist}: {_id}\n")

    search_artist = "Red Hot Chili Peppers"
    songs_list = get_songs_by_artist(search_artist)
    print(f"All songs for artist {search_artist}")
    for i, song in enumerate(songs_list, start=1):
        print(f"{i}. {song[0]}: {song[1]}")


if __name__ == '__main__':
    main()
