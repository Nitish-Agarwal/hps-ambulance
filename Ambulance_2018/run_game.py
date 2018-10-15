import sys

from getopt import getopt

from server import GameServer
from client import Player

def main():
    """
    This invokes the game and can run either the server or the client.

    To run the server run: python run_game.py -t s -i <path to input text file>

    To run the client run: python run_game.py -t c -i <name of player>
    """
    try:
        opts, args = getopt(sys.argv[1:], 't:n:i:')
    except GetoptError:
        sys.stderr.write('Error parsing options\n')
        sys.stderr.write(__doc__)
        exit(-1)

    name = 'Default_Player'
    input_file = None

    server_client_flag = None

    for o, a in opts:
        if o == '-t':
            server_client_flag = a
        elif o == '-n':
            name = a
        elif o == '-i':
            input_file = a

    if server_client_flag == 's':
        gs = GameServer(input_file)
    elif server_client_flag == 'c':
        pl = Player(name)
        pl.play_game()
    else:
        print('None or invalid game side provided. Please specify "s" for server or "c" for client')
        exit(1)

if __name__ == '__main__':
    main()
