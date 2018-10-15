import json
from hps.clients import SocketClient
import sys
import time

HOST = '127.0.0.1'
PORT = 5000

class Player(object):
    def __init__(self, name):
        self.name = name
        self.client = SocketClient(HOST, PORT)
        self.client.send_data(json.dumps({'name': self.name}))


    def play_game(self):
        buffer_size_message = json.loads(self.client.receive_data(size=2048))
        buffer_size = int(buffer_size_message['buffer_size'])
        game_state = json.loads(self.client.receive_data(size=buffer_size))
        self.patients = game_state['patients']
        self.hospitals = game_state['hospitals']
        self.ambulances = game_state['ambulances']

        # Get hospital locations and ambulance routes
        (hos_locations, amb_routes) = self.your_algorithm()

        response = {'hospital_loc': hos_locations, 'ambulance_moves': amb_routes}
       
        print('sending data')
        min_buffer_size = sys.getsizeof(json.dumps(response))
        print(min_buffer_size)
        print(response)

        buff_size_needed = 1 << (min_buffer_size - 1).bit_length()
        buff_size_needed = max(buff_size_needed, 2048)
        buff_size_message = {'buffer_size': buff_size_needed}
        self.client.send_data(json.dumps(buff_size_message))
        time.sleep(2)
        self.client.send_data(json.dumps(response))

        # Get results of game
        game_result = json.loads(self.client.receive_data(size=8192))
        if game_result['game_completed']:
            print(game_result['message'])
            print('Patients that lived:')
            print(game_result['patients_saved'])
            print('---------------')
            print('Number of patients saved = ' + str(game_result['number_saved']))
        else:
            print('Game failed run/validate ; reason:')
            print(game_result['message'])

    def your_algorithm(self):
        """
        PLACE YOUR ALGORITHM HERE

        You have access to the dictionaries 'patients', 'hospitals', and 'ambulances'
        These dictionaries are structured as follows:
            patients[patient_id] = {'xloc': x, 'yloc': y, 'rescuetime': rescuetime}

            hospitals[hospital_id] = {'xloc': None, 'yloc': None, 'ambulances_at_start': [array of ambulance_ids]}

            ambulances[ambulance_id] = {'starting_hospital': hospital_id, 'route': None}

        IMPORTANT: Although all values are integers (inlcuding ids) JSON converts everything into strings. Hence,
                   if you wish to use the values as integers, please remember to cast them into ints. Likewise, to index
                   into the dictionaries please remember to cast numeric ids into strings i.e. self.patients[str(p_id)]

        RETURN INFO
        -----------
        You must return a tuple of dictionaries (hospital_locations, ambulance_routes). These MUST be structured as:
            hospital_locations[hospital_id] = {'xloc': x, 'yloc': y}
                These values indicate where you want the hospital with hospital_id to start on the grid

            ambulance_routes[ambulance_id] = {[array of stops along route]}
                This array follows the following rules:
                    - The starting location of each ambulance is known so array must start with first patient that
                      it must pick up (or hospital location that it will head to)
                    - There can only ever be up to 4 patients in an ambulance at a time so any more than 4
                      patient stops in a row will result in an invalid input
                    - A stop for a patient is a string starting with 'p' followed by the id of the patient i.e. 'p32'
                        + The 'p' can be uppercase or lowercase
                        + There can be no whitespace, i.e. 'p 32' will not be accepted
                    - A stop for a hospital is the same as the patient except with an 'h', i.e. 'h3'
                        + The 'h' can also be uppercase or lowercase

            Example:
                ambulance_routes[3] = ['p0', 'p43', 'h4', 'p102', 'p145', 'p241', 'p32', 'h1']

                This will be read as ambulance #3 starts at it's designated hospital, goes to patient #0, then to
                patient #43, then drops both off at hospital #4, then picks up patients #102, #145, #241, #32 in that
                order then drops all of them off at hospital #1
        """
        res_hos = {}
        res_amb = {}
        res_hos[0] = {'xloc':36, 'yloc':52}
        res_hos[1] = {'xloc':18, 'yloc':20}
        res_hos[2] = {'xloc':70, 'yloc':85}


        # testing no movement/time increment
        res_amb[0] = ['p0', 'p1', 'p2', 'p3', 'h0', 'p4', 'h0']

        # testing more than 4 pickups
        res_amb[1] = ['p5', 'p6', 'p7', 'p8', 'h1']

        # testing duplicate person id
        res_amb[2] = ['p9', 'p10', 'p11', 'h2']
        res_amb[3] = ['p12', 'p11', 'h2']

        # testing rescue overtime (t = 0 before unloading)
        res_amb[4] = ['p13', 'h0']

        # testing rescue barely making it (t = 0 after unloading counts as rescued)
        res_amb[5] = ['p14', 'h0']
     
        return (res_hos, res_amb)