import json
from hps.clients import SocketClient
import sys
import time
from sklearn.cluster import KMeans
import numpy as np
import math

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
        self.patients = {int(key): value for key, value in game_state['patients'].items()}
        self.hospitals = {int(key): value for key, value in game_state['hospitals'].items()}
        self.ambulances = {int(key): value for key, value in game_state['ambulances'].items()}

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

            ambulances[ambulance_id] = {'starting_hospital': hospital_id}

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
        counter = 0
        p_count = 0

        total_patients = len(self.patients)

        ## create data for clustering
        locData = []
        for patient_id in self.patients:
            patient = self.patients[patient_id]
            locData.append((patient['xloc'], patient['yloc']))
        locData = np.array(locData)
        
        ## cluster
        kmeans = KMeans(n_clusters=5, random_state=0).fit(locData)
        
        ## assign hospitals
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        clustering = {i: np.where(labels == i)[0] for i in range(5)}
        sorted_by_number_elements_cluster = sorted(clustering.items(), key=lambda kv: len(kv[1]))
        
        ## creating res_hos [i.e. which hospital to place where, based on num of avail ambulances]
        hospitalsByNumber = {}
        for hospital_id in self.hospitals:
            hospitalsByNumber[hospital_id] = self.hospitals[hospital_id]['ambulances_at_start']
        sorted_by_number_ambulances = sorted(hospitalsByNumber.items(), key=lambda kv: len(kv[1]))
        ## cluster with min number of patients matched with hospital with min number of avail ambulances
        mappingCenterToAmbulances = {}
        for i in range(len(sorted_by_number_ambulances)):
            x,y = centers[sorted_by_number_elements_cluster[i][0]]
            res_hos[sorted_by_number_ambulances[i][0]] = {'xloc' : x, 'yloc' : y}
            mappingCenterToAmbulances[(x,y)] = sorted_by_number_ambulances[i][1]
       
        # can do this because labels are 0,1,2,3,4
        for i in range(len(clustering)):
            cluster = clustering[i]
            center = tuple(centers[i])
            ambulancesOperatingInThisCluster = mappingCenterToAmbulances[center]
            ambulanceBucketPatient = self.radDivide(cluster, ambulancesOperatingInThisCluster, center)
            # decide on order for ambulance using search
            for ambulance_id in ambulanceBucketPatient:
                patientsToVisit = ambulanceBucketPatient[ambulance_id]
                sorted(patientsToVisit, key=lambda patient_id : self.patients[patient_id]['rescuetime'])
                #aa
                print(ambulance_id)
                print(center)
                print(patientsToVisit)

                #order = orderOfPatients(patientsToVisit, center)
                res_amb[ambulance_id] = ['h'+str(self.ambulances[ambulance_id]['starting_hospital'])]
                #parse(order, patientsToVisit)
        return (res_hos, res_amb)

    def orderOfPatients(self, patientsToVisit, center):
        return patientsToVisit
        #if len(patientsToVisit) >= THRESHOLD:
        #    # do something
        #else:
        #    visited = [0]*len(patientsToVisit)
        #    for i in range(len(patientsToVisit)):
        #        visited[i] = 1
        #        if canBeSaved(center, 
        #        dfs(i, patientsToVisit, visited, [i], 1, center, 0)
        #        visited[i] = 0
    
    # nextPatientToSaveIndex might also cater hospital
    #def dfs(nextPatientToSaveIndex, allPatients, visited, path, ambulanceLocation, timeElapsed):
        
    def parse(self, order):
        return 0

    def radDivide(self, patientsIndexList, ambulances, hospital):
        num_ambulances = len(ambulances)
        num_patients = len(patientsIndexList)
        ret = {}
        angle = []
        for pID in range(0, num_patients):
            pat = self.patients[patientsIndexList[pID]]
            theta = math.atan2(pat['yloc'] - hospital[1], pat['xloc'] - hospital[0])
            if theta < 0.0:
                theta += 2 * math.pi
            angle.append([patientsIndexList[pID], theta])
        angle = sorted(angle, key = lambda x: x[1])
        n = num_patients/num_ambulances
        for i in range(0, num_ambulances):
            ambID = ambulances[i]
            ambPat = []
            for j in range(int(i * n), int((i + 1) * n)):
                ambPat.append(angle[j][0])
            ret[ambID] = ambPat
        return ret
