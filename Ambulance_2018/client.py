import json
from hps.clients import SocketClient
import sys
import time
from sklearn.cluster import KMeans
import numpy as np
import math
import copy
#from numba import jit

HOST = '127.0.0.1'
PORT = 5001

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
            res_hos[sorted_by_number_ambulances[i][0]] = {'xloc' : int(x), 'yloc' : int(y)}
            mappingCenterToAmbulances[(int(x),int(y))] = sorted_by_number_ambulances[i][1]
       
        # can do this because labels are 0,1,2,3,4
        for i in range(len(clustering)):
            #print("clustering")
            cluster = clustering[i]
            #print(len(cluster))
            center = tuple(centers[i])
            center = (int(center[0]), int(center[1]))
            ambulancesOperatingInThisCluster = mappingCenterToAmbulances[center]
            ambulanceBucketPatient = self.radDivide(cluster, ambulancesOperatingInThisCluster, center)
            # decide on order for ambulance using search
            for ambulance_id in ambulanceBucketPatient:
                patientsToVisit = ambulanceBucketPatient[ambulance_id]
                sorted(patientsToVisit, key=lambda patient_id : self.patients[patient_id]['rescuetime'])
                #aa
                #print(ambulance_id)
                #print(center)
                #print(patientsToVisit)
                
                numBuckets = math.ceil(len(patientsToVisit)/5)
                nn = len(patientsToVisit)/numBuckets
                globalOrder = []
                ii = 0
                tt = 0
                while ii < numBuckets:
                    #print(ii)
                    #print(nn)
                    ll = patientsToVisit[int(ii * nn): int((ii + 1) * nn)]
                    #print(ll)
                    order, tt = self.orderOfPatients(ll, center, tt)
                    #order.append(-1)
                    #order = [ll[x] if x is not -1 else -1 for x in order]
                    globalOrder += order
                    ii += 1 

                #print("bestPath")
                #print(order)
    
                res_amb[ambulance_id] = self.parse(globalOrder, self.ambulances[ambulance_id]['starting_hospital'])
        return (res_hos, res_amb)

    def orderOfPatients(self, patientsToVisit, center, time):
        self.onPath = [0]*len(patientsToVisit)
        self.path = []
        self.bestPath = []
        self.pathLen = 0
        self.bestPathValue = 0
        self.timeT = 0
        visited = [0] * len(patientsToVisit)
        for i in range(len(patientsToVisit)):
            self.dfs(i, patientsToVisit, center, time)
        print("saved " + str(self.bestPathValue) + "/" + str(len(patientsToVisit)))
        #print(self.bestPath)
        return self.bestPath, time + self.timeT
    
    # nextPatientToSaveIndex might also cater hospital
    def dfs(self, nextPatientToSaveIndex, patientsToVisitIndices, center, ttt):
        self.path.append(nextPatientToSaveIndex)
        if nextPatientToSaveIndex != -1:
            self.onPath[nextPatientToSaveIndex] = 1
            self.pathLen += 1
        if self.pathLen == len(patientsToVisitIndices):
            translatedPath = [patientsToVisitIndices[x] if x is not -1 else -1 for x in self.path]
            translatedPath.append(-1)
            value, timetaken = self.peopleSaved(translatedPath, center, ttt)
            if value > self.bestPathValue:
                self.bestPath = copy.deepcopy(translatedPath)
                self.bestPathValue = value
                self.timeT = timetaken
        else:
            for i in range(len(patientsToVisitIndices)):
                if self.onPath[i] == 0:
                    self.dfs(i, patientsToVisitIndices, center, ttt)
            # hospital to hospital case
            if nextPatientToSaveIndex != -1:
                self.dfs(-1, patientsToVisitIndices, center, ttt) 
        self.path.pop()
        if nextPatientToSaveIndex != -1:
            self.onPath[nextPatientToSaveIndex] = 0
            self.pathLen -= 1
        
    def dist(self, a, b):
        return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def canBeSaved(self, hosp, patient, time):
        return True

    def parse(self, order, hospital_id):
        return [('p'+ str(x)) if x is not -1 else ('h'+str(hospital_id)) for x in order]

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
    
    def peopleSaved(self, path, hospital, tt):
        ret = 0
        time = 0
        locX = hospital[0]
        locY = hospital[1]
        p1, p2, p3, p4 = [-1]*4
        for p in path:
            if p == -1:
                num_p = (p1 > -1) + (p2 > -1) + (p3 > -1) + (p4 > -1)
                time += self.dist([hospital[0], hospital[1]], [locX, locY]) + num_p
                locX = hospital[0]
                locY = hospital[1]
                if p4 != -1:
                    ret += self.saved(self.patients[p4], time + tt)
                if p3 != -1:
                    ret += self.saved(self.patients[p3], time + tt)
                if p2 != -1:
                    ret += self.saved(self.patients[p2], time + tt)
                if p1 != -1:
                    ret += self.saved(self.patients[p1], time + tt)
                p1 = -1
                p2 = -1
                p3 = -1
                p4 = -1
            else:
                if p4 != -1:
                    return -1, 0
                p4 = p3
                p3 = p2
                p2 = p1
                p1 = p
                time += self.dist( [self.patients[p]['xloc'], self.patients[p]['yloc']], [locX, locY]) + 1
                locX = self.patients[p]['xloc']
                locY = self.patients[p]['yloc']
        return ret, time
    
    def saved(self, pID, time):
	    if time > pID['rescuetime']:
		    return 0
	    else:
		    return 1
