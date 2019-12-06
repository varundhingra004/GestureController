""""

CONTROL THE MOUSE POINTER OF A COMPUTER USING HAND GESTURES DETECTED BY THE WEB CAMERA AND REPLACE THE NEED OF A MOUSE/TRACKPAD.
__author__ : Varun Dhingra
__author__ : Dr. Thomas Kinsman


This project uses Open CV3 on Python3 to detect Hand Gestures and control the mouse pointers with those hand gestures. You will need two green colored tapes
wrapped around two fingers (thumb and index finger are more intuitive).



LIBRARIES REQUIRED:

1. Open CV3
2. os
3. time
4. numpy
5. punput

You can use pip to install Open CV, numpy and pynput.





MOUSE EVENTS AND THEIR CORRESPONDING HAND GESTURES:


The hand gesures can be explained by the following actions:

1. OPEN  ===========> The taped fingers are NOT in contact.
2. CLOSE ===========> The taped fingers are in contact.


Mapping of mose events with hand gestures:

1. Single Click ===========>  Fingers start from OPEN position, go to CLOSE position and then go back to OPEN position.
2. Double Click ===========>  Two single clicks done one after the other within a certain time measured in milliseconds (delta_time).
3. Mouse Move   ===========>  ONE finger in front of the camera and move the finger.
4. Click & Drag ===========>  Fingers start from OPEN position, go to CLOSE position, move the closed fingers to the new location, 
                              release the fingers back to OPEN position


Notes:

1. This code should be executed in very bright lighting.

2. Ensure that you have a stable testing environment (external factors like fixed lighting, webcam at a fixed angle, fixed background, etc.)
    External factors are a part of the input, thus if external factors are unstable and keep changing, the input keeps changing.
     This leads to bad test cases and thus getting to know the real issue becomes challenging.

3. ALWAYS ENSURE THAT THERE ARE NO GREEN OBEJCTS, BACKGROUND, etc. IN FRONT OF THE WEB CAM. THIS PROGRAM WILL ERROR OUT IF
    EXECUTED IN A GARDEN, LAWN, FRONT OF A TREE, ANYTHING GREEN.

4. If green background is unavoidable, use a different colored tape. You will have to make subsequent changes to the mask when
    extracting colors.

5. The click and drag operation is commented out because there is a major flaw in the code which might lead to editing the entire
    gesture recognition section of the code.


STAGES IN THIS PROJECT:

1. Color detection
2. Isolate the Green color pixels
3. Morphological Operations to get rid of Salt & Pepper noise.
4. Find Contours
5. State Instantiation
6. Gesture Recognition & Perform mouse actions.


"""
import cv2
import os
import time
import numpy as np
from pynput.mouse import Button, Controller


"""

DISPLAY FLAGS:

The following global variables are used to view the different stages of the project.
These flags are used for debugging.

"""

camera_display_flag 	= True
contours_display_flag 	= False
mask_display_flag 	    = False
post_morphology_flag 	= False


delta_time = 700        # Time within which a double click operation must happen
time_out_time = 3000    # Time within which a state is valid.
QUEUE_LEN 	= 5         # Depth of the rotatinal queue


class State:
    """
    
    State class records the information like No. of Contours detected and the system time at which they were detected. 
    There are three kinds of State:
    
    S0 : No contour detected. This happens when there is no taped finger in front of the camera.
    S1 : One contour detected. This happens when there is one taped finger or when the two fingers are in closed position in front of the camera.
    S2 : Two contours detected. This happens when there are two taped fingers in the open position in front of the camera. 
    
    A gesture is an order of states. For example, a Single click gesture is defined by the order - S2, S1, S2
    
    """
    __slots__ = "id","no_of_contours_detected", "time"

    def __init__(self, id, no_of_contours_detected, time):
        """
        Constructor
        """
        
        self.id = id
        self.no_of_contours_detected = no_of_contours_detected
        self.time = time

    def __str__(self):
        """
        Print state id for debugging. This is like overloading toString() in Java.
        """
        
        return self.id


class RotationQueue:
    
    """
    
    
    The rotational queue keeps track of the history of states and their order upto a certain depth then deletes them.
    The depth is the maximum number of states of a gesture at a given time.
    
    Since a gesture is a series of states, we must keep track of the new states created, old states created and the order
    in which the states were created. 
    
    The maximum number of states that can occur in a gesture is 5. Thus, at a time we need to keep track of only 5 states.
    Once a new state is created and enqueued, the program need not bother about the oldest state. Thus, the oldest state is dequeued
    and the positions of the states are changed accoddingly.
    
    For Example, suppose at a given time, the content of the rotational queue is S1,S0,S1,S2,S1.
    Now, suppose the user does an opens the taped figers. The content of the rotational queue then becomes S0,S1,S2,S1,S2.
    THe first state S1 is dequeued and thus deleted because it has no revelance in the program.
    
    
    """
    
    __slots__ = "queue","length","op_q"

    def __init__(self):
        """
        Constructor
        """
        
        self.length = QUEUE_LEN
        self.queue = [State("s0", 0, 0)] * self.length
        self.op_q = []

    def enqueue(self, state_object):
        """
        Enqueue new state. Push previous states one position towards the head and deque the first element if the queue is full.
        """
        
        assert isinstance(state_object, State), "Only a State object can be enqued"
        if len(self.queue) == self.length:
            self.queue.pop(0) 	# If full, remove the first element.
        self.queue.append(state_object) # Always append new state to the end

    def latest_state(self):
        """
        Obtain the most recent state created.
        """
        
        return self.queue[-1].id

    def oldest_state(self):
        """
        Obtain the oldest relevant state created.
        """
        
        return self.queue[0]

    
    def state_id(self, index):
        """
        GEt the state id in the inputted position of the queue.
        """
        
        return self.queue[index].id
    
    def state_time(self, index):
        
        """
        GEt the state time in the inputted position of the queue.
        """
        
        return self.queue[index].time

    
    def __str__(self):
        """
        Print status of queue for debugging. This is like overloading toString() in Java.
        """
        for state in self.queue:
            self.op_q.append(state.id)
        print(str(self.op_q))
        self.op_q = []
        return ""


def main():
    # The Controller Class is part of the pynput input class.
    mouse         = Controller()
    previous_states 	= RotationQueue()

    # Boundaries for the green color mask
    # THis snippet of code is taken from https://thecodacus.com/gesture-recognition-virtual-mouse-using-opencv-python/
    lowerBound	= np.array([33,80,40])
    upperBound	= np.array([102,255,255])

    # Open the camera:
    cam		= cv2.VideoCapture(0)
    
    # Morphological operations Open followed by closing to get rid of salt and pepper noise.
    # THis snippet of code is taken from https://thecodacus.com/gesture-recognition-virtual-mouse-using-opencv-python/
    kernelOpen	= np.ones((5,5))
    kernelClose	= np.ones((20,20))

    
    current_mouse_position = mouse.position #Store the current mouse pointer positions
    
    # X and Y coordinates of the centroid of the single contour saved in S1.
    cx = 0
    cy = 0
    
    
    update_position = 0

    cv2.namedWindow("controller", cv2.WINDOW_AUTOSIZE)
    while True:
        # Get an image from the webcam:
        ret, img = cam.read()
        # Set the size to a set size:
        img = cv2.resize(img,(800,800))
        # Flip image so that our left side and the image's left side are the same.
        cv2.flip(img, 1)

   #COLOR DETECTION

        imgHSV		= cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # Convert from BGR to HSV
        # create the Mask
        # THis snippet of code is taken from https://thecodacus.com/gesture-recognition-virtual-mouse-using-opencv-python/
        kernelOpen	= np.ones((5,5))
        mask		= cv2.inRange(imgHSV,lowerBound,upperBound) 
        # morphology
        # THis snippet of code is taken from https://thecodacus.com/gesture-recognition-virtual-mouse-using-opencv-python/
        kernelOpen	= np.ones((5,5))
        maskOpen	= cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
        maskClose	= cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
        maskFinal	= maskClose
        
        
        # Contour detection:
        output, contours,hierarchy = cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        #   STATE INSTANTIATION
        #   We do not want consecutive states to be the same in the queue hence,
        #   we always check the last index to ensure that the duplicate state is not created
        
        #   For example, the queue S1,S2,S1,S0,S0 os illegal.
        
        
        change_detected = True # Redundant line of code.
        
        if len(contours) == 0:
            
            """
            If no contours detected and latest state is not S0, then create state S0 and enqueue.
            """
            
            if ( previous_states.latest_state() != "s0" ):
                s0 = State("s0", 0, int(round(time.time() * 1000)))
                previous_states.enqueue(s0)
                change_detected = True
        elif len(contours) == 1:
            """
            If 1 contour is detected and latest state is not S1, then create state S1 and enqueue.
            
            """
            
            if ( previous_states.latest_state() != "s1" ):
                s1 = State("s1", 1, int(round(time.time() * 1000)))
                previous_states.enqueue(s1)
                change_detected = True
        elif len(contours) == 2:
            
            """
            If 2 contours detected and latest state is not S2, then create state S2 and enqueue.
            
            """
            
            if ( previous_states.latest_state() != "s2" ):
                s2 = State("s2", 2, int(round(time.time() * 1000)))
                previous_states.enqueue(s2)
                change_detected = True
                # print( previous_states, end=" " )
        else:
            # Noise. To be ignored.
            # No-op.  Do nothing.... 
            pass
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    #RECOGNIZE GESTURES AND PERFORM MOUSE ACTIONS
    
        # mouse_x and mouse_y are the new center of mass of any single contour
    
        if change_detected is True:
            """
            Check for a stable double click, based on time and queue status.
            
            Queue status for double click - S2, S1, S2, S1, S2
            
            """
            if  previous_states.state_id(0) == "s2" and \
                previous_states.state_id(1) == "s1" and \
                previous_states.state_id(2) == "s2" and \
                previous_states.state_id(3) == "s1" and \
                previous_states.state_id(4) == "s2" and \
                previous_states.state_time(4) - previous_states.state_time(0) <= delta_time:
                    # Press left mouse buttin 2 times.
                    mouse.click(Button.left, 2) 
            
            
            #   Check for a stable single click, based on time and queue status.
            #   Queue status for double click - <Any state>, <Any state>, S2, S1, S2
            
            elif previous_states.state_id(4) == "s2" and \
                 previous_states.state_id(3) == "s1" and \
                 previous_states.state_id(2) == "s2" and \
                 int(round(time.time() * 1000)) - previous_states.state_time(4) > delta_time:
                      # Press left mouse buttin once
                     mouse.click(Button.left, 1)
                     
                     """
                     After the button is pressed, we need to flush the rotation queue 
                     with state s0. If we do not flush, then the states s2, s1, s2 detected
                     previously (and which have already been implemented as single click)
                      will affect the decision making of the states enqueued.
                     """
                     
                     previous_states.enqueue(State("s0", 0, int(round(time.time() * 1000))))
                     previous_states.enqueue(State("s0", 0, int(round(time.time() * 1000))))
                     previous_states.enqueue(State("s0", 0, int(round(time.time() * 1000))))
                     previous_states.enqueue(State("s0", 0, int(round(time.time() * 1000))))
                     previous_states.enqueue(State("s0", 0, int(round(time.time() * 1000))))
                     
                     """
                     
                     The following is the code for click and drag which has a major flaw and hence has been commented out. 
                     I had no time to correct it because it would lead to redoing the entire code.
                     
                     """
                     
            # elif previous_states.state_id(4) == "s1" and \
#                  previous_states.state_id(3) == "s2" and \
#                  int(round(time.time() * 1000)) - previous_states.state_time(4) > delta_time:
#                     # print("CLick and Drag")
#                     # cv2.waitKey()

#                     while previous_states.latest_state() == "s1":
#                         for contour in contours:
#                             moment = cv2.moments(contour)
#                             if moment['m00'] == 0:
#                                 moment['m00'] = 1
#                             cx = int(moment['m10']/moment['m00'])
#                             cy = int(moment['m01']/moment['m00'])
#                             cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)
#                         update_position = (cx,cy)
#                         mouse.position = update_position
#                     mouse.release(Button.left)
            elif previous_states.latest_state() == "s1":
                
                """
                If none of the previous cases (SIngle click, double click , click & drag) are fulfulled
                and the latest state in the rotation queue is s1, then it is a mouse move operation.
                
                For mouse move, we need to calculate the centroid of the contour detected and update the mouse pointer
                positions to the coordinates of this contour.
                
                The contour calculation is done below. The code is taken from OpenCV docs.
                https://docs.opencv.org/3.2.0/dd/d49/tutorial_py_contour_features.html
                
                """
                
                for contour in contours:
                    moment = cv2.moments(contour)
                    if moment['m00'] == 0:
                        moment['m00'] = 1
                    cx = int(moment['m10']/moment['m00'])
                    cy = int(moment['m01']/moment['m00'])
                    cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)
                update_position = (cx,cy)
                mouse.position = update_position

            else:
                """
                If none of the gestures are recognized, do nothing.
                """
                None
                
        """
                If the roation queue does not get updated for a while or is in the same state for a certain period of time (time_out_time), 
                the states that were already in the queue become invalid. They thus need to be flushed out of the queue to
                ensure that they do not pariciapte in any decision making.
                
                Example : 
                
                If at a certain time, the queue status is S0, S2, S0, S2, S1.
                
                Now suppose no action is taken for a certain period of time. After that if the user
                wants to perform a double click event, it will lead to an issue,
                
                This is because the moment the user begins the double click action,
                when he opens his fingers, the opening motion will combine with the previous 
                queue states and generate a single click even. Here is a detailed queue
                diagram :
                
                Current queue state : S0, S2, S0, S2, S1.
                
                When user starts the double click(S2, S1, S2, S1, S2), the queue state updates as below.
                
                Updated queue state: S2, S0, S2, S1, S2. ====> This leads to a single click.
                
                
                
                To avoid the above, we the program waits for a certain time out period and then flushes the queue.
                
                Current queue state : S0, S2, S0, S2, S1.
                
                Time out period extended.
                
                Updated queue stauts : S0, S0, S0, S0, S0
                
                Double click action performed.
                
                Updated queue status : S2, S1, S2, S1, S2
                
                
                
        """
                
        if (int(round(time.time() * 1000)) - previous_states.state_time(0)) > time_out_time:
            previous_states.enqueue(State("s0", 0, int(round(time.time() * 1000))))
            previous_states.enqueue(State("s0", 0, int(round(time.time() * 1000))))
            previous_states.enqueue(State("s0", 0, int(round(time.time() * 1000))))
            previous_states.enqueue(State("s0", 0, int(round(time.time() * 1000))))
            previous_states.enqueue(State("s0", 0, int(round(time.time() * 1000))))
            print(("TIMED OUT"))

        """
            Display the different views using the values of the display flags.
        """
            
        if ( camera_display_flag ):
            cv2.imshow( "LiveVideo", img )

        if ( contours_display_flag ):
            cv2.imshow( "ContoursDetected", output )

        if ( mask_display_flag ):
            cv2.imshow( "Mask of Green Pixels",  mask )

        if ( post_morphology_flag ):
            cv2.imshow( "Final Mask", maskFinal )

        # Program Termination
        key = cv2.waitKey(1)
        if key > 0 :
            print(key)
            cam.release()
            break

def test():
    """
    
    Test method. This is just used for testing and experimentation
    This is not caled in the final program implementation.
    
    """
    collection = RotationQueue()
    collection._enqueue(State("item 1", 5, 10))
    collection._enqueue(State("item 2", 5, 10))
    collection._enqueue(State("item 3", 5, 10))
    collection._enqueue(State("item 4", 5, 10))
    collection._enqueue(State("item 5", 5, 10))

    for item in collection.queue:
        print(item)


#Call the main()
if __name__ == '__main__':
    main()

