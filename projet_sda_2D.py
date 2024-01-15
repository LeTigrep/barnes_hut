from numpy import *
from random import random,seed
from time import time
import pygame
import numpy as np
from numpy import zeros, array, sqrt
import random
from math import pi, cos, sin

pygame.init()
screen_size = (1000, 800)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("N-Body Simulation")
clock = pygame.time.Clock()



seed(323)

G = 1.0 # constante gravitationelle
THRESHOLD = 1 # Nombre de corps après lesquels subdiviser Quad
MAXDEPTH = 10
THETA = 0.5 # Rapport de précision Barnes-Hut
ETA = 0.5 # Facteur de ramollissement


NUM_CHECKS = 0 

class QuadTree:
    """ Class container for for N points stored as a 2D Quadtree """
    root = None
    def __init__(self, bbox, N, theta = THETA):
        self.bbox = bbox
        self.N = N
        self.theta = theta
        
    def reset(self):
        self.root = Quad(self.bbox)
    def generate(self):
        # Build up nodes of tree fresh for bodies
        self.reset()
        for x in range(self.N):
            # For each body, add to root
            self.root.addBody(x,0)

    def updateSys(self, dt):
        self.calculateBodyAccels()
        global VEL
        VEL += ACC * dt

    def calculateBodyAccels(self):
        # Update ACC table based on current POS
        for k in range(self.N):
            ACC[k] = self.calculateBodyAccel(k)
    def calculateBodyAccel(self, bodI):
        return self.calculateBodyAccelR(bodI, self.root)

    def calculateBodyAccelR(self, bodI, node):
        # Calculate acceleration on body I
        # key difference is that body is ignored in calculations
        acc = zeros(2,dtype=float)
        if (node.leaf):
            # print "Leaf"
            # Leaf node, no children
            for k in node.bods:
                if k != bodI: # Skip same body
                    acc += getForce( POS[bodI] ,1.0,POS[k],MASS[k])
        else:
            s = max( node.bbox.sideLength )
            d = node.center - POS[bodI]
            r = sqrt(d.dot(d))
            if (r > 0 and s/r < self.theta):
                # Far enough to do approximation
                acc += getForce( POS[bodI] ,1.0, node.com, node.mass)
            else:
                # Too close to approximate, recurse down tree
                for k in range(4):
                    if node.children[k] != None:
                        acc += self.calculateBodyAccelR(bodI, node.children[k])

        return acc

        

def getForce(p1,m1,p2,m2):
    # need to d
    global NUM_CHECKS
    d = p2-p1
    r = sqrt(d.dot(d)) + ETA
    f = array( d * G*m1*m2 / r**3 )
    NUM_CHECKS += 1
    return f


class Quad:
    """ A rectangle of space, contains point bodies """
    def __init__(self,bbox,bod = None,depth=0):
        self.bbox = bbox
        self.center = bbox.center
        self.leaf = True # Whether is a parent or not
        self.depth = depth
        if bod != None: # want to capture 0 int also
            self.setToBody(bod)
            self.N = 1
        else:
            self.bods = []
            self.mass = 0.
            self.com = array([0,0], dtype=float)
            self.N = 0
            
        self.children = [None]*4 # top-left,top-right,bot-left,bot-right

    def addBody(self, idx,depth):
        # Recurse if you have a body or you have children
        if len(self.bods) > 0 or not self.leaf:
            # Not empty
            if (depth >= MAXDEPTH):
                self.bods.append(idx)
            else:
                # Subdivide tree
                subBods = [idx] # bodies to add to children
                if len(self.bods) > 0:
                    # if node has no children yet, move own body down to child
                    subBods.append(self.bods[0])
                    self.bods = []

                for bod in subBods:
                    quadIdx = self.getQuadIndex(bod)
                    if self.children[quadIdx]:
                        # child exists, recursively call 
                        self.children[quadIdx].addBody(bod,depth+1)
                    else:
                        # create child
                        subBBox = self.bbox.getSubQuad(quadIdx)
                        self.children[quadIdx] = Quad(subBBox, bod,depth+1)

                self.leaf = False

            bodyMass   = MASS[idx]
            self.com   = (self.com * self.mass + POS[idx] * bodyMass) / (self.mass + bodyMass)
            self.mass += bodyMass
        else:
            # Empty Quad, add body directly
            self.setToBody(idx)

        self.N += 1 # Number of bodies incremented
    
        
    def setToBody(self,idx):
        self.bods = [idx]
        self.mass = float( MASS[idx].copy() )
        self.com  = POS[idx].copy()

    def getQuadIndex(self,idx):
        return self.bbox.getQuadIdx(POS[idx])


class BoundingBox:
    def __init__(self, box, dim=2):
        # Ensure the dimension is correct for the bounding box
        assert(dim*2 == len(box))
        # Convert the box to a numpy array of float type
        self.box = array(box, dtype=float)
        # Calculate the center of the bounding box
        self.center = array([(self.box[2]+self.box[0])/2, (self.box[3]+self.box[1])/2], dtype=float)
        # Store the dimension
        self.dim = dim
        # Calculate the side length of the bounding box
        self.sideLength = self.max() - self.min()


    def max(self):
        return self.box[self.dim:]
    def min(self):
        return self.box[:self.dim]
    def inside(self, p):
        # Check if each coordinate of p is inside the bounding box
        if any(p < self.min()) or any(p > self.max()):
            return False
        else:
            return True

    def getQuadIdx(self,p):   
        # y goes up
        # 0 1
        # 2 3
        # Compare the coordinates of p with the center of the bounding box
        if p[0] > self.center[0]: # x > mid
            if p[1] > self.center[1]: # y > mid
                return 1
            else:
                return 3
        else:
            if p[1] > self.center[1]: # y > mid
                return 0
            else:
                return 2
    def getSubQuad(self,idx):
        # 0 1
        # 2 3
        # [x  y x2 y2]
        #  0  1  2  3
        # Initialize an array for the new coordinates of the bounding box
        b = array([None,None,None,None])
        # Determine the new coordinates based on the quadrant index
        if idx % 2 == 0:
            # Even #, left half
            b[::2] = [self.box[0], self.center[0]] # x - midx
        else:
            b[::2] = [self.center[0], self.box[2]] # midx - x2
        if idx < 2:
            # Upper half (0 1)
            b[1::2] = [self.center[1], self.box[3]] # midy - y2
        else:
            b[1::2] = [self.box[1], self.center[1]] # y - midy
        # Create and return a new instance of BoundingBox
        return BoundingBox(b,self.dim)




def draw_bodies_pygame():
    for i in range(N):
        if BOUNDS.inside(POS[i]):
            x, y = convert_to_screen_coords(POS[i])
            pygame.draw.circle(screen, (0, 0, 0), (x, y), int(MASS[i] * 2))

def draw_bbox_pygame(node):
    if node is not None:
        x0, y0 = convert_to_screen_coords(node.bbox.min())
        x1, y1 = convert_to_screen_coords(node.bbox.max())
        pygame.draw.rect(screen, (0, 0, 255), (x0, y0, x1 - x0, y1 - y0), 1)
        if node.leaf:
            color = (0, 255, 0)  # Green for leaf nodes
        else:
            color = (255, 0, 0)  # Red for parent nodes
        pygame.draw.rect(screen, color, (x0, y0, x1 - x0, y1 - y0), 1)
        for child in node.children:
            draw_bbox_pygame(child)

def convert_to_screen_coords(p):
    screen_pos = (p - BOUNDS.min()) / (BOUNDS.max() - BOUNDS.min()) * np.array(screen_size)
    return np.trunc(screen_pos).astype(int)


N = 100
BOUNDS = BoundingBox([0,0,10,10])



# Global variables
# 2D Position
MASS = zeros(N,dtype=float)
POS = zeros((N,2),dtype=float)
VEL = zeros((N,2),dtype=float)
ACC = zeros((N,2),dtype=float)
for i in range(N):
    MASS[i] = 1 
    POS[i] = BOUNDS.min() + array([random.random(), random.random()]) * BOUNDS.sideLength


# Calculate the center of mass for the entire system
total_mass = MASS.sum()
center_of_mass = np.average(POS, axis=0, weights=MASS)
# Maximum speed calculation
DT = 0.00001
T = 0
max_speed = min(BOUNDS.sideLength) / (2 * DT)
# Define the central point of the simulation (where the horizontal and vertical lines intersect)
central_point = np.array([BOUNDS.sideLength[0] / 2, BOUNDS.sideLength[1] / 2])

sys = QuadTree(BOUNDS, N)
# Calculate the initial angles for each body
angles = np.arctan2(POS[:, 1] - central_point[1], POS[:, 0] - central_point[0])

angular_speed = 0.01
# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for i in range(N):
        radius = np.linalg.norm(POS[i] - central_point)
        angles[i] += angular_speed  # Update the angle

        # Update POS[i] to be on the circle
        POS[i][0] = central_point[0] + radius * np.cos(angles[i])
        POS[i][1] = central_point[1] + radius * np.sin(angles[i])


    sys.generate()
    sys.updateSys(DT)

    


    # Drawing
    screen.fill((255, 255, 255))
    draw_bodies_pygame()
    draw_bbox_pygame(sys.root)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()


        
