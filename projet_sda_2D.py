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
        
