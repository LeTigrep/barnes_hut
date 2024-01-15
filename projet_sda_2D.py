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
       # Construction des nœuds d'arbre frais pour les corps
        self.reset()
        for x in range(self.N):
            # Pour chaque corps, ajoutez à la racine
            self.root.addBody(x,0)

    def updateSys(self, dt):
        self.calculateBodyAccels()
        global VEL
        VEL += ACC * dt

    def calculateBodyAccels(self):
        # Mettre à jour la table ACC en fonction du POS actuel
        for k in range(self.N):
            ACC[k] = self.calculateBodyAccel(k)
    def calculateBodyAccel(self, bodI):
        return self.calculateBodyAccelR(bodI, self.root)

    def calculateBodyAccelR(self, bodI, node):
        # Calculer l'accélération sur le corps I
        # La principale différence est que le corps est ignoré dans les calculs
        acc = zeros(2,dtype=float)
        if (node.leaf):
            # print "Leaf"
            # Nœud feuille, pas d'enfant
            for k in node.bods:
                if k != bodI: # Skip same body
                    acc += getForce( POS[bodI] ,1.0,POS[k],MASS[k])
        else:
            s = max( node.bbox.sideLength )
            d = node.center - POS[bodI]
            r = sqrt(d.dot(d))
            if (r > 0 and s/r < self.theta):
                # Assez loin pour faire une approximation
                acc += getForce( POS[bodI] ,1.0, node.com, node.mass)
            else:
                # Trop proche de l'approximation, récursion vers le bas de l'arbre
                for k in range(4):
                    if node.children[k] != None:
                        acc += self.calculateBodyAccelR(bodI, node.children[k])
        # print "ACC : %s" % acc
        return acc

        

def getForce(p1,m1,p2,m2):
    # need to d
    global NUM_CHECKS
    d = p2-p1
    r = sqrt(d.dot(d)) + ETA
    f = array( d * G*m1*m2 / r**3 )
    NUM_CHECKS += 1
    return f
